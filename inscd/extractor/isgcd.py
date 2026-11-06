import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .._base import _Extractor

import scipy.sparse as sp


class ISGCD_Extractor(_Extractor, nn.Module):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int, latent_dim: int, device,
                 dtype, gcn_layers=3, keep_prob=0.9, mode='all', ssl_temp=0.8, ssl_weight=1e-2, **kwargs):
        super().__init__()
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.latent_dim = latent_dim
        self.mode = mode
        self.ssl_temp = ssl_temp
        self.ssl_weight = ssl_weight

        self.device = device
        self.dtype = dtype
        self.gcn_layers = gcn_layers
        self.keep_prob = keep_prob
        self.gcn_drop = True
        self.graph_dict = ...

        self.__student_emb = nn.Embedding(self.student_num, self.latent_dim, dtype=self.dtype).to(self.device)
        self.__knowledge_emb = nn.Embedding(self.knowledge_num, self.latent_dim, dtype=self.dtype).to(self.device)
        self.__exercise_emb = nn.Embedding(self.exercise_num, self.latent_dim, dtype=self.dtype).to(self.device)
        self.__disc_emb = nn.Embedding(self.exercise_num, 1, dtype=self.dtype).to(self.device)
        self.__knowledge_impact_emb = nn.Embedding(self.exercise_num, self.latent_dim, dtype=self.dtype).to(self.device)

        self.__emb_map = {
            "mastery": self.__student_emb.weight,
            "diff": self.__exercise_emb.weight,
            "disc": self.__disc_emb.weight,
            "knowledge": self.__knowledge_emb.weight
        }

        self.concat_layer = nn.Linear(2 * self.latent_dim, self.latent_dim, dtype=self.dtype).to(self.device)
        self.transfer_student_layer = nn.Linear(self.latent_dim, self.knowledge_num, dtype=self.dtype).to(self.device)
        self.transfer_exercise_layer = nn.Linear(self.latent_dim, self.knowledge_num, dtype=self.dtype).to(self.device)
        self.transfer_knowledge_layer = nn.Linear(self.latent_dim, self.knowledge_num, dtype=self.dtype).to(self.device)
        self.apply(self.initialize_weights)

        self.sibling_regression = nn.Linear(self.latent_dim, self.latent_dim, dtype=self.dtype).to(self.device)


        self.stu_exer_mapper = nn.Linear(
            self.exercise_num, 
            self.student_num, 
            dtype=self.dtype
        ).to(self.device)

    def map_exer_to_student_dim(self, exer_emb):
        return self.stu_exer_mapper(exer_emb.t()).t()  # [student_num, latent_dim]

    def __denoise_student(self, student_emb, exer_emb):
        exer_emb = self.map_exer_to_student_dim(exer_emb)
        noise_component = self.sibling_regression(exer_emb)  

        purified_emb = student_emb - noise_component

        return purified_emb


    def get_graph_dict(self, graph_dict):
        self.graph_dict = graph_dict

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)

    def get_all_emb(self):
        stu_emb, exer_emb, know_emb = (self.__student_emb.weight,
                                       self.__exercise_emb.weight,
                                       self.__knowledge_emb.weight)
        all_emb = torch.cat([stu_emb, exer_emb, know_emb]).to(self.device)
        return all_emb

    def convolution(self, graph):
        all_emb = self.get_all_emb()
        emb = [all_emb]
        for layer in range(self.gcn_layers):
            all_emb = torch.sparse.mm(self.__graph_drop(graph), all_emb)
            emb.append(all_emb)
        out_emb = torch.mean(torch.stack(emb, dim=1), dim=1)
        return out_emb

    def __common_forward(self, right, wrong):
        all_emb = self.get_all_emb()
        emb = [all_emb]
        right_emb = all_emb
        wrong_emb = all_emb
        for layer in range(self.gcn_layers):
            right_emb = torch.sparse.mm(right, right_emb)
            wrong_emb = torch.sparse.mm(wrong, wrong_emb)
            all_emb = self.concat_layer(torch.cat([right_emb, wrong_emb], dim=1))
            emb.append(all_emb)
        out_emb = torch.mean(torch.stack(emb, dim=1), dim=1)
        return out_emb[:self.student_num], out_emb[self.student_num:self.student_num + self.exercise_num], out_emb[
                                                                                                           self.exercise_num + self.student_num:]

    # def __common_forward(self, right, wrong):
    #     right_emb, wrong_emb = self.convolution(right), self.convolution(wrong)
    #     out = self.concat_layer(torch.cat([right_emb, wrong_emb], dim=1))
    #     return out[:self.student_num], out[self.student_num:self.student_num + self.exercise_num], out[
    #                                                                                                self.exercise_num + self.student_num:]

    def __dropout(self, graph, keep_prob, node_embeddings):
        """基于节点相似性的结构化去噪"""
        if self.gcn_drop and self.training:
            # 获取稀疏矩阵的元信息
            indices = graph.indices().t()
            values = graph.values()
            size = graph.size()
            
            # 提取连接节点的嵌入表示
            src_nodes = indices[:, 0]  # 源节点索引
            tgt_nodes = indices[:, 1]  # 目标节点索引
            src_emb = node_embeddings[src_nodes]  # [num_edges, latent_dim]
            tgt_emb = node_embeddings[tgt_nodes]  # [num_edges, latent_dim]

            # 计算边的重要性得分
            similarity = torch.sigmoid(torch.sum(src_emb * tgt_emb, dim=1))  # [num_edges]
            
            # 生成动态保留概率
            dynamic_probs = keep_prob * similarity  # 基础概率×相似度
            
            # 伯努利采样
            mask = torch.bernoulli(dynamic_probs).bool()
            
            # 应用掩码
            filtered_indices = indices[mask]
            filtered_values = values[mask] / dynamic_probs[mask]  # 重要性加权

            # 重建稀疏矩阵
            return torch.sparse.DoubleTensor(
                filtered_indices.t(), 
                filtered_values, 
                size
            ).to(self.device)
        else:
            return graph


    def __noise(self, graph):
        if self.training:
            size = graph.size()
            indices = graph.indices().t()
            values = graph.values()
        
            original_edge_count = len(graph.values())  # 使用原始边数量计算
            noise_count = max(1, int(0.15 * original_edge_count))  # 至少 1 条
            
            # 随机生成噪声边坐标（允许重复）
            noise_rows = torch.randint(0, size[0], (noise_count,), device=indices.device)
            noise_cols = torch.randint(0, size[1], (noise_count,), device=indices.device)
            noise_indices = torch.stack([noise_rows, noise_cols], dim=1)
            
            # 生成噪声值（0~1 均匀分布）
            noise_values = torch.rand(noise_count, device=values.device)
            
            # 合并原始边和噪声边
            combined_indices = torch.cat([indices, noise_indices], dim=0)
            combined_values = torch.cat([values, noise_values], dim=0)
            
            # 重建稀疏张量（自动处理重复坐标的值相加）
            return torch.sparse_coo_tensor(
                combined_indices.t(), 
                combined_values, 
                size,
                dtype=values.dtype,
                device=self.device
            ).coalesce()
        else:
            return graph


    def __graph_add(self, graph):
        g_dropped = self.__noise(graph)
        return g_dropped

    def __graph_denoise(self, graph, is_right=True):
        all_emb = self.get_all_emb()
    
        return self.__dropout(graph, self.keep_prob, all_emb)

    def hsic(self, X, Y):
        n = X.size(0)
        H = torch.eye(n, device=X.device) - 1/n  # 中心化矩阵
        K = self._rbf_kernel(X, 0.2)
        L = self._rbf_kernel(Y, 0.2)
        
        # 计算HSIC
        K = K.float()
        L = L.float()
        H = H.float()
        
        HKH = torch.mm(torch.mm(K, H), L).mm(H)
        return torch.trace(HKH) / ((n-1)**2)
    
    def _rbf_kernel(self, X, alpha=0.2):
        """
        RBF核函数
        """
        # 计算成对距离
        dist_sq = torch.cdist(X, X, p=2)**2
        return torch.exp(-dist_sq / (2 * alpha))
    
    def mutual_information_loss(self, stu_reliable, exer_reliable, stu_original, exer_original):
  
        hsic_stu = self.hsic(stu_reliable, stu_original)
        hsic_exer = self.hsic(exer_reliable, exer_original)
        
        return hsic_stu + hsic_exer

    def extract(self, student_id, exercise_id, q_mask):
        if 'dis' not in self.mode:
            # right = self.__graph_add(self.graph_dict['right'])
            # wrong = self.__graph_add(self.graph_dict['wrong'])
            right = self.graph_dict['right']
            wrong = self.graph_dict['wrong']
            stu_forward, exer_forward, know_forward = self.__common_forward(right,
                                                                            wrong)
            
            # stu_forward = self.__denoise_student(stu_forward, exer_forward)
            right = self.__graph_denoise(right)
            wrong = self.__graph_denoise(wrong)
            stu_forward_flip, exer_forward_flip, know_forward_flip = self.__common_forward(
                right, wrong)
        else:
            out = self.convolution(self.graph_dict['all'])
            stu_forward, exer_forward, know_forward = out[:self.student_num], out[
                                                                              self.student_num:self.student_num + self.exercise_num], out[
                                                                                                                                      self.exercise_num + self.student_num:]

        # stu_forward_flip_v1, exer_forward_flip_v1, know_forward_flip_v1 = self.__common_forward(self.graph_dict['right_v1'],
        #                                                                                self.graph_dict['wrong_v1'])
        # stu_forward_flip_v2, exer_forward_flip_v2, know_forward_flip_v2 = self.__common_forward(self.graph_dict['right_v2'],
        #                                                                                self.graph_dict['wrong_v2'])
        #
        extra_loss = 0

        def InfoNCE(view1, view2, temperature: float = 1.0, b_cos: bool = False):
            """
            Args:
                view1: (torch.Tensor - N x D)
                view2: (torch.Tensor - N x D)
                temperature: float
                b_cos (bool)

            Return: Average InfoNCE Loss
            """
            if b_cos:
                view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

            pos_score = (view1 @ view2.T) / temperature
            score = torch.diag(F.log_softmax(pos_score, dim=1))
            return -score.mean()

        if 'cl' not in self.mode:
            extra_loss = self.ssl_weight * (InfoNCE(stu_forward, stu_forward_flip, temperature=self.ssl_temp)
                                            + InfoNCE(exer_forward, exer_forward_flip,
                                                      temperature=self.ssl_temp))
            # print(extra_loss)
        # original_emb = torch.cat([stu_forward, exer_forward], dim=0)
        # reliable_emb = torch.cat([stu_forward_flip, exer_forward_flip], dim=0)
        # hsic_loss = self.mutual_information_loss(stu_forward_flip, exer_forward_flip, stu_forward, exer_forward)
        # extra_loss = extra_loss + 0.5 * hsic_loss
        student_ts = self.transfer_student_layer(
            F.embedding(student_id, stu_forward)) if 'tf' not in self.mode else F.embedding(student_id, stu_forward)
        diff_ts = self.transfer_exercise_layer(
            F.embedding(exercise_id, exer_forward)) if 'tf' not in self.mode else F.embedding(exercise_id, exer_forward)
        knowledge_ts = self.transfer_knowledge_layer(know_forward) if 'tf' not in self.mode else know_forward

        disc_ts = self.__disc_emb(exercise_id)
        knowledge_impact_ts = self.__knowledge_impact_emb(exercise_id)
        return student_ts, diff_ts, disc_ts, knowledge_ts, {'extra_loss': extra_loss,
                                                            'knowledge_impact': knowledge_impact_ts}

    def get_flip_graph(self):
        def get_flip_data(data):
            import numpy as np
            np_response_flip = data.copy()
            column = np_response_flip[:, 2]
            probability = np.random.choice([True, False], size=column.shape,
                                           p=[self.graph_dict['flip_ratio'], 1 - self.graph_dict['flip_ratio']])
            column[probability] = 1 - column[probability]
            np_response_flip[:, 2] = column
            return np_response_flip

        response_flip = get_flip_data(self.graph_dict['response'])
        se_graph_right_flip, se_graph_wrong_flip = [self.__create_adj_se(response_flip, is_subgraph=True)[i] for i in
                                                    range(2)]
        ek_graph = self.graph_dict['Q_Matrix']
        self.graph_dict['right_flip'], self.graph_dict['wrong_flip'] = self.__final_graph(se_graph_right_flip,
                                                                                          ek_graph), self.__final_graph(
            se_graph_wrong_flip, ek_graph)

    def __getitem__(self, item):
        if item not in self.__emb_map.keys():
            raise ValueError("We can only detach {} from embeddings.".format(self.__emb_map.keys()))
        if 'dis' not in self.mode:
            stu_forward, exer_forward, know_forward = self.__common_forward(self.graph_dict['right'],
                                                                            self.graph_dict['wrong'])
        else:
            out = self.convolution(self.graph_dict['all'])
            stu_forward, exer_forward, know_forward = (out[:self.student_num],
                                                       out[self.student_num:self.student_num + self.exercise_num],
                                                       out[self.exercise_num + self.student_num:])

        student_ts = self.transfer_student_layer(stu_forward) if 'tf' not in self.mode else stu_forward
        diff_ts = self.transfer_exercise_layer(exer_forward) if 'tf' not in self.mode else exer_forward
        knowledge_ts = self.transfer_knowledge_layer(know_forward) if 'tf' not in self.mode else know_forward

        disc_ts = self.__disc_emb.weight
        self.__emb_map["mastery"] = student_ts
        self.__emb_map["diff"] = diff_ts
        self.__emb_map["disc"] = disc_ts
        self.__emb_map["knowledge"] = knowledge_ts
        return self.__emb_map[item]

    @staticmethod
    def __get_csr(rows, cols, shape):
        values = np.ones_like(rows, dtype=np.float64)
        return sp.csr_matrix((values, (rows, cols)), shape=shape)

    @staticmethod
    def __sp_mat_to_sp_tensor(sp_mat):
        coo = sp_mat.tocoo().astype(np.float64)
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()

    def __create_adj_se(self, np_response, is_subgraph=False):
        if is_subgraph:
            if self.mode == 'R':
                return np.zeros(shape=(self.student_num, self.exercise_num)), np.zeros(
                    shape=(self.student_num, self.exercise_num))

            train_stu_right = np_response[np_response[:, 2] == 1, 0]
            train_exer_right = np_response[np_response[:, 2] == 1, 1]
            train_stu_wrong = np_response[np_response[:, 2] == 0, 0]
            train_exer_wrong = np_response[np_response[:, 2] == 0, 1]

            adj_se_right = self.__get_csr(train_stu_right, train_exer_right,
                                          shape=(self.student_num, self.exercise_num))
            adj_se_wrong = self.__get_csr(train_stu_wrong, train_exer_wrong,
                                          shape=(self.student_num, self.exercise_num))
            return adj_se_right.toarray(), adj_se_wrong.toarray()

        else:
            if self.mode == 'R':
                return np.zeros(shape=(self.student_num, self.exercise_num))
            response_stu = np_response[:, 0]
            response_exer = np_response[:, 1]
            adj_se = self.__get_csr(response_stu, response_exer, shape=(self.student_num, self.exercise_num))
            return adj_se.toarray()

    def __final_graph(self, se, ek):
        sek_num = self.student_num + self.exercise_num + self.knowledge_num
        se_num = self.student_num + self.exercise_num
        tmp = np.zeros(shape=(sek_num, sek_num))
        tmp[:self.student_num, self.student_num: se_num] = se
        tmp[self.student_num:se_num, se_num:sek_num] = ek
        graph = tmp + tmp.T + np.identity(sek_num)
        graph = sp.csr_matrix(graph)

        rowsum = np.array(graph.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(graph)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return self.__sp_mat_to_sp_tensor(adj_matrix).to(self.device)


def inner_product(a, b):
    return torch.sum(a * b, dim=-1)
