import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .._base import _Extractor
import scipy.sparse as sp
from sklearn.cluster import KMeans  # 添加聚类库
import ast

class Diff_Extractor(_Extractor, nn.Module):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int, latent_dim: int, device,
                 dtype, gcn_layers=3, keep_prob=0.9, mode='all', ssl_temp=0.8, ssl_weight=1e-2, 
                 cluster_num=1, low_degree_threshold=2, diffusion_steps=100, 
                 unconditional_weight=0.5, conditional_weight=0.5, **kwargs):  # 添加聚类超参数
        super().__init__()
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.latent_dim = latent_dim
        self.mode = mode
        self.ssl_temp = ssl_temp
        self.ssl_weight = ssl_weight
        self.cluster_num = cluster_num  
        self.low_degree_threshold = low_degree_threshold  
        self.diffusion_steps = diffusion_steps
        self.unconditional_weight = unconditional_weight
        self.conditional_weight = conditional_weight
        self.device = device
        self.dtype = dtype
        self.gcn_layers = gcn_layers
        self.keep_prob = keep_prob
        self.gcn_drop = True
        self.graph_dict = ...
        self.max_intra_edges = 2

        self.__student_emb = nn.Embedding(self.student_num, self.latent_dim, dtype=self.dtype).to(self.device)
        self.__knowledge_emb = nn.Embedding(self.knowledge_num, self.latent_dim, dtype=self.dtype).to(self.device)
        self.__exercise_emb = nn.Embedding(self.exercise_num, self.latent_dim, dtype=self.dtype).to(self.device)
        self.__disc_emb = nn.Embedding(self.exercise_num, 1, dtype=self.dtype).to(self.device)
        self.__knowledge_impact_emb = nn.Embedding(self.exercise_num, self.latent_dim, dtype=self.dtype).to(self.device)
        llm_stu_emb = []
        llm_exer_emb = []

        # with open("datasets/Assist0910/stu_emb") as f:
        # with open("datasets/Junyi1/stu_emb") as f:

        with open("datasets/Assist0910/stu_emb") as f:
            for line in f:
                # line = '['+line+']'
                # line = '['+line+']'
                line = line.strip()
                if not line:
                    continue
                vec = ast.literal_eval(line) 
                llm_stu_emb.append(vec)
        with open("datasets/Assist0910/exer_emb") as f:
            for line in f:
                # line = '['+line+']'
                # line = '['+line+']'
                line = line.strip()
                if not line:
                    continue
                vec = ast.literal_eval(line) 
                llm_exer_emb.append(vec)
        llm_weight_matrix_stu = np.array(llm_stu_emb, dtype=np.float64)
        llm_weight_matrix_exer = np.array(llm_exer_emb, dtype=np.float64)
        
        # 确保张量在正确的设备上
        llm_stu_embedding = nn.Embedding(self.student_num, self.latent_dim).to(self.device)
        llm_exer_embedding = nn.Embedding(self.exercise_num, self.latent_dim).to(self.device)
        
        llm_weight_tensor_stu = torch.from_numpy(llm_weight_matrix_stu).to(dtype=self.dtype).to(self.device)
        llm_weight_tensor_exer = torch.from_numpy(llm_weight_matrix_exer).to(dtype=self.dtype).to(self.device)
       
        llm_stu_embedding.weight = nn.Parameter(llm_weight_tensor_stu)
        llm_exer_embedding.weight = nn.Parameter(llm_weight_tensor_exer)

        self.__llm_stu_emb = llm_stu_embedding.weight
        self.__llm_exer_emb = llm_exer_embedding.weight
        
        # 定义扩散模型组件
        self.__init_diffusion_components()
        self.__emb_map = {
            "mastery": self.__student_emb.weight,
            "diff": self.__exercise_emb.weight,
            "disc": self.__disc_emb.weight,
            "knowledge": self.__knowledge_emb.weight
        }
        self.intra_edge_prob = 0.5

        self.concat_layer = nn.Linear(2 * self.latent_dim, self.latent_dim, dtype=self.dtype).to(self.device)

        self.transfer_student_layer = nn.Linear(self.latent_dim, self.knowledge_num, dtype=self.dtype).to(self.device)
        self.transfer_exercise_layer = nn.Linear(self.latent_dim, self.knowledge_num, dtype=self.dtype).to(self.device)
        self.transfer_knowledge_layer = nn.Linear(self.latent_dim, self.knowledge_num, dtype=self.dtype).to(self.device)
        self.apply(self.initialize_weights)


    def __init_diffusion_components(self):
        """初始化扩散模型组件"""
        # 噪声调度参数 - 确保在正确设备上
        self.betas = torch.linspace(1e-4, 0.02, self.diffusion_steps).to(self.device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # 无条件扩散网络
        self.unconditional_denoiser = nn.Sequential(
            nn.Linear(self.latent_dim + 64, 256, dtype=self.dtype),  # 64是时间步嵌入维度
            nn.Tanh(),
            nn.LayerNorm(256, dtype=self.dtype),
            nn.Linear(256, self.latent_dim, dtype=self.dtype)
        ).to(self.device)
        
        # 条件扩散网络
        self.conditional_denoiser = nn.Sequential(
            nn.Linear(2 * self.latent_dim + 64, 256, dtype=self.dtype),  # 条件信息拼接
            nn.Tanh(),
            nn.LayerNorm(256, dtype=self.dtype),
            nn.Linear(256, self.latent_dim, dtype=self.dtype)
        ).to(self.device)
        
        # 时间步嵌入
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 64, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(64, 64, dtype=self.dtype)
        ).to(self.device)
        
        # 条件特征提取器
        self.condition_extractor = nn.Sequential(
            nn.Linear(self.latent_dim, 128, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(128, self.latent_dim, dtype=self.dtype)
        ).to(self.device)

    def __time_embed(self, t):
        """时间步嵌入"""
        t = t.double()
        return self.time_embedding(t.view(-1, 1))

    def __forward_diffusion(self, x0, t):
        """前向扩散过程"""
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).view(-1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t]).view(-1, 1)
        noise = torch.randn_like(x0, device=self.device)  # 确保噪声在正确设备上
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return xt, noise

    def __unconditional_reverse_diffusion(self, xt, t):
        """无条件反向扩散"""
        time_emb = self.__time_embed(t)
        return self.unconditional_denoiser(torch.cat([xt, time_emb], dim=-1).double())

    def __conditional_reverse_diffusion(self, xt, condition, t):
        """条件反向扩散"""
        time_emb = self.__time_embed(t)
        return self.conditional_denoiser(torch.cat([xt, condition, time_emb], dim=-1).double())

    def __two_stage_diffusion(self, auxiliary_features, collaborative_features):
        """双阶段扩散过程"""
        batch_size = auxiliary_features.size(0)
        
        # 第一阶段：无条件扩散
        t_uncond = torch.randint(0, self.diffusion_steps, (batch_size,), device=self.device)  # 确保在正确设备上
        xt_uncond, noise_uncond = self.__forward_diffusion(auxiliary_features.double(), t_uncond)
        predicted_noise_uncond = self.__unconditional_reverse_diffusion(xt_uncond.double(), t_uncond)
        unconditional_loss = F.mse_loss(predicted_noise_uncond, noise_uncond)
        
        # 第二阶段：条件扩散
        t_cond = torch.randint(0, self.diffusion_steps, (batch_size,), device=self.device)  # 确保在正确设备上
        xt_cond, noise_cond = self.__forward_diffusion(auxiliary_features, t_cond)
        
        # 提取条件特征
        condition_features = self.condition_extractor(collaborative_features.double())
        
        predicted_noise_cond = self.__conditional_reverse_diffusion(xt_cond.double(), condition_features, t_cond.double())
        conditional_loss = F.mse_loss(predicted_noise_cond, noise_cond)
        
        # 总扩散损失
        diffusion_loss = self.unconditional_weight * unconditional_loss + self.conditional_weight * conditional_loss
        
        # 推理时生成去噪特征
        if not self.training:
            # 使用多步去噪生成最终特征
            denoised_features = self.__diffusion_inference(auxiliary_features, condition_features)
            return denoised_features, diffusion_loss
        
        return auxiliary_features, diffusion_loss

    def __diffusion_inference(self, auxiliary_features, condition_features):
        """扩散模型推理过程"""
        # 使用多步去噪生成最终特征
        xt = auxiliary_features
        for t in range(self.diffusion_steps-1, -1, -1):
            t_batch = torch.full((xt.size(0),), t, device=self.device, dtype=torch.float64)  # 确保在正确设备上
            
            if t > 0:
                noise_pred = self.__conditional_reverse_diffusion(xt.double(), condition_features.double(), t_batch)
                xt = (xt - (self.betas[t] / torch.sqrt(1 - self.alpha_bars[t])) * noise_pred) / torch.sqrt(self.alphas[t])
                xt += torch.sqrt(self.betas[t]) * torch.randn_like(xt, device=self.device)  # 确保噪声在正确设备上
            else:
                noise_pred = self.__conditional_reverse_diffusion(xt.double(), condition_features.double(), t_batch)
                xt = (xt - (self.betas[t] / torch.sqrt(1 - self.alpha_bars[t])) * noise_pred) / torch.sqrt(self.alphas[t])
        
        return xt
    # 新增：学生聚类方法
    def __cluster_students(self, student_emb):
        if student_emb.size(0) == 0:
            return torch.empty(0, dtype=torch.float64, device=self.device)
        kmeans = KMeans(n_clusters=self.cluster_num, random_state=0)
        labels = kmeans.fit_predict(student_emb.detach().cpu().numpy())
        return torch.tensor(labels, dtype=torch.float64, device=self.device)


    def __enhance_graph(self, student_emb, graph):
        if not self.training:
            return graph.coalesce()
        
        labels = self.__cluster_students(student_emb)
        
        # 确保图已合并
        graph = graph.coalesce()
        # 计算学生节点度数 (只取前student_num个节点)
        degrees = torch.sparse.sum(graph, dim=1).to_dense()[:self.student_num]

        enhanced_graph = graph.clone()
        new_edges = []  # 存储要添加的新边
        
        # 遍历每个簇收集需要添加的边
        for cluster_id in range(self.cluster_num):
            cluster_students = torch.where(labels == cluster_id)[0]
            n_in_cluster = len(cluster_students)

            # 获取簇内节点的度数
            cluster_degrees = degrees[cluster_students]
            
            # 计算50%分界点（取节点数的一半）
            num_low_deg = max(1, n_in_cluster // 2)  # 至少取1个节点
            
            # 按度数升序排序，度数小的节点排名靠前
            sorted_indices = torch.argsort(cluster_degrees)
            
            # 选择排名前50%的节点（即度数最小的50%节点）
            low_deg_indices = sorted_indices[:num_low_deg]
            low_degree_students = cluster_students[low_deg_indices]
            
            # 为所有低度数学生生成候选邻居
            for s_idx in low_degree_students:
                other_students = cluster_students[cluster_students != s_idx]
                num_edges = min(self.max_intra_edges, len(other_students))
                selected = torch.randperm(len(other_students), device=self.device)[:num_edges]
                neighbors = other_students[selected]
                
                # 记录双向边
                for n_idx in neighbors:
                    new_edges.append((s_idx, n_idx))
                    new_edges.append((n_idx, s_idx))  # 无向图的对称边
        
        # 如果有新边，一次性添加到图
        if new_edges:
            edges_tensor = torch.tensor(new_edges, device=self.device).t()
            edge_values = torch.ones(edges_tensor.shape[1], device=self.device)
            new_edge_tensor = torch.sparse_coo_tensor(
                edges_tensor, 
                edge_values, 
                enhanced_graph.size(),
                device=self.device
            ).coalesce()
            
            enhanced_graph = (enhanced_graph + new_edge_tensor).coalesce()   
        return enhanced_graph

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
        # 新增：获取学生嵌入并增强连通性
        # student_emb = self.get_all_emb()[:self.student_num]
        # right = self.__enhance_graph(student_emb, right)
        # wrong = self.__enhance_graph(student_emb, wrong)
        all_emb = self.get_all_emb()
        emb = [all_emb]
        right_emb = all_emb
        wrong_emb = all_emb
        for layer in range(self.gcn_layers):
            right_emb = torch.sparse.mm(self.__graph_drop(right), right_emb)
            wrong_emb = torch.sparse.mm(self.__graph_drop(wrong), wrong_emb)
            all_emb = self.concat_layer(torch.cat([right_emb, wrong_emb], dim=1))
            emb.append(all_emb)
        out_emb = torch.mean(torch.stack(emb, dim=1), dim=1)
        return out_emb[:self.student_num], out_emb[self.student_num:self.student_num + self.exercise_num], out_emb[
                                                                                                           self.exercise_num + self.student_num:]


    def __common_forward1(self, right):
        all_emb = self.get_all_emb()
        emb = [all_emb]
        for layer in range(self.gcn_layers):
            aemb = torch.sparse.mm(self.__graph_drop(right), all_emb)
            emb.append(aemb)
        out_emb = torch.mean(torch.stack(emb, dim=1), dim=1)
        return out_emb[:self.student_num], out_emb[self.student_num:self.student_num + self.exercise_num], out_emb[
                                                                                                           self.exercise_num + self.student_num:]


    def __dropout(self, graph, keep_prob):
        if self.gcn_drop and self.training:
            size = graph.size()
            index = graph.indices().t()
            values = graph.values()
            random_index = torch.rand(len(values)) + keep_prob
            random_index = random_index.int().bool()
            index = index[random_index]
            values = values[random_index] / keep_prob
            g = torch.sparse.DoubleTensor(index.t(), values, size)
            return g
        else:
            return graph

    def __graph_drop(self, graph):
        g_dropped = self.__dropout(graph, self.keep_prob)
        return g_dropped

    def extract(self, student_id, exercise_id, q_mask):
        if 'dis' not in self.mode:
            # stu_forward, exer_forward, know_forward = self.__common_forward(self.graph_dict['right'],
            #                                                                 self.graph_dict['wrong'])
            # student_emb = stu_forward
            # right_en = self.__enhance_graph(student_emb, self.graph_dict['right'])
            # wrong_en = self.__enhance_graph(student_emb, self.graph_dict['wrong'])
            # stu_forward_en, exer_forward_en, know_forward_en = self.__common_forward(
            #     right_en,
            #     wrong_en)

            right_stu_forward, right_exer_forward, right_know_forward = self.__common_forward1(self.graph_dict['right'])
            wrong_stu_forward, wrong_exer_forward, wrong_know_forward = self.__common_forward1(self.graph_dict['wrong'])
            stu_forward = self.concat_layer(torch.cat([right_stu_forward, wrong_stu_forward], dim=1))
            exer_forward = self.concat_layer(torch.cat([right_exer_forward, wrong_exer_forward], dim=1))
            know_forward = self.concat_layer(torch.cat([right_exer_forward, wrong_exer_forward], dim=1))

            right_en = self.__enhance_graph(right_stu_forward, self.graph_dict['right'])
            wrong_en = self.__enhance_graph(wrong_stu_forward, self.graph_dict['wrong'])
            right_stu_forward_en, right_exer_forward_en, right_know_forward_en = self.__common_forward1(right_en)
            wrong_stu_forward_en, wrong_exer_forward_en, wrong_know_forward_en = self.__common_forward1(wrong_en)
            # stu_forward_en = self.concat_layer(torch.cat([right_stu_forward_en, wrong_stu_forward_en], dim=1))
            # exer_forward_en = self.concat_layer(torch.cat([right_exer_forward_en, wrong_exer_forward_en], dim=1))
            # know_forward_en = self.concat_layer(torch.cat([right_exer_forward_en, wrong_exer_forward_en], dim=1))
        else:
            out = self.convolution(self.graph_dict['all'])
            stu_forward, exer_forward, know_forward = out[:self.student_num], out[
                                                                              self.student_num:self.student_num + self.exercise_num], out[
                                                                                                                                      self.exercise_num + self.student_num:]


        extra_loss = 0
        diffusion_loss = 0

        llm_stu_features = self.__llm_stu_emb[student_id]
        llm_exer_features = self.__llm_exer_emb[exercise_id]
        right_stu_features = right_stu_forward[student_id]
        wrong_stu_features = wrong_stu_forward[student_id]
        right_exer_features = right_exer_forward[exercise_id]
        wrong_exer_features = wrong_exer_forward[exercise_id]
        
        right_stu_features_en = right_stu_forward_en[student_id]
        wrong_stu_features_en = wrong_stu_forward_en[student_id]
        right_exer_features_en = right_exer_forward_en[exercise_id]
        wrong_exer_features_en = wrong_exer_forward_en[exercise_id]

        diffused_llm_stu_features, stu_diff_loss = self.__two_stage_diffusion(
            llm_stu_features, 
            stu_forward[student_id]
        )
        
        diffused_llm_exer_features, exer_diff_loss = self.__two_stage_diffusion(
            llm_exer_features,
            exer_forward[exercise_id]
        )

        diffused_right_stu_features, right_stu_diff_loss = self.__two_stage_diffusion(
            right_stu_features_en, 
            right_stu_forward[student_id]
        )

        diffused_wrong_stu_features, wrong_stu_diff_loss = self.__two_stage_diffusion(
            wrong_stu_features_en, 
            wrong_stu_forward[student_id]
        )
        
        diffused_right_exer_features, right_exer_diff_loss = self.__two_stage_diffusion(
            right_exer_features_en, 
            right_exer_forward[exercise_id]
        )

        diffused_wrong_exer_features, wrong_exer_diff_loss = self.__two_stage_diffusion(
            wrong_exer_features_en, 
            wrong_exer_forward[exercise_id]
        )

        
        diffusion_loss = stu_diff_loss + exer_diff_loss + right_stu_diff_loss + wrong_stu_diff_loss + right_exer_diff_loss + wrong_exer_diff_loss

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
            extra_loss = self.ssl_weight * (InfoNCE(stu_forward[student_id], diffused_llm_stu_features, temperature=self.ssl_temp) +
                                           InfoNCE(exer_forward[exercise_id], diffused_llm_exer_features, temperature=self.ssl_temp))
            # extra_loss += self.ssl_weight * (InfoNCE(stu_forward[student_id], stu_forward_en[student_id], temperature=self.ssl_temp) +
            #                                 InfoNCE(exer_forward[exercise_id], exer_forward_en[exercise_id], temperature=self.ssl_temp))
            extra_loss += self.ssl_weight * (InfoNCE(right_stu_forward[student_id], diffused_right_stu_features, temperature=self.ssl_temp) +
                                            InfoNCE(right_exer_forward[exercise_id], diffused_right_exer_features, temperature=self.ssl_temp))
            extra_loss += self.ssl_weight * (InfoNCE(wrong_stu_forward[student_id], diffused_wrong_stu_features, temperature=self.ssl_temp) +
                                            InfoNCE(wrong_exer_forward[exercise_id], diffused_wrong_exer_features, temperature=self.ssl_temp))
            # print(extra_loss)
            extra_loss += self.ssl_weight * diffusion_loss
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