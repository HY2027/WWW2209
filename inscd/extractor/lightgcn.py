import torch
import torch.nn as nn
import torch.nn.functional as F
from .._base import _Extractor
from sklearn.cluster import KMeans  # 添加聚类库


class LIGHTGCN_Extractor(_Extractor, nn.Module):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int, latent_dim: int, device,
                 dtype, gcn_layers=3, keep_prob=0.9):
        super().__init__()
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.latent_dim = latent_dim
        self.cluster_num = 4
        # self.low_degree_threshold = 1  # 低度节点阈值

        self.device = device
        self.dtype = dtype
        self.gcn_layers = gcn_layers
        self.keep_prob = keep_prob
        self.max_intra_edges = 2

        self.__student_emb = nn.Embedding(self.student_num, self.latent_dim, dtype=self.dtype).to(self.device)
        self.__knowledge_emb = nn.Embedding(self.knowledge_num, self.latent_dim, dtype=self.dtype).to(self.device)
        self.__exercise_emb = nn.Embedding(self.exercise_num, self.latent_dim, dtype=self.dtype).to(self.device)
        self.__disc_emb = nn.Embedding(self.exercise_num, 1, dtype=self.dtype).to(self.device)
        self.__emb_map = {
            "mastery": self.__student_emb.weight,
            "diff": self.__exercise_emb.weight,
            "disc": self.__disc_emb.weight,
            "knowledge": self.__knowledge_emb.weight
        }
        self.__knowledge_impact_emb = nn.Embedding(self.exercise_num, self.latent_dim, dtype=self.dtype).to(self.device)
        self.apply(self.initialize_weights)

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)

    def get_graph(self, graph):
        self.graph = graph

    def __cluster_students(self, student_emb):
        if student_emb.size(0) == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        kmeans = KMeans(n_clusters=self.cluster_num, random_state=0)
        labels = kmeans.fit_predict(student_emb.detach().cpu().numpy())
        return torch.tensor(labels, dtype=torch.long, device=self.device)

    def __dropout(self, graph, keep_prob):
        if self.training:
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

    # def __enhance_graph(self, student_emb, graph):
    #     if not self.training:
    #         return graph.coalesce()
        
    #     labels = self.__cluster_students(student_emb)
        
    #     # 确保图已合并
    #     graph = graph.coalesce()
    #     # 计算学生节点度数 (只取前student_num个节点)
    #     degrees = torch.sparse.sum(graph, dim=1).to_dense()[:self.student_num]
        
    #     # deg_np = degrees.cpu().numpy()
        
    #     # # 计算度数统计量
    #     # max_deg = deg_np.max().item()
    #     # min_deg = deg_np.min().item()
    #     # avg_deg = deg_np.mean().item()
    #     # median_deg = np.median(deg_np).item()
        
    #     # # 打印度数统计信息
    #     # print(f"[Pre-enhancement] Student degree stats: "
    #     #     f"Max={max_deg}, Min={min_deg}, "
    #     #     f"Avg={avg_deg:.2f}, Median={median_deg}")

    #     enhanced_graph = graph.clone()
    #     new_edges = []  # 存储要添加的新边
        
    #     # 遍历每个簇收集需要添加的边
    #     for cluster_id in range(self.cluster_num):
    #         cluster_students = torch.where(labels == cluster_id)[0]
    #         n_in_cluster = len(cluster_students)

    #         # 获取簇内节点的度数
    #         cluster_degrees = degrees[cluster_students]
            
    #         # 计算50%分界点（取节点数的一半）
    #         num_low_deg = max(1, n_in_cluster // 2)  # 至少取1个节点
            
    #         # 按度数升序排序，度数小的节点排名靠前
    #         sorted_indices = torch.argsort(cluster_degrees)
            
    #         # 选择排名前50%的节点（即度数最小的50%节点）
    #         low_deg_indices = sorted_indices[:num_low_deg]
    #         low_degree_students = cluster_students[low_deg_indices]
            
    #         # 为所有低度数学生生成候选邻居
    #         for s_idx in low_degree_students:
    #             other_students = cluster_students[cluster_students != s_idx]
    #             num_edges = min(self.max_intra_edges, len(other_students))
    #             selected = torch.randperm(len(other_students), device=self.device)[:num_edges]
    #             neighbors = other_students[selected]
                
    #             # 记录双向边
    #             for n_idx in neighbors:
    #                 new_edges.append((s_idx, n_idx))
    #                 new_edges.append((n_idx, s_idx))  # 无向图的对称边
        
    #     # 如果有新边，一次性添加到图
    #     if new_edges:
    #         edges_tensor = torch.tensor(new_edges, device=self.device).t()
    #         edge_values = torch.ones(edges_tensor.shape[1], device=self.device)
    #         new_edge_tensor = torch.sparse_coo_tensor(
    #             edges_tensor, 
    #             edge_values, 
    #             enhanced_graph.size(),
    #             device=self.device
    #         ).coalesce()
            
    #         enhanced_graph = (enhanced_graph + new_edge_tensor).coalesce()   
    #     return enhanced_graph

    def __common_forward(self):
        stu_emb = self.__student_emb.weight
        exer_emb = self.__exercise_emb.weight
        # en = self.__enhance_graph(stu_emb, self.graph)
        all_emb = torch.cat([stu_emb, exer_emb]).to(self.device)
        embs = [all_emb]
        # for layer in range(self.gcn_layers):
        #     if isinstance(self.graph, list):
        #         # g_droped = self.__graph_drop(self.graph[layer])
        #         all_emb = torch.sparse.mm(self.graph[layer], all_emb)
        #         embs.append(all_emb)
        #     else:
        #         # g_droped = self.__graph_drop(self.graph)
        #         all_emb = torch.sparse.mm(en, all_emb)
        #         embs.append(all_emb)
        for layer in range(self.gcn_layers):
            if isinstance(self.graph, list):
                # g_droped = self.__graph_drop(self.graph[layer])
                all_emb = torch.sparse.mm(self.graph[layer], all_emb)
                embs.append(all_emb)
            else:
                # g_droped = self.__graph_drop(self.graph)
                all_emb = torch.sparse.mm(self.graph, all_emb)
                embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        stus, exers = torch.split(light_out, [self.student_num, self.exercise_num])
        return stus, exers

    def extract(self, student_id, exercise_id, q_mask):
        stu_forward, exer_forward = self.__common_forward()


        student_ts = F.embedding(student_id, stu_forward)
        diff_ts = F.embedding(exercise_id, exer_forward)
        disc_ts = self.__disc_emb(exercise_id)
        knowledge_ts = self.__knowledge_emb.weight
        knowledge_impact_ts = self.__knowledge_impact_emb(exercise_id)
        return student_ts, diff_ts, disc_ts, knowledge_ts, {'knowledge_impact': knowledge_impact_ts}

    def __getitem__(self, item):
        if item not in self.__emb_map.keys():
            raise ValueError("We can only detach {} from embeddings.".format(self.__emb_map.keys()))
        stu_forward, exer_forward = self.__common_forward()
        disc_ts = self.__disc_emb.weight
        self.__emb_map["mastery"] = stu_forward
        self.__emb_map["diff"] = exer_forward
        self.__emb_map["disc"] = disc_ts
        return self.__emb_map[item]

    def get_all_emb(self):
        stu_emb, exer_emb, know_emb = (self.__student_emb.weight,
                                       self.__exercise_emb.weight,
                                       self.__knowledge_emb.weight)
        all_emb = torch.cat([stu_emb, exer_emb, know_emb]).to(self.device)
        return all_emb

    # def __noise(self, graph):
    #     if self.training:
    #         size = graph.size()
    #         indices = graph.indices().t()
    #         values = graph.values()
        
    #         original_edge_count = len(graph.values())  # 使用原始边数量计算
    #         noise_count = max(1, int(0.15 * original_edge_count))  # 至少 1 条
            
    #         # 随机生成噪声边坐标（允许重复）
    #         noise_rows = torch.randint(0, size[0], (noise_count,), device=indices.device)
    #         noise_cols = torch.randint(0, size[1], (noise_count,), device=indices.device)
    #         noise_indices = torch.stack([noise_rows, noise_cols], dim=1)
            
    #         # 生成噪声值（0~1 均匀分布）
    #         noise_values = torch.rand(noise_count, device=values.device)
            
    #         # 合并原始边和噪声边
    #         combined_indices = torch.cat([indices, noise_indices], dim=0)
    #         combined_values = torch.cat([values, noise_values], dim=0)
            
    #         # 重建稀疏张量（自动处理重复坐标的值相加）
    #         return torch.sparse.DoubleTensor(combined_indices.t(), combined_values, size)
    #     else:
    #         return graph


    # def __graph_drop(self, graph):
    #     g_dropped = self.__noise(graph)
    #     return g_dropped
    
    # def __graph_drop(self, graph):
    #     g_dropped = self.__dropout(graph, self.keep_prob)
    #     return g_dropped

    # def __common_forward(self):
    #     stu_emb = self.__student_emb.weight
    #     exer_emb = self.__exercise_emb.weight
    #     all_emb = torch.cat([stu_emb, exer_emb]).to(self.device)
    #     embs = [all_emb]
    #     for layer in range(self.gcn_layers):
    #         if isinstance(self.graph, list):
    #             g_droped = self.__graph_drop(self.graph[layer])
    #             all_emb = torch.sparse.mm(g_droped, all_emb)
    #             embs.append(all_emb)
    #         else:
    #             g_droped = self.__graph_drop(self.graph)
    #             all_emb = torch.sparse.mm(g_droped, all_emb)
    #             embs.append(all_emb)
    #     embs = torch.stack(embs, dim=1)
    #     light_out = torch.mean(embs, dim=1)
    #     stus, exers = torch.split(light_out, [self.student_num, self.exercise_num])
    #     return stus, exers