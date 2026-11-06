import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp

from ...._base import _CognitiveDiagnosisModel
from ....datahub import DataHub
from ....interfunc import NCD_IF, DP_IF, MIRT_IF, MF_IF, KANCD_IF, CDMFKC_IF, IRT_IF, KSCD_IF
from ....extractor import LIGHTGCN_Extractor
import matplotlib.pyplot as plt
import os

# def compute_and_save_singular_values_for_adj(adj_matrix, save_path, k=100):
#     """
#     对异构图邻接矩阵进行SVD分解并保存奇异值
    
#     参数:
#     adj_matrix -- 稀疏邻接矩阵张量
#     save_path -- 保存路径
#     k -- 保留的奇异值数量
#     """
#     print(f"Starting SVD for adjacency matrix of shape {adj_matrix.size()}...")
    
#     # 计算Frobenius范数作为总方差
#     frob_norm_sq = torch.sum(adj_matrix.values()**2).item()
    
#     # 使用partial SVD计算前k个奇异值
#     with torch.no_grad():
#         if min(adj_matrix.size()) <= k:
#             k = min(adj_matrix.size()) - 1
#             print(f"Reduced k to {k} due to matrix size constraints")
        
#         U, S, V = torch.svd_lowrank(adj_matrix, q=k, niter=5)
#         singular_values = S.cpu().numpy()
#         right_singular_vectors = V.cpu()
#         left_singular_vectors = U.cpu()

    
#     # 计算解释方差比例和累积值
#     explained_variance = singular_values ** 2
#     cumulative_explained = np.cumsum(explained_variance)
#     cumulative_perc = cumulative_explained / frob_norm_sq
    
#     # 确保目录存在
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
#     # 保存结果
#     with open(save_path, 'w') as f:
#         # 奇异值
#         f.write("Top Singular Values:\n")
#         for i, sv in enumerate(singular_values):
#             f.write(f"Component {i+1}: {sv:.6f}\n")
        
#         # 解释方差比例
#         f.write("\nExplained Variance Proportion:\n")
#         for i, (ev, cp) in enumerate(zip(explained_variance/frob_norm_sq, cumulative_perc)):
#             f.write(f"Component {i+1}: {ev:.6f} | Cum: {cp:.4f} ({cp*100:.2f}%)\n")
        
#         # 元数据
#         f.write("\nMetadata:\n")
#         f.write(f"Matrix Shape: {adj_matrix.size()}\n")
#         f.write(f"Non-zero Elements: {adj_matrix._nnz()}\n")
#         f.write(f"Frobenius Norm^2: {frob_norm_sq:.4f}\n")
#         f.write(f"Top-k Components: {k}\n")
#         f.write(f"Total Explained Variance: {cumulative_perc[-1]:.4f} ({cumulative_perc[-1]*100:.2f}%)\n")
    
#     vector_path = os.path.splitext(save_path)[0] + "_sV.pt"
#     torch.save(right_singular_vectors, vector_path)
#     vector_path = os.path.splitext(save_path)[0] + "_u.pt"
#     torch.save(left_singular_vectors, vector_path)
#     print(f"Adjacency matrix singular values saved to {save_path}")
#     return singular_values, frob_norm_sq

# def compute_and_save_singular_values(student_representation, save_path):

#     # 中心化数据
#     centered = student_representation - student_representation.mean(dim=0)
    
#     # 计算奇异值分解
#     with torch.no_grad():
#         U, S, V = torch.linalg.svd(centered, full_matrices=False)
#         singular_values = S.cpu().numpy()
#         right_singular_vectors = V.cpu()
#         left_singular_vectors = U.cpu()

#     # 计算累积奇异值
#     cumulative_sv = np.cumsum(singular_values)
#     cumulative_perc = cumulative_sv / cumulative_sv[-1]
    
#     # 确保目录存在
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # # 保存到TXT文件
    # with open(save_path, 'w') as f:
    #     # 写入奇异值
    #     f.write("Singular Values:\n")
    #     for i, sv in enumerate(singular_values):
    #         f.write(f"Component {i+1}: {sv:.6f}\n")
        
    #     # 写入累积奇异值
    #     f.write("\nCumulative Singular Values:\n")
    #     for i, csv in enumerate(cumulative_sv):
    #         f.write(f"Component {i+1}: {csv:.6f}\n")
        
    #     # 写入解释方差比例
    #     f.write("\nExplained Variance Proportion:\n")
    #     for i, perc in enumerate(cumulative_perc):
    #         f.write(f"Component {i+1}: {perc:.4f} ({perc*100:.2f}%)\n")
        
    #     # 写入元数据
    #     num_components = len(singular_values)
    #     f.write("\nMetadata:\n")
    #     f.write(f"Total Components: {num_components}\n")
    #     f.write(f"Sum of Singular Values: {sum(singular_values):.4f}\n")
    #     f.write(f"Representation Dimensions: {student_representation.shape}\n")
    #     f.write(f"Number of Students: {student_representation.shape[0]}\n")

    # vector_path = os.path.splitext(save_path)[0] + "_V1.1.pt"
    # torch.save(right_singular_vectors, vector_path)
    # vector_path = os.path.splitext(save_path)[0] + "_u1.1.pt"
    # torch.save(left_singular_vectors, vector_path)
    # print(f"Singular values saved to {save_path}")
    # return singular_values



class LIGHTGCN(_CognitiveDiagnosisModel):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int):
        """
        Description:
        ULCDF ...

        Parameters:
        student_num: int type
            The number of students in the response logs
        exercise_num: int type
            The number of exercises in the response logs
        knowledge_num: int type
            The number of knowledge concepts in the response logs
        method: Ignored
            Not used, present here for API consistency by convention.
        """
        super().__init__(student_num, exercise_num, knowledge_num)

    def build(self, device: str = "cpu", gcn_layers: int = 3, if_type='dp-linear'
              , keep_prob=0.9,
              dtype=torch.float32, hidden_dims: list = None,  mode='all', **kwargs):
        if hidden_dims is None:
            hidden_dims = [512, 256]

        if if_type == 'kancd':
            latent_dim = 32
        else:
            latent_dim = self.knowledge_num


        self.extractor = LIGHTGCN_Extractor(
            student_num=self.student_num,
            exercise_num=self.exercise_num,
            knowledge_num=self.knowledge_num,
            latent_dim=latent_dim,
            device=device,
            dtype=dtype,
            gcn_layers=gcn_layers,
            keep_prob=keep_prob,
        )

        self.device = device
        if if_type == 'ncd':
            self.inter_func = NCD_IF(knowledge_num=self.knowledge_num,
                                     hidden_dims=hidden_dims,
                                     dropout=0,
                                     device=device,
                                     dtype=dtype)
        elif 'dp' in if_type:
            self.inter_func = DP_IF(knowledge_num=self.knowledge_num,
                                    hidden_dims=hidden_dims,
                                    dropout=0,
                                    device=device,
                                    dtype=dtype,
                                    kernel=if_type)
        elif 'mirt' in if_type:
            self.inter_func = MIRT_IF(
                knowledge_num=self.knowledge_num,
                latent_dim=16,
                device=device,
                dtype=dtype,
                utlize=True)

        elif 'kancd' in if_type:
            self.inter_func = KANCD_IF(
                knowledge_num=self.knowledge_num,
                latent_dim=latent_dim,
                device=device,
                dtype=dtype,
                hidden_dims=hidden_dims,
                dropout=0.5
            )

        elif 'cdmfkc' in if_type:
            self.inter_func = CDMFKC_IF(
                g_impact_a=0.5,
                g_impact_b=0.5,
                knowledge_num=self.knowledge_num,
                hidden_dims=hidden_dims,
                dropout=0.5,
                device=device,
                dtype=dtype,
                latent_dim=latent_dim
            )

        elif 'irt' in if_type:
            self.inter_func = IRT_IF(
                device=device,
                dtype=dtype,
                latent_dim=latent_dim
            )

        else:
            raise ValueError("Remain to be aligned....")

    def __noise(self, graph):
        size = graph.size()
        indices = graph.indices().t()
        values = graph.values()
    
        original_edge_count = len(graph.values())  # 使用原始边数量计算
        noise_count = max(1, int(0.01 * original_edge_count))  # 至少 1 条
        
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
        return torch.sparse.DoubleTensor(combined_indices.t(), combined_values, size)

    def train(self, datahub: DataHub, set_type="train", valid_set_type="valid",
              valid_metrics=None, epoch=10, lr=5e-4, weight_decay=0.0005, batch_size=256):
        graph = self.__sp_mat_to_sp_tensor(self.__create_adj_mat(datahub["train"])).to(self.device)
        # graph = self.__noise(graph)
        self.extractor.get_graph(graph)
        if valid_metrics is None:
            valid_metrics = ["acc", "auc", "f1", "doa", 'ap']
        loss_func = nn.BCELoss()
        optimizer = optim.Adam([{'params': self.extractor.parameters(),
                                 'lr': lr, "weight_decay": weight_decay},
                                {'params': self.inter_func.parameters(),
                                 'lr': lr, "weight_decay": weight_decay}])
        for epoch_i in range(0, epoch):
            print("[Epoch {}]".format(epoch_i + 1))
            self._train(datahub=datahub, set_type=set_type,
                        valid_set_type=valid_set_type, valid_metrics=valid_metrics,
                        batch_size=batch_size, loss_func=loss_func, optimizer=optimizer)
        # sv = compute_and_save_singular_values(self.extractor["mastery"],save_path="./results/svd_analysis.txt")
        # sv = compute_and_save_singular_values_for_adj(graph,save_path="./results/asvd_analysis.txt")
    # 可视化结果
        # visualize_singular_values(singular_values=sv,save_path="./results/svd_visualization.png",show=True)

    def predict(self, datahub: DataHub, set_type, batch_size=256, **kwargs):
        return self._predict(datahub=datahub, set_type=set_type, batch_size=batch_size)

    def score(self, datahub: DataHub, set_type, metrics: list, batch_size=256, **kwargs) -> dict:
        if metrics is None:
            metrics = ["acc", "auc", "f1", "doa", 'ap']
        return self._score(datahub=datahub, set_type=set_type, metrics=metrics, batch_size=batch_size)

    def diagnose(self):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return self.inter_func.transform(self.extractor["mastery"],
                                         self.extractor["knowledge"])

    def load(self, ex_path: str, if_path: str):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        self.extractor.load_state_dict(torch.load(ex_path))
        self.inter_func.load_state_dict(torch.load(if_path))

    def save(self, ex_path: str, if_path: str):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        torch.save(self.extractor.state_dict(), ex_path)
        torch.save(self.inter_func.state_dict(), if_path)

    def diagnose(self):
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call \"build\" method to build interaction function before calling this method.")
        return self.inter_func.transform(self.extractor["mastery"],
                                         self.extractor["knowledge"])

    @staticmethod
    def __get_csr(rows, cols, shape):
        values = np.ones_like(rows, dtype=np.float64)
        return sp.csr_matrix((values, (rows, cols)), shape=shape)

    @staticmethod
    def __sp_mat_to_sp_tensor(sp_mat):
        coo = sp_mat.tocoo().astype(np.float64)
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()

    def __create_adj_mat(self, np_train):
        n_nodes = self.student_num + self.exercise_num
        train_stu = np_train[:, 0]
        train_exer = np_train[:, 1]
        ratings = np.ones_like(train_stu, dtype=np.float64)
        tmp_adj = sp.csr_matrix((ratings, (train_stu, train_exer + self.student_num)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix

    # def __create_adj_mat(self, np_train):
    #     # 添加5%的随机噪声交互（50%正确+50%错误）
    #     n_original = len(np_train)
    #     n_noise = int(np.ceil(n_original * 0.15))  # 计算噪声数量
        
    #     if n_noise > 0:
    #         noise_stu = np.random.randint(0, self.student_num, n_noise)
    #         noise_exer = np.random.randint(0, self.exercise_num, n_noise)
    #         noise_labels = np.random.choice([0, 1], size=n_noise)
    #         noise_data = np.zeros((n_noise, np_train.shape[1]), dtype=np_train.dtype)
    #         noise_data[:, 0] = noise_stu
    #         noise_data[:, 1] = noise_exer
    #         noise_data[:, 2] = noise_labels  
            
    #         np_train = np.vstack((np_train, noise_data))
        
    #     # 原始处理逻辑保持不变
    #     n_nodes = self.student_num + self.exercise_num
    #     train_stu = np_train[:, 0]
    #     train_exer = np_train[:, 1]
    #     ratings = np.ones_like(train_stu, dtype=np.float64)
        
    #     # 构建邻接矩阵 - 注意这里只考虑连接性
    #     tmp_adj = sp.csr_matrix((ratings, (train_stu, train_exer + self.student_num)),
    #                             shape=(n_nodes, n_nodes))
    #     adj_mat = tmp_adj + tmp_adj.T
        
    #     # 归一化处理
    #     rowsum = np.array(adj_mat.sum(1))
    #     d_inv = np.power(rowsum, -0.5).flatten()
    #     d_inv[np.isinf(d_inv)] = 0.
    #     d_mat_inv = sp.diags(d_inv)
    #     norm_adj_tmp = d_mat_inv.dot(adj_mat)
    #     adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        
    #     return adj_matrix