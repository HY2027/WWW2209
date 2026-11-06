import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp

from ...._base import _CognitiveDiagnosisModel
from ....datahub import DataHub
from ....interfunc import NCD_IF, DP_IF, MIRT_IF, MF_IF, KANCD_IF, CDMFKC_IF, IRT_IF, KSCD_IF
from ....extractor import GIDD_Extractor

from CAT.model.abstract_model import AbstractModel
from CAT.dataset import AdapTestDataset, TrainDataset, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score


class GIDD(_CognitiveDiagnosisModel, AbstractModel):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int, save_flag=False):
        super().__init__(student_num, exercise_num, knowledge_num, save_flag)

    def build(self, latent_dim=32, device: str = "cpu", gcn_layers: int = 3, if_type='dp-linear',
              keep_prob=0.9, dtype=torch.float64, hidden_dims: list = None, mode='all',
              flip_ratio=0.1, ssl_temp=0.8, ssl_weight=1e-2,
              # ===== Two-Stage Contrastive Learning Params =====
              stage1_temp=0.2, stage1_weight=1.0,   # Stage 1: reserved (no aug-graph source now)
              stage2_temp=0.8, stage2_weight=1.0,   # Stage 2: LLM-semantic vs Orig-graph
              # ===== Flip-Graph Contrastive Learning Params =====
              flip_temp=0.2, flip_weight=0.5,       # Orig-graph vs Flipped-graph
              # ===== Diffusion =====
              diffusion_steps=100, g_sae_k=8, g_sae_beta=0.01, lambda_g=0.5,
              channel_partition_threshold=0.5, attribution_steps=3,
              attribution_threshold=0.01, alpha_slow=0.5, noise_variance=1.0,
              # ===== Fast attribution =====
              use_fast_attribution=True,
              **kwargs):

        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.device = device
        self.mode = mode
        self.flip_ratio = flip_ratio

        # Build GIDD Extractor (KASA removed; original-graph + flip-graph +
        # Stage-2 + diffusion)
        self.extractor = GIDD_Extractor(
            student_num=self.student_num,
            exercise_num=self.exercise_num,
            knowledge_num=self.knowledge_num,
            latent_dim=latent_dim,
            device=device,
            dtype=dtype,
            gcn_layers=gcn_layers,
            keep_prob=keep_prob,
            mode=mode,
            ssl_temp=ssl_temp,
            ssl_weight=ssl_weight,
            stage1_temp=stage1_temp,
            stage1_weight=stage1_weight,
            stage2_temp=stage2_temp,
            stage2_weight=stage2_weight,
            flip_temp=flip_temp,
            flip_weight=flip_weight,
            diffusion_steps=diffusion_steps,
            g_sae_k=g_sae_k,
            g_sae_beta=g_sae_beta,
            lambda_g=lambda_g,
            channel_partition_threshold=channel_partition_threshold,
            attribution_steps=attribution_steps,
            attribution_threshold=attribution_threshold,
            alpha_slow=alpha_slow,
            noise_variance=noise_variance,
            use_fast_attribution=use_fast_attribution
        )

        # Build Interaction Function
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
                latent_dim=32,
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
        elif 'kscd' in if_type:
            self.inter_func = KSCD_IF(
                dropout=0.5,
                knowledge_num=self.knowledge_num,
                latent_dim=latent_dim,
                device=device,
                dtype=dtype)
        else:
            raise ValueError(f"Unsupported interaction function type: {if_type}")

    def train(self, datahub: DataHub, set_type="train", valid_set_type="valid",
              valid_metrics=None, epoch=10, lr=5e-4, weight_decay=0.0005, batch_size=256):
        """
        Train the GIDD model.

        Parameters:
        -----------
        datahub : DataHub
            Data hub with response data
        set_type : str
            Training set type
        valid_set_type : str
            Validation set type
        valid_metrics : list
            Validation metrics
        epoch : int
            Number of training epochs
        lr : float
            Learning rate
        weight_decay : float
            Weight decay
        batch_size : int
            Batch size
        """
        # Build knowledge graph
        if self.mode == 'Q':
            ek_graph = np.zeros(shape=datahub.q_matrix.shape)
        else:
            ek_graph = datahub.q_matrix.copy()

        # Create student-exercise subgraphs
        se_graph_right, se_graph_wrong = [
            self.__create_adj_se(datahub[set_type], is_subgraph=True)[i] for i in range(2)]
        se_graph = self.__create_adj_se(datahub[set_type], is_subgraph=False)

        # Build graph dictionary
        graph_dict = {
            'right': self.__final_graph(se_graph_right, ek_graph),
            'wrong': self.__final_graph(se_graph_wrong, ek_graph),
            'response': datahub[set_type],
            'Q_Matrix': datahub.q_matrix.copy(),
            'flip_ratio': self.flip_ratio,
            'all': self.__final_graph(se_graph, ek_graph)
        }

        self.extractor.get_graph_dict(graph_dict)

        if valid_metrics is None:
            valid_metrics = ["acc", "auc", "f1", "doa", 'ap']

        # Loss function and optimizer
        loss_func = nn.BCELoss()
        optimizer = optim.Adam([
            {'params': self.extractor.parameters(),
             'lr': lr, "weight_decay": weight_decay},
            {'params': self.inter_func.parameters(),
             'lr': lr, "weight_decay": weight_decay}
        ])

        # Training loop
        for epoch_i in range(epoch):
            print(f"[Epoch {epoch_i + 1}]")
            self.extractor.get_flip_graph()
            self._train(datahub=datahub, set_type=set_type,
                        valid_set_type=valid_set_type, valid_metrics=valid_metrics,
                        batch_size=batch_size, loss_func=loss_func, optimizer=optimizer)

    def predict(self, datahub: DataHub, set_type, batch_size=256, **kwargs):
        """Generate predictions for a given dataset."""
        return self._predict(datahub=datahub, set_type=set_type, batch_size=batch_size)

    def score(self, datahub: DataHub, set_type, metrics: list, batch_size=256, **kwargs) -> dict:
        """Evaluate model performance on specified metrics."""
        if metrics is None:
            metrics = ["acc", "auc", "f1", "doa", 'ap']
        return self._score(datahub=datahub, set_type=set_type, metrics=metrics, batch_size=batch_size)

    def diagnose(self):
        """Perform cognitive diagnosis to get student mastery levels."""
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call 'build' method to build interaction function before calling this method.")
        return self.inter_func.transform(self.extractor["mastery"],
                                         self.extractor["knowledge"])

    def get_attribute(self, attribute_name):
        """Get specific model attribute."""
        if attribute_name == 'mastery':
            return self.diagnose().detach().cpu().numpy()
        elif attribute_name == 'diff':
            return self.inter_func.transform(self.extractor["diff"],
                                              self.extractor["knowledge"]).detach().cpu().numpy()
        elif attribute_name == 'knowledge':
            return self.extractor["knowledge"].detach().cpu().numpy()
        else:
            return None

    def load(self, ex_path: str, if_path: str):
        """Load model weights from files."""
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call 'build' method first.")
        self.extractor.load_state_dict(torch.load(ex_path))
        self.inter_func.load_state_dict(torch.load(if_path))

    def save(self, ex_path: str, if_path: str):
        """Save model weights to files."""
        if self.inter_func is Ellipsis or self.extractor is Ellipsis:
            raise RuntimeError("Call 'build' method first.")
        torch.save(self.extractor.state_dict(), ex_path)
        torch.save(self.inter_func.state_dict(), if_path)

    def update_graph(self, data, q_matrix):
        """Update graph structure with new data."""
        se_graph_right, se_graph_wrong = [
            self.__create_adj_se(data, is_subgraph=True)[i] for i in range(2)]
        self.extractor.graph_dict['right'] = self.__final_graph(se_graph_right, q_matrix)
        self.extractor.graph_dict['wrong'] = self.__final_graph(se_graph_wrong, q_matrix)

    @property
    def name(self):
        """Model name."""
        return 'gidd'

    # ========== CAT Framework Methods ==========

    def adaptest_update(self, adaptest_data: AdapTestDataset, epoch=10, lr=5e-4, weight_decay=0, batch_size=256):
        """Update model with adaptive testing data."""
        loss_func = nn.BCELoss()
        optimizer = optim.Adam([
            {'params': self.extractor.parameters(),
             'lr': lr, "weight_decay": weight_decay},
            {'params': self.inter_func.parameters(),
             'lr': lr, "weight_decay": weight_decay}
        ])

        tested_dataset = adaptest_data.get_tested_dataset(last=True)
        dataloader = torch.utils.data.DataLoader(tested_dataset, batch_size=batch_size, shuffle=True)

        for epoch_i in range(epoch):
            loss = 0.0
            for student_ids, question_ids, concepts_emb, labels in dataloader:
                student_ids = student_ids.to(self.device)
                question_ids = question_ids.to(self.device)
                labels = labels.to(self.device)
                concepts_emb = concepts_emb.to(self.device)

                _ = self.extractor.extract(student_ids, question_ids, concepts_emb)
                student_ts, diff_ts, disc_ts, knowledge_ts = _[:4]

                pred = self.inter_func.compute(
                    student_ts=student_ts,
                    diff_ts=diff_ts,
                    disc_ts=disc_ts,
                    q_mask=concepts_emb,
                    knowledge_ts=knowledge_ts
                )

                bz_loss = loss_func(pred, labels.double())
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                self.inter_func.monotonicity()
                loss += bz_loss.data.float()

    def evaluate(self, adaptest_data: AdapTestDataset):
        """Evaluate model on adaptive testing data."""
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map

        real = []
        pred = []

        with torch.no_grad():
            self.extractor.eval()
            self.inter_func.eval()
            for sid in data:
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                concepts_embs = []
                for qid in question_ids:
                    concepts = concept_map[qid]
                    concepts_emb = [0.] * adaptest_data.num_concepts
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)
                real += [data[sid][qid] for qid in question_ids]
                student_ids = torch.LongTensor(student_ids).to(self.device)
                question_ids = torch.LongTensor(question_ids).to(self.device)
                concepts_embs = torch.Tensor(concepts_embs).to(self.device)

                _ = self.extractor.extract(student_ids, question_ids, concepts_embs)
                student_ts, diff_ts, disc_ts, knowledge_ts = _[:4]
                output = self.inter_func.compute(
                    student_ts=student_ts,
                    diff_ts=diff_ts,
                    disc_ts=disc_ts,
                    q_mask=concepts_embs,
                    knowledge_ts=knowledge_ts
                )
                pred += output.tolist()
            self.extractor.train()
            self.inter_func.train()

        # Compute coverage
        coverages = []
        for sid in data:
            all_concepts = set()
            tested_concepts = set()
            for qid in data[sid]:
                all_concepts.update(set(concept_map[qid]))
            for qid in adaptest_data.tested[sid]:
                tested_concepts.update(set(concept_map[qid]))
            coverage = len(tested_concepts) / len(all_concepts) if len(all_concepts) > 0 else 0
            coverages.append(coverage)
        cov = sum(coverages) / len(coverages) if coverages else 0

        real = np.array(real)
        real = np.where(real < 0.5, 0.0, 1.0)
        pred = np.array(pred)
        auc = roc_auc_score(real, pred) if len(set(real)) > 1 else 0.5
        binary_pred = (pred >= 0.5).astype(int)
        acc = accuracy_score(real, binary_pred)

        return {'auc': auc, 'cov': cov, 'acc': acc}

    def init_model(self, data: Dataset):
        """Initialize model with dataset."""
        pass

    def adaptest_save(self, path):
        """Save model for adaptive testing."""
        extractor_dict = self.extractor.state_dict()
        torch.save(extractor_dict, path[0])
        interaction_dict = self.inter_func.state_dict()
        torch.save(interaction_dict, path[1])

    def adaptest_load(self, path):
        """Load model for adaptive testing."""
        self.extractor.load_state_dict(torch.load(path[0]), strict=False)
        self.extractor.to(self.device)
        self.inter_func.load_state_dict(torch.load(path[1]), strict=False)
        self.inter_func.to(self.device)

    def get_pred(self, adaptest_data: AdapTestDataset):
        """Get predictions for all data in adaptive testing dataset."""
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map

        pred_all = {}
        with torch.no_grad():
            self.extractor.eval()
            self.inter_func.eval()
            for sid in data:
                pred_all[sid] = {}
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                concepts_embs = []
                for qid in question_ids:
                    concepts = concept_map[qid]
                    concepts_emb = [0.] * adaptest_data.num_concepts
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)
                student_ids = torch.LongTensor(student_ids).to(self.device)
                question_ids = torch.LongTensor(question_ids).to(self.device)
                concepts_embs = torch.Tensor(concepts_embs).to(self.device)

                _ = self.extractor.extract(student_ids, question_ids, concepts_embs)
                student_ts, diff_ts, disc_ts, knowledge_ts = _[:4]
                output = self.inter_func.compute(
                    student_ts=student_ts,
                    diff_ts=diff_ts,
                    disc_ts=disc_ts,
                    q_mask=concepts_embs,
                    knowledge_ts=knowledge_ts
                )
                output = output.detach().cpu().numpy().tolist()
                for i, qid in enumerate(list(data[sid].keys())):
                    pred_all[sid][qid] = output[i]
            self.extractor.train()
            self.inter_func.train()
        return pred_all

    def expected_model_change(self, sid: int, qid: int, adaptest_data: AdapTestDataset, pred_all: dict, config):
        """Compute expected model change for question selection."""
        loss_func = nn.BCELoss()
        optimizer = optim.Adam([
            {'params': self.extractor.parameters(),
             'lr': config['lr'], "weight_decay": 0},
            {'params': self.inter_func.parameters(),
             'lr': config['lr'], "weight_decay": 0}
        ])

        for name, param in self.inter_func.named_parameters():
            param.requires_grad = False

        original_state_dict = {k: v.clone() for k, v in self.extractor.state_dict().items()}

        student_ids = torch.LongTensor([sid]).to(self.device)
        question_ids = torch.LongTensor([qid]).to(self.device)
        concepts = adaptest_data.concept_map[qid]
        concepts_embs = [0.] * adaptest_data.num_concepts
        for concept in concepts:
            concepts_embs[concept] = 1.0
        concepts_embs = torch.Tensor([concepts_embs]).to(self.device)

        # Forward pass with correct answer
        for ep in range(config['epoch']):
            optimizer.zero_grad()
            _ = self.extractor.extract(student_ids, question_ids, concepts_embs)
            student_ts, diff_ts, disc_ts, knowledge_ts = _[:4]
            pred = self.inter_func.compute(
                student_ts=student_ts, diff_ts=diff_ts, disc_ts=disc_ts,
                q_mask=concepts_embs, knowledge_ts=knowledge_ts
            )
            loss = loss_func(pred, torch.LongTensor([1]).to(self.device).double())
            loss.backward()
            optimizer.step()

        pos_weights = self.extractor['mastery'].data.clone()
        self.extractor.load_state_dict(original_state_dict)

        # Forward pass with wrong answer
        for ep in range(config['epoch']):
            optimizer.zero_grad()
            _ = self.extractor.extract(student_ids, question_ids, concepts_embs)
            student_ts, diff_ts, disc_ts, knowledge_ts = _[:4]
            pred = self.inter_func.compute(
                student_ts=student_ts, diff_ts=diff_ts, disc_ts=disc_ts,
                q_mask=concepts_embs, knowledge_ts=knowledge_ts
            )
            loss = loss_func(pred, torch.LongTensor([0]).to(self.device).double())
            loss.backward()
            optimizer.step()

        neg_weights = self.extractor['mastery'].data.clone()
        self.extractor.load_state_dict(original_state_dict)

        for name, param in self.inter_func.named_parameters():
            param.requires_grad = True

        pred = pred_all[sid][qid]
        return pred * torch.norm(pos_weights - self.extractor['mastery'].data).item() + \
            (1 - pred) * torch.norm(neg_weights - self.extractor['mastery'].data).item()

    # ========== Graph Construction Utilities ==========

    @staticmethod
    def __get_csr(rows, cols, shape):
        """Create sparse row matrix from rows and columns."""
        values = np.ones_like(rows, dtype=np.float64)
        return sp.csr_matrix((values, (rows, cols)), shape=shape)

    @staticmethod
    def __sp_mat_to_sp_tensor(sp_mat):
        """Convert scipy sparse matrix to PyTorch sparse tensor."""
        coo = sp_mat.tocoo().astype(np.float64)
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()

    def __create_adj_se(self, np_response, is_subgraph=False):
        """Create student-exercise adjacency matrices."""
        if is_subgraph:
            if self.mode == 'R':
                return (np.zeros(shape=(self.student_num, self.exercise_num)),
                        np.zeros(shape=(self.student_num, self.exercise_num)))

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
            adj_se = self.__get_csr(response_stu, response_exer,
                                    shape=(self.student_num, self.exercise_num))
            return adj_se.toarray()

    def __final_graph(self, se, ek):
        """Construct normalized combined graph."""
        sek_num = self.student_num + self.exercise_num + self.knowledge_num
        se_num = self.student_num + self.exercise_num
        tmp = np.zeros(shape=(sek_num, sek_num))
        tmp[:self.student_num, self.student_num:se_num] = se
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
