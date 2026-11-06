import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .._base import _Extractor
import scipy.sparse as sp
import ast
# NOTE: KASA module removed; GIDD now relies solely on the original-graph
# structural representation learning + LLM-semantic Stage-2 contrastive
# alignment + interpretable denoising diffusion.


class GIDD_Extractor(_Extractor, nn.Module):
    def __init__(self, student_num: int, exercise_num: int, knowledge_num: int, latent_dim: int, device,
                 dtype, gcn_layers=3, keep_prob=0.9, mode='all',
                 # ===== Two-Stage Contrastive Learning Hyperparameters =====
                 ssl_temp=0.5, ssl_weight=1e-2,
                 stage2_temp=0.5, stage2_weight=1.0,      # Stage 2: LLM semantic vs Orig-graph
                 # ===== Flip-Graph Contrastive Learning (Random-Flip Augmentation) =====
                 flip_temp=0.2, flip_weight=0.5,           # Orig-graph vs Flipped-graph
                 # ===== Diffusion =====
                 diffusion_steps=20, g_sae_k=32, g_sae_beta=0.01, lambda_g=0.7,
                 channel_partition_threshold=0.5, attribution_steps=10,
                 attribution_threshold=0.01, alpha_slow=0.5, noise_variance=1.0,
                 # ===== Fast attribution =====
                 use_fast_attribution=False,
                 **kwargs):
        super().__init__()
        self.student_num = student_num
        self.exercise_num = exercise_num
        self.knowledge_num = knowledge_num
        self.latent_dim = latent_dim
        self.mode = mode
        self.stage2_temp = ssl_temp
        self.stage2_weight = ssl_weight
        self.flip_temp = flip_temp
        self.flip_weight = flip_weight
        self.diffusion_steps = diffusion_steps

        # G-SAE parameters
        self.g_sae_k = g_sae_k
        self.g_sae_beta = g_sae_beta
        self.lambda_g = lambda_g

        # Feature channel partitioning
        self.channel_partition_threshold = channel_partition_threshold

        # Noise attribution parameters
        self.attribution_steps = attribution_steps
        self.attribution_threshold = attribution_threshold
        self.use_fast_attribution = use_fast_attribution

        # MixFlow parameters
        self.alpha_slow = alpha_slow

        # GDCW parameters
        self.noise_variance = noise_variance

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

        self.__init_semantic_embeddings()

        self.struct_fusion_layer = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(self.latent_dim, self.latent_dim, dtype=self.dtype)
        ).to(self.device)

        self.g_sae_encoder = nn.Linear(self.latent_dim, self.latent_dim, dtype=self.dtype).to(self.device)
        # Bias term: b_pre (learnable)
        self.g_sae_bias = nn.Parameter(torch.zeros(self.latent_dim, dtype=self.dtype)).to(self.device)
        # Decoder: W_dec for reconstruction
        self.g_sae_decoder = nn.Linear(self.latent_dim, self.latent_dim, dtype=self.dtype).to(self.device)
        # GCN projection for structure-aware encoding
        self.g_sae_gcn_proj = nn.Linear(self.latent_dim, self.latent_dim, dtype=self.dtype).to(self.device)

        self.structure_extractor = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(self.latent_dim // 2, self.latent_dim, dtype=self.dtype)
        ).to(self.device)

        self.semantic_extractor = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(self.latent_dim // 2, self.latent_dim, dtype=self.dtype)
        ).to(self.device)

        self.__init_diffusion_components()


        self.snr_params = nn.Parameter(torch.ones(self.diffusion_steps, dtype=self.dtype)).to(self.device)

        # ========== Output Layers ==========
        self.concat_layer = nn.Linear(2 * self.latent_dim, self.latent_dim, dtype=self.dtype).to(self.device)
        self.transfer_student_layer = nn.Linear(self.latent_dim, self.knowledge_num, dtype=self.dtype).to(self.device)
        self.transfer_exercise_layer = nn.Linear(self.latent_dim, self.knowledge_num, dtype=self.dtype).to(self.device)
        self.transfer_knowledge_layer = nn.Linear(self.latent_dim, self.knowledge_num, dtype=self.dtype).to(self.device)

        # Embedding map for attribute access
        self.__emb_map = {
            "mastery": self.__student_emb.weight,
            "diff": self.__exercise_emb.weight,
            "disc": self.__disc_emb.weight,
            "knowledge": self.__knowledge_emb.weight
        }

        # Store channel partition results (updated during training)
        self.structure_channel_mask = None
        self.semantic_channel_mask = None
        self.noise_feature_set = None

        # Cache for channel partitioning and noise attribution (computed once per epoch)
        self._cached_F_struct_stu = None
        self._cached_F_sem_stu = None
        self._cached_F_noise_stu = None
        self._cached_attr_stu = None
        self._cached_F_struct_exer = None
        self._cached_F_sem_exer = None
        self._cached_F_noise_exer = None
        self._cached_attr_exer = None
        self._cache_valid = False
        self._last_H_fused_stu_shape = None

        # Cache for GDCW spectral transform
        self._gdcw_spectral_cache = None
        self._gdcw_eigen_cache = None


        self.apply(self.initialize_weights)

    def __init_semantic_embeddings(self):
        llm_stu_emb = []
        llm_exer_emb = []
# Junyi1
# Assist0910
# NeurIPS
        try:
            with open("datasets/NeurIPS/stu_emb") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    vec = ast.literal_eval(line)
                    llm_stu_emb.append(vec)

            with open("datasets/NeurIPS/exer_emb") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    vec = ast.literal_eval(line)
                    llm_exer_emb.append(vec)

            llm_weight_matrix_stu = np.array(llm_stu_emb, dtype=np.float64)
            llm_weight_matrix_exer = np.array(llm_exer_emb, dtype=np.float64)

            # Read the *actual* LLM embedding dims from on-disk content
            llm_stu_dim = int(llm_weight_matrix_stu.shape[1])
            llm_exer_dim = int(llm_weight_matrix_exer.shape[1])

            llm_stu_embedding = nn.Embedding(self.student_num, llm_stu_dim).to(self.device)
            llm_exer_embedding = nn.Embedding(self.exercise_num, llm_exer_dim).to(self.device)

            llm_weight_tensor_stu = torch.from_numpy(llm_weight_matrix_stu).to(dtype=self.dtype).to(self.device)
            llm_weight_tensor_exer = torch.from_numpy(llm_weight_matrix_exer).to(dtype=self.dtype).to(self.device)

            llm_stu_embedding.weight = nn.Parameter(llm_weight_tensor_stu)
            llm_exer_embedding.weight = nn.Parameter(llm_weight_tensor_exer)

            self.__llm_stu_emb = llm_stu_embedding.weight
            self.__llm_exer_emb = llm_exer_embedding.weight

        except Exception:
            # Fallback: random embeddings, matching latent_dim
            llm_stu_dim = llm_exer_dim = self.latent_dim
            self.__llm_stu_emb = nn.Parameter(
                torch.randn(self.student_num, llm_stu_dim, dtype=self.dtype).to(self.device))
            self.__llm_exer_emb = nn.Parameter(
                torch.randn(self.exercise_num, llm_exer_dim, dtype=self.dtype).to(self.device))

        # ---- Build the learnable semantic projections (shape-adaptive) ----
        self.pca_projection_student = nn.Linear(llm_stu_dim, self.latent_dim,
                                                dtype=self.dtype).to(self.device)
        self.pca_projection_exercise = nn.Linear(llm_exer_dim, self.latent_dim,
                                                 dtype=self.dtype).to(self.device)

    def __init_diffusion_components(self):
        """Initialize diffusion model components for Graph MixFlow."""
        # Noise schedule
        self.betas = torch.linspace(1e-4, 0.02, self.diffusion_steps).to(self.device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # Noise prediction network for normal trajectory
        self.epsilon_theta = nn.Sequential(
            nn.Linear(self.latent_dim + 64 + self.latent_dim, 256, dtype=self.dtype),
            nn.Tanh(),
            nn.LayerNorm(256, dtype=self.dtype),
            nn.Linear(256, 128, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(128, self.latent_dim, dtype=self.dtype)
        ).to(self.device)

        # Noise prediction network for slow flow trajectory
        self.epsilon_theta_slow = nn.Sequential(
            nn.Linear(self.latent_dim + 64 + self.latent_dim, 256, dtype=self.dtype),
            nn.Tanh(),
            nn.LayerNorm(256, dtype=self.dtype),
            nn.Linear(256, 128, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(128, self.latent_dim, dtype=self.dtype)
        ).to(self.device)

        # Time step embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 64, dtype=self.dtype),
            nn.Tanh(),
            nn.Linear(64, 64, dtype=self.dtype)
        ).to(self.device)

        # Slow Flow gate network
        self.slow_flow_gate = nn.Sequential(
            nn.Linear(self.latent_dim, 64, dtype=self.dtype),
            nn.Sigmoid()
        ).to(self.device)

    @staticmethod
    def initialize_weights(module):
        """Initialize weights using Xavier normal for Linear and Embedding layers."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)

    def get_graph_dict(self, graph_dict):
        """Store the graph dictionary for forward computation.

        NOTE: KASA augmentation trigger removed; the original graph is the
        only structural source now.
        """
        self.graph_dict = graph_dict
        # Invalidate caches when graph changes
        self._cache_valid = False
        self._gdcw_eigen_cache = None

    def invalidate_cache(self):
        """Invalidate all caches (call at start of each epoch)."""
        self._cache_valid = False
        self._gdcw_eigen_cache = None

    def _compute_channel_partition_and_attribution(self, H_fused_stu, H_fused_exer, graph_adj):
        with torch.no_grad():
            self._cached_F_struct_stu, self._cached_F_sem_stu = self.__partition_feature_channels(
                H_fused_stu, graph_adj)
            self._cached_F_struct_exer, self._cached_F_sem_exer = self.__partition_feature_channels(
                H_fused_exer, graph_adj)

        F_noise_stu, attr_stu = self.__compute_noise_attribution(
            H_fused_stu, self._cached_F_struct_stu, self._cached_F_sem_stu, graph_adj)
        F_noise_exer, attr_exer = self.__compute_noise_attribution(
            H_fused_exer, self._cached_F_struct_exer, self._cached_F_sem_exer, graph_adj)

        self._cached_F_noise_stu = F_noise_stu.detach()
        self._cached_attr_stu = attr_stu.detach()
        self._cached_F_noise_exer = F_noise_exer.detach()
        self._cached_attr_exer = attr_exer.detach()

        self.structure_channel_mask = self._cached_F_struct_stu
        self.semantic_channel_mask = self._cached_F_sem_stu
        self.noise_feature_set = self._cached_F_noise_stu

        self._cache_valid = True

    def get_all_emb(self):
        """Concatenate student, exercise, and knowledge embeddings."""
        stu_emb = self.__student_emb.weight
        exer_emb = self.__exercise_emb.weight
        know_emb = self.__knowledge_emb.weight
        all_emb = torch.cat([stu_emb, exer_emb, know_emb]).to(self.device)
        return all_emb

    def convolution(self, graph):
        """Apply multi-layer GCN convolution on a single graph."""
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
            right_emb = torch.sparse.mm(self.__graph_drop(right), right_emb)
            wrong_emb = torch.sparse.mm(self.__graph_drop(wrong), wrong_emb)
            all_emb = self.concat_layer(torch.cat([right_emb, wrong_emb], dim=1))
            emb.append(all_emb)
        out_emb = torch.mean(torch.stack(emb, dim=1), dim=1)
        return (out_emb[:self.student_num],
                out_emb[self.student_num:self.student_num + self.exercise_num],
                out_emb[self.exercise_num + self.student_num:])

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
        return self.__dropout(graph, self.keep_prob)

    def __time_embed(self, t):
        t = t.double()
        return self.time_embedding(t.view(-1, 1))


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
        
        graph = graph.coalesce()

        degrees = torch.sparse.sum(graph, dim=1).to_dense()[:self.student_num]

        enhanced_graph = graph.clone()
        new_edges = []  
      
        for cluster_id in range(self.cluster_num):
            cluster_students = torch.where(labels == cluster_id)[0]
            n_in_cluster = len(cluster_students)

           
            cluster_degrees = degrees[cluster_students]
            
            num_low_deg = max(1, n_in_cluster // 2) 
            
            sorted_indices = torch.argsort(cluster_degrees)
            
            low_deg_indices = sorted_indices[:num_low_deg]
            low_degree_students = cluster_students[low_deg_indices]
            
            for s_idx in low_degree_students:
                other_students = cluster_students[cluster_students != s_idx]
                num_edges = min(self.max_intra_edges, len(other_students))
                selected = torch.randperm(len(other_students), device=self.device)[:num_edges]
                neighbors = other_students[selected]
                
                for n_idx in neighbors:
                    new_edges.append((s_idx, n_idx))
                    new_edges.append((n_idx, s_idx))  
        
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

    def __forward_diffusion(self, x0, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).view(-1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t]).view(-1, 1)
        noise = torch.randn_like(x0, device=self.device)
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return xt, noise

    def __normal_denoise(self, xt, t, context):
        time_emb = self.__time_embed(t)
        return self.epsilon_theta(torch.cat([xt, time_emb, context], dim=-1).double())

    def __slow_denoise(self, xt, t, context):
        time_emb = self.__time_embed(t)
        return self.epsilon_theta_slow(torch.cat([xt, time_emb, context], dim=-1).double())

    def __slow_flow_reverse(self, x_T, t, F_noise, attr, context, gamma=0.8):
        batch_size = x_T.size(0)
        xt = x_T.clone()

        # Compute noise suppression factor based on attribution
        if F_noise is not None and attr is not None and F_noise.sum() > 0:
            noise_suppression = torch.ones_like(xt)
            # Apply suppression to noise features
            attr_weights = torch.abs(attr) / (torch.abs(attr).max() + 1e-8)
            suppression_factor = 1.0 - gamma * attr_weights * F_noise.float()
            noise_suppression = noise_suppression * suppression_factor.unsqueeze(0)
        else:
            noise_suppression = torch.ones_like(xt)

        # Single step reverse with suppression
        alpha_t = self.alphas[t].view(-1, 1)

        # Predict noise using clean context (not noisy xt)
        time_emb = self.__time_embed(torch.full((batch_size,), t, device=self.device, dtype=self.dtype))
        predicted_noise = self.epsilon_theta_slow(
            torch.cat([xt, time_emb, context], dim=-1).double()
        )

        # Denoise with suppression
        xt_denoised = (xt - predicted_noise * (1 - noise_suppression)) / torch.sqrt(alpha_t + 1e-8)

        # Add controlled noise (controlled by suppression factor)
        if t > 0:
            xt_denoised = xt_denoised + torch.sqrt(
                self.betas[t].view(-1, 1) + 1e-8
            ) * torch.randn_like(xt) * noise_suppression

        return xt_denoised


    def __g_sae_encode(self, h, graph_adj, gcn_cache=None):
        base_encoding = self.g_sae_encoder(h - self.g_sae_bias)

        if gcn_cache is not None:
            gcn_component = gcn_cache
        elif graph_adj is not None and h.size(0) == graph_adj.size(0):
            gcn_component = self.g_sae_gcn_proj(torch.sparse.mm(graph_adj, h))
        else:
            # Batch-level encoding without GCN (or no graph available)
            gcn_component = torch.zeros_like(base_encoding)

        # Combined encoding
        combined = base_encoding + self.lambda_g * gcn_component

        # Top-K selection along feature dimension
        if self.training:
            # During training, use soft selection via sigmoid-weighted approach
            importance_scores = torch.abs(combined)
            topk_indices = torch.topk(importance_scores, k=min(self.g_sae_k, self.latent_dim),
                                      dim=-1).indices
            s = torch.zeros_like(combined)
            s.scatter_(1, topk_indices, combined.gather(1, topk_indices))
        else:
            # During inference, hard TopK
            s = combined

        return s, combined

    def __g_sae_decode(self, s):
        """Step 8: G-SAE Decoder to reconstruct features."""
        return self.g_sae_decoder(s)

    def __compute_g_sae_loss(self, h, graph_adj, gcn_cache=None):
        s, h_encoded = self.__g_sae_encode(h, graph_adj, gcn_cache=gcn_cache)
        h_hat = self.__g_sae_decode(s)

        # Reconstruction loss
        recon_loss = F.mse_loss(h_hat, h)

        # L1 sparsity loss on selected features
        sparsity_loss = self.g_sae_beta * torch.mean(torch.abs(s))

        return recon_loss + sparsity_loss

    def __partition_feature_channels(self, H_fused, graph):

        batch_size, dim = H_fused.shape

        with torch.no_grad():
            # Compute structure sensitivity (response to graph perturbation)
            H_perturbed = H_fused + torch.randn_like(H_fused) * 0.1
            H_semantic_replaced = self.semantic_extractor(H_fused)

            structure_response = torch.abs(H_fused - H_perturbed).mean(dim=0)
            semantic_response = torch.abs(H_fused - H_semantic_replaced).mean(dim=0)

            total_response = structure_response + semantic_response + 1e-8
            structure_ratio = structure_response / total_response

            F_struct = structure_ratio > self.channel_partition_threshold
            F_sem = ~F_struct

            min_channels = max(1, dim // 8)  # At least 1 channel, or dim/8
            if F_struct.sum() < min_channels:
                # Fallback: select top-k channels by structure_ratio
                topk_struct = torch.topk(structure_ratio, k=min_channels).indices
                F_struct = torch.zeros(dim, device=H_fused.device, dtype=torch.bool)
                F_struct[topk_struct] = True
                F_sem = ~F_struct
            elif F_sem.sum() < min_channels:
                topk_sem = torch.topk(1 - structure_ratio, k=min_channels).indices
                F_sem = torch.zeros(dim, device=H_fused.device, dtype=torch.bool)
                F_sem[topk_sem] = True
                F_struct = ~F_sem

        # Store masks for later use
        self.structure_channel_mask = F_struct
        self.semantic_channel_mask = F_sem

        return F_struct, F_sem

    def __integrated_gradients(self, loss_fn, x, baseline, steps=None):
        if steps is None:
            steps = self.attribution_steps

        attributions = torch.zeros_like(x)

        for step in range(steps):
            alpha = (step + 1) / steps
            interpolated = baseline + alpha * (x - baseline)
            interpolated = interpolated.detach().requires_grad_(True)

            loss = loss_fn(interpolated)
            grad = torch.autograd.grad(loss, interpolated, retain_graph=False)[0]

            attributions += grad

        attributions = attributions * (x - baseline) / steps
        return attributions

    def __compute_noise_attribution(self, H_fused, F_struct, F_sem, graph):
        batch_size = H_fused.size(0)
        dim = H_fused.size(1)
        baseline = torch.zeros_like(H_fused)

        was_training = self.training
        self.eval()  # Switch to eval mode to make __g_sae_encode fully differentiable

        def structure_loss(x):
            x_masked = x * F_struct.float().unsqueeze(0).expand_as(x)
            s, _ = self.__g_sae_encode(x_masked, graph)
            h_hat = self.__g_sae_decode(s)
            return F.mse_loss(h_hat * F_struct.float().unsqueeze(0).expand_as(x), x_masked)

        # Loss function for semantic channel denoising
        def semantic_loss(x):
            x_masked = x * F_sem.float().unsqueeze(0).expand_as(x)
            s, _ = self.__g_sae_encode(x_masked, graph)
            h_hat = self.__g_sae_decode(s)
            return F.mse_loss(h_hat * F_sem.float().unsqueeze(0).expand_as(x), x_masked)

        if self.use_fast_attribution:
            with torch.enable_grad():
                H_clone = H_fused.detach().clone().requires_grad_(True)
                
                # Structure attribution: single gradient
                struct_loss = structure_loss(H_clone)
                struct_grad = torch.autograd.grad(struct_loss, H_clone, retain_graph=True)[0]
                struct_attr = torch.abs(struct_grad * H_clone.detach())  # Simple gradient * input
                
                # Semantic attribution: single gradient
                sem_loss = semantic_loss(H_clone)
                sem_grad = torch.autograd.grad(sem_loss, H_clone, retain_graph=False)[0]
                sem_attr = torch.abs(sem_grad * H_clone.detach())
        else:
            # Standard attribution: full IntegratedGradients
            with torch.enable_grad():
                H_clone = H_fused.detach().clone().requires_grad_(True)
                struct_attr = self.__integrated_gradients(structure_loss, H_clone, baseline)
                sem_attr = self.__integrated_gradients(semantic_loss, H_clone, baseline)

        # Restore training mode
        if was_training:
            self.train()

        # Combine attributions
        attr = torch.abs(struct_attr) + torch.abs(sem_attr)
        channel_attr = attr.mean(dim=0)  # [D]

        # Define noise feature set with fallback mechanisms
        F_noise = channel_attr > self.attribution_threshold

        # Guard: ensure F_noise is not all-False (would make slow flow identical to normal)
        min_noise_channels = max(1, dim // 8)
        if F_noise.sum() < min_noise_channels:
            # Fallback: select top-k channels by attribution magnitude
            topk_idx = torch.topk(torch.abs(channel_attr), k=min_noise_channels).indices
            F_noise = torch.zeros(dim, device=H_fused.device, dtype=torch.bool)
            F_noise[topk_idx] = True

        # Guard: ensure F_noise is not all-True (would destroy all signal)
        max_noise_channels = dim - max(1, dim // 4)
        if F_noise.sum() > max_noise_channels:
            # Select top-k channels by attribution magnitude
            topk_idx = torch.topk(torch.abs(channel_attr), k=max_noise_channels).indices
            F_noise = torch.zeros(dim, device=H_fused.device, dtype=torch.bool)
            F_noise[topk_idx] = True

        self.noise_feature_set = F_noise

        return F_noise, channel_attr

    def __compute_mixflow_loss(self, H_fused, F_noise, attr):
        batch_size = H_fused.size(0)
        context = H_fused.detach()

        t = torch.randint(1, self.diffusion_steps, (1,)).item()

        # ===== Normal trajectory =====
        x_t_normal, noise_normal = self.__forward_diffusion(H_fused, t)
        pred_noise_normal = self.__normal_denoise(
            x_t_normal,
            torch.full((batch_size,), t, device=self.device, dtype=self.dtype),
            context
        )
        l_normal = F.mse_loss(pred_noise_normal, noise_normal)

        x_T = torch.randn_like(H_fused)

        # Generate noise-suppressed features using clean context
        x_t_slow = self.__slow_flow_reverse(
            x_T, t, F_noise, attr, context, gamma=0.8
        )

        pred_noise_slow = self.__slow_denoise(
            x_t_slow,
            torch.full((batch_size,), t, device=self.device, dtype=self.dtype),
            context
        )
        l_slow = F.mse_loss(pred_noise_slow, noise_normal)

        # Combined MixFlow loss
        loss_mixflow = l_normal + self.alpha_slow * l_slow

        return loss_mixflow

    def __compute_gdcw_loss(self, H_denoised, graph_adj):
        batch_size, dim = H_denoised.shape

        if not hasattr(self, '_gdcw_spectral_cache') or self._gdcw_spectral_cache is None:
            self._gdcw_spectral_cache = self.__compute_feature_spectral_transform(dim)

        spectral_u, spectral_eigenvalues = self._gdcw_spectral_cache
        H_projected = H_denoised @ spectral_u

        # Compute correction weights: w_i = (1 + SNR) / (1 + SNR * exp(-lambda_i / sigma^2))
        snr = self.snr_params.mean()
        sigma_sq = self.noise_variance
        correction_weights = (1 + snr) / (1 + snr * torch.exp(-spectral_eigenvalues / (sigma_sq + 1e-8)))

        # Apply correction in spectral domain
        H_corrected_spectral = H_projected * correction_weights.unsqueeze(0)

        H_corrected_back = H_corrected_spectral @ spectral_u.T

        # GDCW loss
        loss_gdcw = F.mse_loss(H_denoised, H_corrected_back)

        return loss_gdcw

    def __compute_feature_spectral_transform(self, dim):

        K = min(dim, max(16, dim // 4))  

        try:
            with torch.no_grad():
                indices = torch.arange(dim, device=self.device).unsqueeze(0)
                distances = torch.abs(indices - indices.T).float()
                feature_sim = torch.exp(-distances / (dim // 4 + 1e-8))

                # Make it symmetric and add identity for numerical stability
                feature_laplacian = torch.eye(dim, device=self.device, dtype=self.dtype) - \
                                    feature_sim / (feature_sim.sum(dim=1, keepdim=True) + 1e-8)

                # Eigendecomposition
                eigenvalues_full, eigenvectors_full = torch.linalg.eigh(feature_laplacian)

                # Keep top-K components
                spectral_u = eigenvectors_full[:, :K].to(self.dtype)
                eigenvalues = eigenvalues_full[:K].to(self.dtype)

        except Exception:
            # Fallback: use identity
            spectral_u = torch.eye(dim, K, device=self.device, dtype=self.dtype)
            eigenvalues = torch.ones(K, device=self.device, dtype=self.dtype)

        return spectral_u, eigenvalues

    def reset_gdcw_cache(self):
        """Reset the GDCW spectral cache (call at start of each epoch)."""
        self._gdcw_spectral_cache = None
        self._gdcw_eigen_cache = None


    def extract(self, student_id, exercise_id, q_mask):
        
        def InfoNCE(view1, view2, temperature):
            """Standard InfoNCE for cross-view alignment."""
            v1 = F.normalize(view1, dim=1)
            v2 = F.normalize(view2, dim=1)
            logits = (v1 @ v2.T) / temperature  # [B, B]
            # Diagonal entries are positives
            return -torch.diag(F.log_softmax(logits, dim=1)).mean()

        def do_struct_forward(right, wrong):
            """Apply LightGCN + fusion.  Returns (stu, exer, know, H_stu, H_exer)."""
            if 'dis' not in self.mode:
                stu_f, exer_f, know_f = self.__common_forward(right, wrong)
            else:
                out = self.convolution(right if 'all' not in self.graph_dict else self.graph_dict['all'])
                stu_f = out[:self.student_num]
                exer_f = out[self.student_num:self.student_num + self.exercise_num]
                know_f = out[self.exercise_num + self.student_num:]
            if 'dis' not in self.mode:
                H_stu = self.struct_fusion_layer(stu_f)
                H_exer = self.struct_fusion_layer(exer_f)
            else:
                H_stu, H_exer = stu_f, exer_f
            return stu_f, exer_f, know_f, H_stu, H_exer

        batch_size = student_id.size(0)
        orig_right = self.graph_dict['right']
        orig_wrong = self.graph_dict['wrong']
        stu_orig, exer_orig, know_orig, H_orig_stu, H_orig_exer = do_struct_forward(orig_right, orig_wrong)

        zero_loss = torch.tensor(0.0, device=H_orig_stu.device, dtype=H_orig_stu.dtype)
        l_stage1 = zero_loss

        flip_right = self.graph_dict.get('right_flip')
        flip_wrong = self.graph_dict.get('wrong_flip')
        if flip_right is not None and flip_wrong is not None:

            _, _, _, H_flip_stu, H_flip_exer = do_struct_forward(flip_right, flip_wrong)

            l_flip_stu = InfoNCE(H_orig_stu[student_id], H_flip_stu[student_id],
                                 self.flip_temp)
            l_flip_exer = InfoNCE(H_orig_exer[exercise_id], H_flip_exer[exercise_id],
                                  self.flip_temp)
            l_flip = l_flip_stu + l_flip_exer
        else:

            l_flip = zero_loss

        llm_stu_batch = self.__llm_stu_emb[student_id]   # [B, D]
        llm_exer_batch = self.__llm_exer_emb[exercise_id] # [B, D]
        v_s = self.pca_projection_student(llm_stu_batch)
        v_e = self.pca_projection_exercise(llm_exer_batch)


        l_stage2_stu = InfoNCE(v_s, H_orig_stu[student_id], self.stage2_temp)
        l_stage2_exer = InfoNCE(v_e, H_orig_exer[exercise_id], self.stage2_temp)
        l_stage2 = l_stage2_stu + l_stage2_exer

        l_semantic = l_stage2

        H_fused_stu = H_orig_stu
        H_fused_exer = H_orig_exer

        if 'all' in self.graph_dict:
            graph_adj = self.graph_dict['all']
        else:
            graph_adj = self.graph_dict['right']

        cache_key = (H_fused_stu.shape[0], H_fused_exer.shape[0])
        if not hasattr(self, '_gcn_cache_key') or self._gcn_cache_key != cache_key or not self._cache_valid:
            all_nodes = torch.cat([H_fused_stu, H_fused_exer, know_orig], dim=0)
            gcn_full = self.g_sae_gcn_proj(torch.sparse.mm(graph_adj, all_nodes))
            self._gcn_full_cache = gcn_full.detach()
            self._gcn_cache_key = cache_key
            self._cache_valid = True
        gcn_full = self._gcn_full_cache
        gcn_stu_full = gcn_full[:self.student_num]
        gcn_exer_full = gcn_full[self.student_num:self.student_num + self.exercise_num]
        gcn_cache_stu = gcn_stu_full[student_id]
        gcn_cache_exer = gcn_exer_full[exercise_id]

        H_batch_stu = H_fused_stu[student_id]
        H_batch_exer = H_fused_exer[exercise_id]
        l_g_sae = (self.__compute_g_sae_loss(H_batch_stu, graph_adj, gcn_cache=gcn_cache_stu) +
                   self.__compute_g_sae_loss(H_batch_exer, graph_adj, gcn_cache=gcn_cache_exer))

        if not self._cache_valid or self._last_H_fused_stu_shape != H_fused_stu.shape:
            self._compute_channel_partition_and_attribution(H_fused_stu, H_fused_exer, graph_adj)
            self._last_H_fused_stu_shape = H_fused_stu.shape

        F_noise_stu, attr_stu = self._cached_F_noise_stu, self._cached_attr_stu
        F_noise_exer, attr_exer = self._cached_F_noise_exer, self._cached_attr_exer

        l_mixflow = (self.__compute_mixflow_loss(H_batch_stu, F_noise_stu, attr_stu) +
                     self.__compute_mixflow_loss(H_batch_exer, F_noise_exer, attr_exer))

        s_stu, _ = self.__g_sae_encode(H_batch_stu, graph_adj, gcn_cache=gcn_cache_stu)
        H_denoised_stu = self.__g_sae_decode(s_stu)
        s_exer, _ = self.__g_sae_encode(H_batch_exer, graph_adj, gcn_cache=gcn_cache_exer)
        H_denoised_exer = self.__g_sae_decode(s_exer)
        l_gdcw = (self.__compute_gdcw_loss(H_denoised_stu, graph_adj) +
                  self.__compute_gdcw_loss(H_denoised_exer, graph_adj))

        extra_loss = (self.stage2_weight * (l_stage2 + l_g_sae + l_mixflow + l_gdcw)
                      + self.flip_weight * l_flip)


        if 'tf' not in self.mode:
            student_ts = self.transfer_student_layer(stu_orig[student_id])
            diff_ts = self.transfer_exercise_layer(exer_orig[exercise_id])
            knowledge_ts = self.transfer_knowledge_layer(know_orig)
        else:
            student_ts = stu_orig[student_id]
            diff_ts = exer_orig[exercise_id]
            knowledge_ts = know_orig

        disc_ts = self.__disc_emb(exercise_id)
        knowledge_impact_ts = self.__knowledge_impact_emb(exercise_id)

        return student_ts, diff_ts, disc_ts, knowledge_ts, {
            'extra_loss': extra_loss,
            'knowledge_impact': knowledge_impact_ts,
            'l_stage1': l_stage1.detach(),          
            'l_stage2': l_stage2.detach(),          
            'l_semantic': l_semantic.detach(),      
            'l_flip': l_flip.detach(),             
            'l_g_sae': l_g_sae.detach(),
            'l_mixflow': l_mixflow.detach(),
            'l_gdcw': l_gdcw.detach()
        }

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
        se_graph_right_flip, se_graph_wrong_flip = [
            self.__create_adj_se(response_flip, is_subgraph=True)[i] for i in range(2)]
        ek_graph = self.graph_dict['Q_Matrix']
        self.graph_dict['right_flip'] = self.__final_graph(se_graph_right_flip, ek_graph)
        self.graph_dict['wrong_flip'] = self.__final_graph(se_graph_wrong_flip, ek_graph)

    def __getitem__(self, item):
        """Retrieve specific embedding attribute."""
        if item not in self.__emb_map.keys():
            raise ValueError(f"We can only detach {self.__emb_map.keys()} from embeddings.")

        if 'dis' not in self.mode:
            stu_forward, exer_forward, know_forward = self.__common_forward(
                self.graph_dict['right'], self.graph_dict['wrong'])
        else:
            out = self.convolution(self.graph_dict['all'])
            stu_forward = out[:self.student_num]
            exer_forward = out[self.student_num:self.student_num + self.exercise_num]
            know_forward = out[self.exercise_num + self.student_num:]

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
        """Convert scipy sparse matrix to PyTorch sparse tensor."""
        coo = sp_mat.tocoo().astype(np.float64)
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape, dtype=torch.float64).coalesce()

    def __create_adj_se(self, np_response, is_subgraph=False):

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

        sek_num = self.student_num + self.exercise_num + self.knowledge_num
        se_num = self.student_num + self.exercise_num
        tmp = np.zeros(shape=(sek_num, sek_num))
        tmp[:self.student_num, self.student_num:se_num] = se
        tmp[self.student_num:se_num, se_num:sek_num] = ek
        graph = tmp + tmp.T + np.identity(sek_num)
        graph = sp.csr_matrix(graph)

        # Symmetric normalization
        rowsum = np.array(graph.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(graph)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return self.__sp_mat_to_sp_tensor(adj_matrix).to(self.device)


def inner_product(a, b):
    """Compute inner product between two tensors."""
    return torch.sum(a * b, dim=-1)
