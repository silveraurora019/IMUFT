# 文件名: new_combined_server.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional

# 导入 FedIGMS 聚合器
from aggregator_mixed_sim import MixedSimAggregator

class CombinedServer:
    """
    服务器侧：联合 'FedIGMS 混合相似度' + 'UFT 不确定性税'
    
    - FedIGMS (Sim):
        * S_mix = alpha*RAD + (1-alpha)*MINE
        * w_sim_i ∝ Σ_j S_mix(i,j)
    - UFT (Tax):
        * τ_i = exp(beta_u * U_i)
    - 权重:
        w_i ∝ base_i * w_sim_i * (1 + τ_i)^(-1)
    """

    def __init__(
        self,
        # UFT 参数
        beta_u: float = 1.0,      
        # FedIGMS 聚合器参数
        feat_dim: int = 2560, # (z=2048 + shadow=512)
        rad_gamma: float = 1.0,
        mine_hidden: int = 128,
        lr_mine: float = 1e-3,
        alpha_init: float = 0.5,
        # 其他
        clamp_small: float = 1e-8,
        device: str = "cpu"
    ):
        self.beta_u = float(beta_u)
        self.eps = float(clamp_small)
        self.device = device
        
        # 实例化 FedIGMS 聚合器
        self.aggregator = MixedSimAggregator(
            feat_dim=feat_dim,
            rad_gamma=rad_gamma,
            mine_hidden=mine_hidden,
            lr_mine=lr_mine,
            device=device
        )
        # 初始化 Alpha
        with torch.no_grad():
            self.aggregator.mixer.alpha_param.fill_(torch.logit(torch.tensor(alpha_init)))

        # 存储上一轮的平均熵，用于广播
        self.prev_mean_entropy = torch.tensor(0.0, device=device) 

    @staticmethod
    def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        s = v.sum()
        if s < eps:
            return torch.full_like(v, 1.0 / v.shape[0])
        return v / (s + eps)

    def compute_weights(
        self,
        client_features_list: List[torch.Tensor], # FedIGMS 所需
        U_entropy_list: List[float],             # UFT 所需
        base_weights: List[float]                # 基础权重 (例如样本量)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, float]:
        """
        计算联合权重 w_i
        返回: (w, stats, S_rad, S_mi, alpha)
        """
        M = len(U_entropy_list)
        if M == 0:
            return torch.tensor([]), {}, torch.tensor([]), torch.tensor([]), 0.0
            
        device = self.device
        
        # --- 1. FedIGMS 相似度部分 ---
        S_mix, S_rad, S_mi, current_alpha = self.aggregator.compute_similarity_matrix(client_features_list)
        
        # 使用 FedIGMS 相似度矩阵计算基础的 sim_weight
        # (我们使用带温度的 softmax 来锐化权重)
        w_sim = self.aggregator.weights_from_similarity(S_mix, temperature=0.1) # [M]

        # --- 2. UFT 税收部分 ---
        U_entropy_list_safe = U_entropy_list if U_entropy_list else [0.0] * M
        Ue = torch.tensor(U_entropy_list_safe, dtype=torch.float32, device=device)  # [M]
        # UFT: 税 τ_i
        tau = torch.exp(self.beta_u * Ue)  # [M]
        
        # --- 3. 基础权重 ---
        base_weights_safe = base_weights if base_weights else [1.0/M] * M
        base = torch.tensor(base_weights_safe, dtype=torch.float32, device=device)  # [M]

        # --- 4. 组合权重 ---
        # w_i ∝ base * w_sim * (1 / (1 + tau))
        tau = torch.clamp(tau, min=0.0)
        w_raw = base * w_sim * (1.0 / (1.0 + tau))  # [M]
        
        w = self.normalize(w_raw, eps=self.eps)

        stats = {
            "w_sim": w_sim.detach(),
            "tau": tau.detach(),
            "weights_raw": w_raw.detach(),
            "weights": w.detach(),
            "Ue": Ue.detach()
        }
        
        # 记录与广播用的平均熵
        if Ue.numel() > 0:
            self.prev_mean_entropy = Ue.mean().detach()
        else:
            self.prev_mean_entropy = torch.tensor(0.0, device=device)
            
        return w, stats, S_rad, S_mi, current_alpha

    def update_alpha(self, S_rad: torch.Tensor, S_mi: torch.Tensor, val_improve: torch.Tensor) -> Tuple[float, float, float]:
        """
        包装器，用于调用 FedIGMS 聚合器的 alpha 更新
        """
        return self.aggregator.update_alpha_from_feedback(S_rad, S_mi, val_improve)