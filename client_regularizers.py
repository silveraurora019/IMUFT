# 文件名: new_client_regularizers.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Callable, Tuple, Dict, Any

class UncertaintyTax(nn.Module):
    """
    机会均等 '不确定性税'（UFT）：基于预测分布的熵，惩罚与全局均值的偏离
    R_UFT = exp(beta_u * U_i) * | U_i - U_bar |
    """
    def __init__(self, beta_u: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.beta_u = beta_u
        assert reduction in ("mean", "sum")
        self.reduction = reduction

    @staticmethod
    @torch.no_grad()
    def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
        """
        对每个样本计算类别熵：H(p) = - sum p log p
        logits: [B, C, ...]（除C外其余维会展平）
        return: scalar 熵均值
        """
        if logits.dim() > 2:
            B = logits.shape[0]; C = logits.shape[1]
            logits = logits.view(B, C, -1).permute(0, 2, 1).reshape(-1, C)
        # 防止 log(0)
        p = torch.softmax(logits, dim=1).clamp_min(1e-12)
        H = (-p * p.log()).sum(dim=1)  # [B'] 样本熵
        
        # 处理可能的 NaN (如果 p 仍然接近0)
        if torch.isnan(H).any():
            return torch.tensor(0.0, device=logits.device)
            
        return H.mean()

    def forward(self, logits: torch.Tensor, U_bar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        logits: 本地前向输出
        U_bar: 全局/本轮的 '平均熵'（服务器广播或上一轮统计）
        return: (R_uft, U_i)
        """
        U_i = self.entropy_from_logits(logits)
        
        # .detach() 确保 U_i 不会引入二阶梯度
        tau_i = torch.exp(self.beta_u * U_i.detach())
        
        # U_bar 可能是从服务器传来，确保在同一设备
        if U_bar.device != U_i.device:
            U_bar = U_bar.to(U_i.device)
            
        R = tau_i * torch.abs(U_i - U_bar.detach())
        
        # R_uft 是损失, U_i.detach() 用于报告
        return R, U_i.detach()

