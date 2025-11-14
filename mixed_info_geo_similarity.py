# 文件名: new_mixed_info_geo_similarity.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging # 导入 logging
from typing import Tuple

############################################################
# 1) RAD：基于SPD流形的协方差对齐（Log-Euclidean 距离 → 相似度）
############################################################

# --- 修改 1: 增大默认的 epsilon ---
def _cov_spd(feats: torch.Tensor, eps: float = 1e-3): 
    """
    feats: [N, D]  (样本×通道/特征)
    返回: 协方差矩阵 SPD: [D, D]
    
    (已修改: 默认 eps 从 1e-5 增加到 1e-3 以处理 N < D 时的奇异矩阵问题)
    """
    X = feats - feats.mean(dim=0, keepdim=True)
    C = (X.T @ X) / max(X.shape[0] - 1, 1)
    # 数值稳定 + 保证正定
    # 较大的 eps (正则化项) 可确保矩阵是正定的
    C = 0.5 * (C + C.T) + eps * torch.eye(C.shape[0], device=C.device)
    return C

def _logm_spd(A: torch.Tensor):
    # 对称特征分解 + log
    try:
        evals, evecs = torch.linalg.eigh(A)
    except torch.linalg.LinAlgError as e:
        # --- 修改 2: 增加备用方案的鲁棒性 ---
        logging.warning(f"torch.linalg.eigh 失败: {e}. 正在使用更大的扰动重试。")
        # 备用方案：处理可能的数值问题
        # (增加一个比 _cov_spd 中 eps 更大的扰动值)
        A = A + 1e-2 * torch.eye(A.shape[0], device=A.device) 
        
        try:
             evals, evecs = torch.linalg.eigh(A)
        except torch.linalg.LinAlgError as e2:
             # 如果仍然失败，则放弃并返回一个零矩阵
             logging.error(f"linalg.eigh 再次失败: {e2}. 返回零矩阵。")
             return torch.zeros_like(A)
        
    evals = torch.clamp(evals, min=1e-10) # 保持最小值钳位
    return (evecs @ torch.diag(torch.log(evals)) @ evecs.T)

def rad_distance(feats_x: torch.Tensor, feats_y: torch.Tensor) -> torch.Tensor:
    """
    返回 RAD 距离（标量）。
    """
    Cx = _cov_spd(feats_x)
    Cy = _cov_spd(feats_y)
    Lx = _logm_spd(Cx)
    Ly = _logm_spd(Cy)
    dist = torch.linalg.norm(Lx - Ly, ord="fro")
    return dist

def rad_similarity(feats_x: torch.Tensor, feats_y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    将距离映射为相似度：exp(-gamma * dist) in (0,1]
    """
    d = rad_distance(feats_x, feats_y)
    return torch.exp(-gamma * d)

############################################################
# 2) MINE-Sim：互信息下界估计（NWJ稳定实现）→ 归一化相似度
############################################################
class SmallCritic(nn.Module):
    def __init__(self, in_dim_x: int, in_dim_y: int, hidden: int = 128):
        super().__init__()
        self.fx = nn.Sequential(nn.Linear(in_dim_x, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.fy = nn.Sequential(nn.Linear(in_dim_y, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.out = nn.Sequential(nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, y):
        x = x.float()
        y = y.float()
        h = self.fx(x) * self.fy(y)
        return self.out(h)

class MINEstimator:
    """
    使用 NWJ（f-GAN下界）稳定版本：
      I(X;Y) >= E_{p(x,y)}[T] - E_{p(x)p(y)}[exp(T-1)]
    """
    def __init__(self, in_dim_x: int, in_dim_y: int, hidden: int = 128, lr: float = 1e-3, device="cpu"):
        self.device = device
        self.net = SmallCritic(in_dim_x, in_dim_y, hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        
    def step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        单步训练：x,y -> [B, Dx/Dy]
        """
        x = x.to(self.device)
        y = y.to(self.device)
        B = x.size(0)
        
        if x.size(0) != y.size(0):
            B = min(x.size(0), y.size(0))
            x = x[:B]
            y = y[:B]
            
        T_joint = self.net(x, y)
        y_perm = y[torch.randperm(B, device=self.device)]
        T_prod = self.net(x, y_perm)
        
        loss = -(T_joint.mean() - torch.exp(T_prod - 1.0).mean())
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

    @torch.no_grad()
    def estimate_mi(self, x: torch.Tensor, y: torch.Tensor, normalize=True) -> torch.Tensor:
        """
        返回 MI 下界估计（标量），可选进行0-1归一化。
        """
        x, y = x.to(self.device), y.to(self.device)
        
        if x.size(0) != y.size(0):
            B = min(x.size(0), y.size(0))
            x = x[:B]
            y = y[:B]
            
        T_joint = self.net(x, y).mean()
        y_perm = y[torch.randperm(y.size(0), device=self.device)]
        T_prod = torch.exp(self.net(x, y_perm) - 1.0).mean()
        mi = T_joint - T_prod
        
        if normalize:
            return torch.sigmoid(mi)
        return mi

############################################################
# 3) 信息几何混合：RAD + MINE-Sim，带可学习权重 alpha∈[0,1]
############################################################
class InfoGeoMixer(nn.Module):
    """
    sim_mix = alpha * sim_rad + (1 - alpha) * sim_mi
    其中 alpha = sigmoid(a)，a为可学习参数（服务器侧优化）
    """
    def __init__(self, in_dim_feat: int, mine_hidden: int = 128, lr_mine: float = 1e-3,
                 rad_gamma: float = 1.0, alpha_init: float = 0.5, device="cpu"):
        super().__init__()
        self.device = device
        self.rad_gamma = rad_gamma
        self.mine = MINEstimator(in_dim_feat, in_dim_feat, hidden=mine_hidden, lr=lr_mine, device=device)
        
        alpha_init = torch.clamp(torch.tensor(alpha_init), min=1e-5, max=0.99999)
        a0 = torch.logit(alpha_init)
        self.alpha_param = nn.Parameter(a0.to(device))
        self.alpha_opt = torch.optim.SGD([self.alpha_param], lr=5e-3, momentum=0.9)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_param)

    def rad_sim(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        X_rad = X.to(dtype=torch.float64, device=self.device)
        Y_rad = Y.to(dtype=torch.float64, device=self.device)
        return rad_similarity(X_rad, Y_rad, gamma=self.rad_gamma).to(dtype=torch.float32, device=self.device)

    def mine_train_step(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        return self.mine.step(X, Y)

    @torch.no_grad()
    def mine_sim(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return self.mine.estimate_mi(X, Y, normalize=True).to(dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def mixed_similarity(self, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_rad = self.rad_sim(X, Y)
        s_mi  = self.mine_sim(X, Y)
        a = self.alpha
        return a * s_rad + (1 - a) * s_mi, s_rad, s_mi

    def update_alpha(self, signal_rad: float, signal_mi: float):
        """
        用“外部一致性信号”更新 alpha。
        """
        self.alpha_opt.zero_grad()
        direction = -1.0 if signal_mi > signal_rad else 1.0
        loss = -direction * self.alpha_param 
        loss.backward()
        self.alpha_opt.step()
        
        with torch.no_grad():
            self.alpha_param.clamp_(-5.0, 5.0)