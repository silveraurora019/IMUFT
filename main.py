# 文件名: new_main.py
import logging
import torch
import torch.nn as nn
import os
import numpy as np
import random
import argparse
import copy
from pathlib import Path
from typing import List, Dict, Any

# 导入 fedu/utils.py (假设已复制到同一目录)
from utils import set_for_logger 
# 导入 fedu/dataloaders (假设已复制)
from dataloaders import build_dataloader
# 导入新的损失函数
from loss import DiceLoss, JointLoss, FairFedULoss
import torch.nn.functional as F
# 导入新的模型构建器
from nets import build_model
# 导入新的客户端 UFT 正则化器
from client_regularizers import UncertaintyTax
# 导入新的组合服务器
from combined_server import CombinedServer


@torch.no_grad()
def get_client_features(local_models, dataloaders, device):
    """
    (来自 FedIGMS)
    从所有客户端的验证数据加载器中提取特征 (z 和 shadow)。
    """
    client_feats_list = []
    for model, loader in zip(local_models, dataloaders):
        model.eval()
        all_z = []
        all_shadow_flat = [] # <--- 新增
        
        if len(loader) == 0:
            logging.warning(f"一个客户端的验证/特征加载器为空。")
            client_feats_list.append(torch.empty(0, 1)) # 添加一个带无效维度的空张量
            continue

        try:
            for x, target in loader: # 迭代整个验证集
                x = x.to(device)
                # 假设是 UNet_pro，它返回 (output, z, shadow)
                _, z, shadow = model(x) # <--- 接收 shadow
                
                # --- 将 shadow 展平 ---
                # shadow 的形状是 [B, C, H, W]，z 的形状是 [B, D]
                # 我们用平均池化将其变为 [B, C]
                shadow_flat = F.adaptive_avg_pool2d(shadow, (1, 1)).view(shadow.shape[0], -1)
                # --- 展平结束 ---
                
                all_z.append(z.cpu())
                all_shadow_flat.append(shadow_flat.cpu()) # <--- 存储
            
            if len(all_z) > 0:
                client_z = torch.cat(all_z, dim=0)
                client_shadow = torch.cat(all_shadow_flat, dim=0)
                
                # --- 关键：拼接特征 ---
                combined_features = torch.cat([client_z, client_shadow], dim=1)
                client_feats_list.append(combined_features)
                # --- 拼接结束 ---
                
            else:
                # 这可能发生在 loader 不为空，但所有 batch 都被跳过（例如，如果数据损坏）
                logging.warning(f"一个客户端的验证加载器未产生任何特征。")
                client_feats_list.append(torch.empty(0, 1))

        except Exception as e:
             logging.error(f"提取特征时出错: {e}")
             client_feats_list.append(torch.empty(0, 1))
             
    # 检查是否所有客户端都成功提取了特征
    if not client_feats_list or any(f.shape[0] == 0 or f.dim() != 2 for f in client_feats_list):
        logging.error("未能从一个或多个客户端提取有效特征。中止相似度计算。")
        return None

    # 检查特征维度是否一致
    try:
        # 找到第一个有效特征的维度
        first_valid_feat = next(f for f in client_feats_list if f.shape[0] > 0)
        feat_dim = first_valid_feat.shape[1]
        
        if not all(f.shape[1] == feat_dim for f in client_feats_list if f.shape[0] > 0):
            logging.warning("客户端之间的特征维度不匹配。")
            # 过滤掉维度不匹配的
            client_feats_list_filtered = [f for f in client_feats_list if f.dim() == 2 and f.shape[1] == feat_dim]
            if not client_feats_list_filtered:
                logging.error("特征维度检查后没有剩余客户端。")
                return None
            return client_feats_list_filtered
    except StopIteration:
        logging.error("特征列表为空或所有客户端均无有效特征。")
        return None

    return client_feats_list


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    
    parser.add_argument('--data_root', type=str, required=False, 
                        default="E:/A_Study_Materials/Dataset/Prostate", 
                        help="Data directory (指向 fundus 或 prostate)")
    
    parser.add_argument('--dataset', type=str, default='prostate', 
                        help="Dataset type: 'fundus' (4 站点) 或 'prostate' (6 站点)")
    
    # 强制使用 unet_pro，因为两种方法都需要 z 特征
    parser.add_argument('--model', type=str, default='unet_pro', 
                        help='Model type (unet or unet_pro). 必须是 unet_pro。')

    parser.add_argument('--rounds', type=int, default=200, help='number of maximum communication round')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')

    parser.add_argument('--log_dir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--save_dir', type=str, required=False, default="./weights/", help='Log directory path')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="L2 regularization strength")
    parser.add_argument('--batch-size', type=int, default=8, help='input batch size for training')
    parser.add_argument('--experiment', type=str, default='FedU-MINE-Mix', help='Experiment name')

    parser.add_argument('--test_step', type=int, default=1)
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate for the model (if unet_pro)')

    # --- FedU (UFT) 参数 ---
    parser.add_argument('--lambda_uft', type=float, default=0.1, 
                        help='Weight for the UFT regularizer (lambda_2 in paper)')
    parser.add_argument('--uft_beta_u', type=float, default=1.0, 
                        help='Uncertainty tax coefficient (beta_u) for UFT')

    # --- FedIGMS (MI-Sim) 参数 ---
    parser.add_argument('--rad_gamma', type=float, default=1.0, help='Gamma for RAD similarity')
    parser.add_argument('--mine_hidden', type=int, default=128, help='Hidden layer size for MINE estimator')
    parser.add_argument('--lr_mine', type=float, default=1e-4, help='Learning rate for MINE estimator')
    parser.add_argument('--alpha_init', type=float, default=0.5, help='Initial alpha value for mixed similarity')

    # --- 组合算法参数 ---
    parser.add_argument('--sim_start_round', type=int, default=0, 
                        help='Round to start using similarity aggregation (use FedAvg before)')

    args = parser.parse_args()
    return args

def communication(server_model, models, client_weights):
    """ 标准 FedAvg 聚合 (参数聚合) """
    with torch.no_grad():
        device = next(server_model.parameters()).device
        if not isinstance(client_weights, torch.Tensor):
            client_weights = torch.tensor(client_weights, dtype=torch.float32, device=device)
        else:
            client_weights = client_weights.to(device)
            
        if not torch.isclose(client_weights.sum(), torch.tensor(1.0)):
             logging.warning(f"Client weights do not sum to 1. Sum={client_weights.sum()}. Normalizing.")
             client_weights = client_weights / client_weights.sum()

        for key in server_model.state_dict().keys():
            if 'num_batches_tracked' in key:
                continue
            temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32, device=device)
            for client_idx in range(len(client_weights)):
                temp += client_weights[client_idx] * models[client_idx].state_dict()[key].to(device)
            server_model.state_dict()[key].data.copy_(temp)
            
    return server_model

def train(cid: int, 
          model: nn.Module, 
          dataloader: torch.utils.data.DataLoader, 
          device: torch.device, 
          optimizer: torch.optim.Optimizer, 
          epochs: int, 
          loss_func: FairFedULoss, # <-- 使用新的损失包装器
          global_mean_entropy: torch.Tensor # <-- 接收全局平均熵
         ) -> float:
    """
    修改后的训练函数 (来自 fedu/main.py)
    返回: 该客户端在本轮训练中的平均预测熵 (U_i)
    """
    model.train()
    is_unet_pro = model.__class__.__name__ == 'UNet_pro'
    
    client_mean_entropies = [] # 存储每个 batch 的平均熵
    
    for epoch in range(epochs):
        loss_all = 0.
        task_loss_all = 0.
        reg_loss_all = 0.
        
        if len(dataloader) == 0:
            logging.warning(f"Client {cid} training dataloader is empty.")
            continue
            
        for x, target in dataloader:
            x = x.to(device)
            target = target.to(device)
            
            if is_unet_pro:
                output, _, _ = model(x) 
            else:
                output = model(x) # 假设总是 unet_pro
                
            optimizer.zero_grad()
            
            # --- FairFed-U 修改 ---
            # 1. 计算总损失 (L_task + lambda * R_UFT)
            loss, task_loss, reg_loss, U_i_batch = loss_func(output, target, global_mean_entropy)
            # ---------------------
            
            loss_all += loss.item()
            task_loss_all += task_loss.item()
            reg_loss_all += reg_loss.item()
            client_mean_entropies.append(U_i_batch)
            
            loss.backward()
            optimizer.step()
        
        if len(dataloader) > 0:
            avg_loss = loss_all / len(dataloader)
            avg_task_loss = task_loss_all / len(dataloader)
            avg_reg_loss = reg_loss_all / len(dataloader)
            logging.info(f'Client: [{cid}] Epoch: [{epoch}] '
                         f'L_total: {avg_loss:.4f} (L_task: {avg_task_loss:.4f} + R_UFT: {avg_reg_loss:.4f})')

    # 返回本客户端在本轮的平均熵
    if not client_mean_entropies:
        logging.warning(f"Client {cid} did not train (no data or no batches). Returning 0.0 entropy.")
        return 0.0
    return torch.stack(client_mean_entropies).mean().item()


def test(model, dataloader, device, loss_func):
    """
    测试函数 (评估)
    loss_func: 这里传入的是原始的任务损失 (例如 JointLoss)
    """
    model.eval()
    loss_all = 0
    test_acc = 0
    is_unet_pro = model.__class__.__name__ == 'UNet_pro'
    
    if len(dataloader) == 0:
        logging.warning("Test/Val dataloader is empty.")
        return 0.0, 0.0 # 返回 0 避免除零错误

    with torch.no_grad():
        for x, target in dataloader:
            x = x.to(device)
            target = target.to(device)
            if is_unet_pro:
                output, _, _ = model(x)
            else:
                output = model(x)
                
            # 使用任务损失函数计算损失
            loss = loss_func(output, target)
            loss_all += loss.item()
            # 使用 DiceLoss 实例计算评估指标
            test_acc += DiceLoss().dice_coef(output, target) # .item() 在 DiceLoss().dice_coef 内部处理
        
    acc = test_acc / len(dataloader)
    loss = loss_all / len(dataloader)
    return loss, acc.item()

def main(args):
    set_for_logger(args)
    logging.info(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device)

    # 2. 动态定义客户端列表
    if args.dataset == 'fundus':
        clients = ['site1', 'site2', 'site3', 'site4']
    elif args.dataset == 'prostate':
        clients = ['site1', 'site2', 'site3', 'site4', 'site5', 'site6']
    else:
        raise ValueError(f"Unknown client list for dataset: {args.dataset}")

    # 3. build dataset (传入 clients)
    train_dls, val_dls, test_dls, client_weight = build_dataloader(args, clients)
    
    # 基础权重 (例如，按样本量)
    client_weight_tensor = torch.tensor(client_weight, dtype=torch.float32, device=device)

    # 4. build model (传入 clients)
    local_models, global_model = build_model(args, clients, device)
    
    if args.model != 'unet_pro':
        logging.error("This combined method requires 'unet_pro' model to extract features.")
        return

    n_classes = 2
    if args.dataset == 'prostate': n_classes = 2
    elif args.dataset == 'fundus': n_classes = 2
    elif args.dataset == 'pmri': n_classes = 3 

    # ------------------------------------------------------------------
    # 初始化 组合 模块 (UFT + FedIGMS-Mixer)
    # ------------------------------------------------------------------
    
    # (获取 unet_pro 的 z + shadow 特征维度)
    try:
        # 假设输入 384x384
        dummy_input = torch.randn(2, 3, 384, 384).to(device) 
        _, z_dummy, shadow_dummy = global_model(dummy_input)
        # FedIGMS 期望的特征是 z 和 shadow (avg_pool) 的拼接
        shadow_dummy_flat = F.adaptive_avg_pool2d(shadow_dummy, (1, 1)).view(shadow_dummy.shape[0], -1)
        feat_dim = z_dummy.shape[1] + shadow_dummy_flat.shape[1] # 拼接后的维度
        logging.info(f"Detected feature dimension (z + shadow_flat) = {feat_dim}")
    except Exception as e:
        logging.error(f"Could not determine feature dimension: {e}")
        # (UNet_pro (init_features=32) -> z(2048) + shadow(32*1*1=32) -> 2080 ??)
        # (unet.py: z=F.adaptive_avg_pool2d(bottleneck,2).view(bottleneck.shape[0],-1))
        # bottleneck = 32*16=512. avg_pool(2) -> 512*2*2 = 2048.
        # shadow = dec1 (features=32). avg_pool(1) -> 32.
        feat_dim = 2048 + 32 
        logging.warning(f"Failed to infer feat_dim, defaulting to {feat_dim}")

    # 1. 组合服务器
    server_aggregator = CombinedServer(
        beta_u=args.uft_beta_u,         # UFT 参数
        feat_dim=feat_dim,              # FedIGMS 参数
        rad_gamma=args.rad_gamma,       # FedIGMS 参数
        mine_hidden=args.mine_hidden,   # FedIGMS 参数
        lr_mine=args.lr_mine,         # FedIGMS 参数
        alpha_init=args.alpha_init,   # FedIGMS 参数
        device=device
    )
    
    # 2. 客户端 UFT 正则化器
    client_regularizer = UncertaintyTax(beta_u=args.uft_beta_u).to(device)

    # 3. 任务损失 (评估用)
    task_loss_fn = JointLoss(n_classes=n_classes).to(device) 

    # 4. 包装总损失函数 (训练用)
    loss_fun = FairFedULoss(
        task_loss_fn=task_loss_fn,
        uft_regularizer=client_regularizer,
        lambda_uft=args.lambda_uft # UFT 损失权重
    ).to(device)

    # 5. 全局平均熵 (初始为0，服务器广播)
    global_mean_entropy = torch.tensor(0.0, device=device)

    logging.info(f"Combined FedU(UFT) + FedIGMS(Mixer) initialized. ")
    logging.info(f"UFT Params: lambda_uft={args.lambda_uft}, uft_beta_u={args.uft_beta_u}")
    logging.info(f"IGMS Params: alpha_init={args.alpha_init}, feat_dim={feat_dim}")
    # ------------------------------------------------------------------

    # (Optimizer)
    optimizer = []
    for id in range(len(clients)):
        optimizer.append(torch.optim.Adam(local_models[id].parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99)))

    # (训练循环)
    best_dice = 0
    best_dice_round = 0
    best_local_dice = []
    
    # 用于 alpha 更新的验证集反馈
    last_avg_val_dice_tensor = None 

    weight_save_dir = os.path.join(args.save_dir, args.experiment)
    Path(weight_save_dir).mkdir(parents=True, exist_ok=True)
    logging.info('checkpoint will be saved at {}'.format(weight_save_dir))

    
    for r in range(args.rounds):

        logging.info('-------- Commnication Round: %3d --------'%r)
        
        client_reports_entropy = []
        temp_locals = copy.deepcopy(local_models) # 用于聚合

        # 1. 本地训练 (并收集 U_i)
        for idx, client in enumerate(clients):
            # 客户端模型从上一轮的全局模型开始
            local_models[idx].load_state_dict(global_model.state_dict())
            
            # 训练，并获取客户端的平均熵 U_i
            client_mean_U_i = train(
                idx, 
                local_models[idx], 
                train_dls[idx], 
                device, 
                optimizer[idx], 
                args.epochs, 
                loss_fun, 
                global_mean_entropy # 传入上一轮的全局平均熵
            )
            
            # 存储报告
            client_reports_entropy.append(client_mean_U_i) # 存储 U_i (用于UFT)
            
            # 准备聚合 (使用训练后的模型)
            temp_locals[idx].load_state_dict(local_models[idx].state_dict())

        
        # 2. 提取特征 (用于 FedIGMS)
        # (使用验证集数据 和 *训练后* 的本地模型)
        logging.info("Extracting client features for similarity calculation...")
        client_features = get_client_features(temp_locals, val_dls, device)

        # 3. 服务器聚合
        aggr_weights = None
        S_rad, S_mi = None, None # 用于 alpha 更新
        
        if r >= args.sim_start_round and client_features is not None:
            logging.info('Calculating Combined (FedIGMS-Mixer + UFT-Tax) weights...')
            
            # 3b. 计算组合权重
            aggr_weights, stats, S_rad, S_mi, current_alpha = server_aggregator.compute_weights(
                client_features, 
                client_reports_entropy, 
                client_weight # 传入基础权重 (list of floats)
            )
            aggr_weights = aggr_weights.to(device)
            
            logging.info(f"Current Alpha (RAD-vs-MINE): {current_alpha:.4f}")
            logging.info(f"Stats (w_sim): {[f'{w:.4f}' for w in stats['w_sim'].cpu().numpy()]}")
            logging.info(f"Stats (tau_tax): {[f'{tau:.4f}' for tau in stats['tau'].cpu().numpy()]}")
            logging.info(f"Final Aggr Weights: {[f'{w:.4f}' for w in aggr_weights.cpu().numpy()]}")
            
            # 3c. 更新下一轮的全局平均熵
            global_mean_entropy = server_aggregator.prev_mean_entropy.to(device)
            logging.info(f'New Global Mean Entropy (for Rnd {r+1}): {global_mean_entropy.item():.4f}')

        else: 
            # 3d. 早期轮次或特征提取失败时使用 FedAvg
            if r < args.sim_start_round:
                logging.info('Using standard FedAvg aggregation (pre-start round).')
            else:
                logging.warning('Feature extraction failed. Falling back to FedAvg.')
                
            aggr_weights = client_weight_tensor
            
            # (仍然需要计算下一轮的全局平均熵)
            if client_reports_entropy:
                global_mean_entropy = torch.tensor(np.mean(client_reports_entropy), device=device)
            logging.info(f'New Global Mean Entropy (for Rnd {r+1}): {global_mean_entropy.item():.4f}')

        
        # 3e. 执行聚合
        communication(global_model, temp_locals, aggr_weights)

        # 4. 分发全局模型 (已在下一轮训练开始时完成)


        if r % args.test_step == 0:
            # 5. 测试 (使用 *聚合后* 的全局模型在 *测试集* 上)
            avg_loss = []
            avg_dice = []
            logging.info(f"--- Testing Global Model (Round {r}) on TEST Set ---")
            
            for idx, client in enumerate(clients):
                local_models[idx].load_state_dict(global_model.state_dict())
                loss, dice = test(local_models[idx], test_dls[idx], device, task_loss_fn)
                logging.info('Client: %s  test_loss:  %f   test_acc:  %f '%(client, loss, dice))
                avg_dice.append(dice)
                avg_loss.append(loss)

            avg_dice_v = sum(avg_dice) / len(avg_dice) if len(avg_dice) > 0 else 0
            avg_loss_v = sum(avg_loss) / len(avg_loss) if len(avg_loss) > 0 else 0
            logging.info('Round: [%d]  avg_test_loss: %f avg_test_acc: %f std_test_acc: %f'%(r, avg_loss_v, avg_dice_v, np.std(np.array(avg_dice))))

            # --- 6. 在 *验证集* 上评估 (用于 Alpha 更新) ---
            avg_val_dice = []
            logging.info(f"--- Testing Global Model (Round {r}) on VALIDATION Set ---")
            for idx, client in enumerate(clients):
                # model 已经加载了 global_model
                _, val_dice = test(local_models[idx], val_dls[idx], device, task_loss_fn)
                avg_val_dice.append(val_dice)
                logging.info('Client: %s  val_acc:  %f '%(client, val_dice))
            
            current_avg_val_dice_tensor = torch.tensor(avg_val_dice, device=device, dtype=torch.float32)
            logging.info('Round: [%d]  avg_val_acc (for alpha feedback): %f'%(r, current_avg_val_dice_tensor.mean().item()))
            # --- 评估结束 ---


            # 7. 更新 Alpha (使用验证集性能提升)
            if r >= args.sim_start_round and S_rad is not None:
                if last_avg_val_dice_tensor is not None: 
                    try:
                        val_improve = current_avg_val_dice_tensor - last_avg_val_dice_tensor 
                        logging.info(f"Updating alpha with VALIDATION feedback. Improvement: {val_improve.cpu().numpy()}")
                        sig_rad, sig_mi, new_alpha = server_aggregator.update_alpha(S_rad, S_mi, val_improve)
                        logging.info(f'Alpha update: sig_rad={sig_rad:.4f}, sig_mi={sig_mi:.4f}, new_alpha={new_alpha:.4f}')
                    except Exception as e:
                        logging.warning(f"Could not update alpha: {e}")
                else:
                    logging.info("Skipping alpha update (first validation round).")
                
            last_avg_val_dice_tensor = current_avg_val_dice_tensor
            # --- Alpha 更新结束 ---

            # 8. 保存最佳模型 (基于测试集性能 avg_dice_v)
            if best_dice < avg_dice_v:
                best_dice = avg_dice_v
                best_dice_round = r
                best_local_dice = avg_dice

                weight_save_path = os.path.join(weight_save_dir, 'best_model.pth')
                torch.save(global_model.state_dict(), weight_save_path)
                logging.info(f"--- Best model saved to {weight_save_path} (Avg Dice: {best_dice:.4f}) ---")
            

    logging.info('-------- Training complete --------')
    logging.info('Best avg TEST dice score %f at round %d '%( best_dice, best_dice_round))
    for idx, client in enumerate(clients):
        logging.info('Client: %s  best_test_acc (at round %d):  %f '%(client, best_dice_round, best_local_dice[idx] if idx < len(best_local_dice) else 0.0))


if __name__ == '__main__':
    args = get_args()
    # 确保模型必须是 unet_pro
    if args.model != 'unet_pro':
        print("Error: This algorithm requires --model unet_pro to extract features.")
    else:
        main(args)