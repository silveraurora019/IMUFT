# dataloaders/__init__.py
from .rif import RIF
# 确保您已经创建了这个文件
from .prostate_nifti_dataset import ProstateNIFTIDataset 
import os
from torch.utils.data import DataLoader
import logging
import torch

# 关键：确保这里接受 (args, clients) 两个参数
def build_dataloader(args, clients): 

    # 3. 选择 Dataset 类
    if args.dataset == 'fundus':
        DatasetClass = RIF
    elif args.dataset == 'prostate':
        DatasetClass = ProstateNIFTIDataset
    else:
        logging.error(f"Unknown dataset: {args.dataset}")
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_dls = []
    val_dls = []
    test_dls = []
    dataset_lens = []

    for idx, client in enumerate(clients):
        # 4. 实例化 6:2:2 划分
        # (移除 isVal, 使用 split='val' 和 split='test')
        train_set = DatasetClass(client_idx=idx, base_path=args.data_root,
                                 split='train')
        valid_set = DatasetClass(client_idx=idx, base_path=args.data_root,
                                 split='val')
        test_set = DatasetClass(client_idx=idx, base_path=args.data_root,
                                 split='test')
 
        logging.info('{} train  dataset (60%): {}'.format(client, len(train_set)))
        logging.info('{} val    dataset (20%): {}'.format(client, len(valid_set)))
        logging.info('{} test   dataset (20%): {}'.format(client, len(test_set)))
 
        # ... (DataLoader 创建不变) ...
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               shuffle=True, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size,
                                               shuffle=False, drop_last=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                              shuffle=False, drop_last=False)

        train_dls.append(train_loader)
        val_dls.append(valid_loader)
        test_dls.append(test_loader)

        dataset_lens.append(len(train_set))
    
    # ... (客户端权重计算不变) ...
    client_weight = []
    total_len = sum(dataset_lens)
    if total_len > 0: 
        for i in dataset_lens:
            client_weight.append(i / total_len)
    else:
        logging.warning("Total dataset length is zero. Using uniform weights.")
        client_weight = [1.0 / len(clients)] * len(clients)

    return train_dls, val_dls, test_dls, client_weight



# # dataloaders/__init__.py
# from .rif import RIF
# # 确保您已经创建了这个文件
# from .prostate_nifti_dataset1 import ProstateNIFTIDataset 
# import os
# from torch.utils.data import DataLoader
# import logging
# import torch
# from torchvision.transforms import v2 as T

# # 关键：确保这里接受 (args, clients) 两个参数
# def build_dataloader(args, clients): 

#     # 3. 选择 Dataset 类
#     if args.dataset == 'fundus':
#         DatasetClass = RIF
#     elif args.dataset == 'prostate':
#         DatasetClass = ProstateNIFTIDataset
#         # --- 新增：为前列腺训练集定义数据增强 ---
#         # (这些v2变换可以同时处理图像和掩码)
#         train_transform = T.Compose([
#             # 弹性形变是MRI分割中非常有效的方法，模拟软组织非刚性形变。alpha=60.0 控制变形的强度，sigma=6.0 控制变形的平滑程度。
#             T.ElasticTransform(alpha=40.0, sigma=5.0),
#             # (新增) 添加强度抖动 (v2会自动只应用在图像上)
#             T.ColorJitter(brightness=0.2, contrast=0.2),            
#             # (新增) 添加高斯模糊 (v2会自动只应用在图像上)
#             T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
#         ])
#         # 验证集和测试集不应使用随机增强
#         val_test_transform = None 
#         # --- 增强定义结束 ---
#     else:
#         logging.error(f"Unknown dataset: {args.dataset}")
#         raise ValueError(f"Unknown dataset: {args.dataset}")
    



#     train_dls = []
#     val_dls = []
#     test_dls = []
#     dataset_lens = []

#     for idx, client in enumerate(clients):
#         # 4. 实例化 6:2:2 划分
#         # (移除 isVal, 使用 split='val' 和 split='test')
#         train_set = DatasetClass(client_idx=idx, base_path=args.data_root,
#                                  split='train', transform=train_transform)
#         valid_set = DatasetClass(client_idx=idx, base_path=args.data_root,
#                                  split='val', transform=val_test_transform)
#         test_set = DatasetClass(client_idx=idx, base_path=args.data_root,
#                                  split='test', transform=val_test_transform)
 
#         logging.info('{} train  dataset (60%): {}'.format(client, len(train_set)))
#         logging.info('{} val    dataset (20%): {}'.format(client, len(valid_set)))
#         logging.info('{} test   dataset (20%): {}'.format(client, len(test_set)))
 
#         # ... (DataLoader 创建不变) ...
#         train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
#                                                shuffle=True, drop_last=True)
#         valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size,
#                                                shuffle=False, drop_last=False)
#         test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
#                                               shuffle=False, drop_last=False)

#         train_dls.append(train_loader)
#         val_dls.append(valid_loader)
#         test_dls.append(test_loader)

#         dataset_lens.append(len(train_set))
    
#     # ... (客户端权重计算不变) ...
#     client_weight = []
#     total_len = sum(dataset_lens)
#     if total_len > 0: 
#         for i in dataset_lens:
#             client_weight.append(i / total_len)
#     else:
#         logging.warning("Total dataset length is zero. Using uniform weights.")
#         client_weight = [1.0 / len(clients)] * len(clients)

#     return train_dls, val_dls, test_dls, client_weight