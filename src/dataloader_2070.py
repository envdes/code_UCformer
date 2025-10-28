import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

def get_2070data(parquetpath, bs):
    # 1. 读取标准化参数
    stats_path = os.path.join(parquetpath, 'train_statistics.json')
    with open(stats_path, 'r') as f:
        statistics = json.load(f)

    train_mean = np.array(statistics["train_mean"])
    train_std = np.array(statistics["train_std"])
    scaled_train_fea_max = np.array(statistics["scaled_train_fea_max"]) 
    scaled_train_fea_min = np.array(statistics["scaled_train_fea_min"])
    train_lab_mean = np.array(statistics["train_lab_mean"])
    train_lab_std = np.array(statistics["train_lab_std"])
    scaled_train_lab_max = np.array(statistics["scaled_train_lab_max"]) 
    scaled_train_lab_min = np.array(statistics["scaled_train_lab_min"])

    cities = ['London', 'Newyork', 'Rome', 'Shanghai', 'Singapore', 'SaoP']
    all_datasets = []  # 用于存放每个城市的 TensorDataset

    for city in cities:
        # 2. 读取特征和标签
        fea_path = os.path.join(parquetpath, f'{city}_2070to2074_features.parquet')
        lab_path = os.path.join(parquetpath, f'{city}_2070to2074_labels.parquet')
        tes = pd.read_parquet(fea_path).drop(
            ['EM_PERROAD', 'EM_IMPROAD', 'EM_WALL', 'ALB_PERROAD_DIR', 'ALB_IMPROAD_DIR', 'WTROAD_PERV'], axis=1
        )
        tes_lab = pd.read_parquet(lab_path).drop(['RH2M_U'], axis=1)

        # 3. 标准化
        tes_fea = (tes - train_mean) / train_std
        tes_fea_scaled = 2 * (tes_fea - scaled_train_fea_min) / (scaled_train_fea_max - scaled_train_fea_min) - 1
        tes_lab = (tes_lab - train_lab_mean) / train_lab_std
        tes_lab_scaled = 2 * (tes_lab - scaled_train_lab_min) / (scaled_train_lab_max - scaled_train_lab_min) - 1

        # 4. 转换为Tensor并reshape
        tensor_lab = torch.tensor(tes_lab_scaled.values.astype('float32')).reshape(-1, 8, 3)
        tensor_fea = torch.tensor(tes_fea_scaled.values.astype('float32')).reshape(-1, 8, 30)

        # 5. 组建TensorDataset
        dataset = TensorDataset(tensor_fea, tensor_lab)
        all_datasets.append(dataset)

    # 🔹 拼接所有城市的Dataset
    merged_dataset = ConcatDataset(all_datasets)

    # 🔹 创建一个总的 DataLoader（不打乱顺序）
    merged_loader = DataLoader(merged_dataset, batch_size=bs, shuffle=False)
    # dataloaders = {}

    # for city in cities:
    #     # 2. 读取特征和标签
    #     fea_path = os.path.join(parquetpath, f'{city}_2070to2074_features.parquet')
    #     lab_path = os.path.join(parquetpath, f'{city}_2070to2074_labels.parquet')
    #     tes = pd.read_parquet(fea_path).drop(['EM_PERROAD', 'EM_IMPROAD', 'EM_WALL', 'ALB_PERROAD_DIR', 'ALB_IMPROAD_DIR','WTROAD_PERV'], axis=1)
    #     tes_lab = pd.read_parquet(lab_path).drop(['RH2M_U'], axis=1)
        
    #     # 3. 标准化
    #     tes_fea = (tes - train_mean) / train_std
    #     tes_fea_scaled = 2 * (tes_fea - scaled_train_fea_min) / (scaled_train_fea_max - scaled_train_fea_min) - 1
    #     tes_lab = (tes_lab - train_lab_mean) / train_lab_std
    #     tes_lab_scaled = 2 * (tes_lab - scaled_train_lab_min) / (scaled_train_lab_max - scaled_train_lab_min) - 1

    #     # 4. 转换为Tensor并reshape
    #     tensor_lab = torch.tensor(tes_lab_scaled.values.astype('float32')).reshape(-1, 8, 3)
    #     tensor_fea = torch.tensor(tes_fea_scaled.values.astype('float32')).reshape(-1, 8, 30)

    #     # 5. 组建TensorDataset和DataLoader
    #     tensor_dataset = TensorDataset(tensor_fea, tensor_lab)
    #     dataloaders[f'{city}_2070_loader'] = DataLoader(tensor_dataset, batch_size=bs, shuffle=False)

    return merged_loader, scaled_train_fea_max, scaled_train_lab_min, scaled_train_lab_max, train_lab_mean, train_lab_std
