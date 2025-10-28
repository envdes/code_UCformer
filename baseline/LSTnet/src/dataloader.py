import numpy as np
import pandas as pd
import torch
from torch.utils import data
import torch.optim as optim
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import datasets, transforms
import math
import os
import tempfile
from torch.utils.data.sampler import SubsetRandomSampler

def get_data(parquetpath, bs):
    # parquetpath = '/jiyang/newdata/'
    tra = pd.read_parquet(parquetpath + 'train_2020to2044_features.parquet').drop(['EM_PERROAD', 'EM_IMPROAD', 'EM_WALL', 'ALB_PERROAD_DIR', 'ALB_IMPROAD_DIR','WTROAD_PERV'], axis=1)
    val = pd.read_parquet(parquetpath + 'val_2045to2049_features.parquet').drop(['EM_PERROAD', 'EM_IMPROAD', 'EM_WALL', 'ALB_PERROAD_DIR', 'ALB_IMPROAD_DIR','WTROAD_PERV'], axis=1)
    tes = pd.read_parquet(parquetpath + 'tes_2050to2054_features.parquet').drop(['EM_PERROAD', 'EM_IMPROAD', 'EM_WALL', 'ALB_PERROAD_DIR', 'ALB_IMPROAD_DIR','WTROAD_PERV'], axis=1)
    tra_lab = pd.read_parquet(parquetpath + 'train_2020to2044_labels.parquet').drop(['RH2M_U'], axis=1)
    val_lab = pd.read_parquet(parquetpath + 'val_2045to2049_labels.parquet').drop(['RH2M_U'], axis=1)
    tes_lab = pd.read_parquet(parquetpath + 'tes_2050to2054_labels.parquet').drop(['RH2M_U'], axis=1)
    ### 步骤 1: Z-score标准化训练和验证的特征数据
    train_fea = tra.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    # 计算训练集标准化后的最大值和最小值
    scaled_train_fea_max = train_fea.max()
    scaled_train_fea_min = train_fea.min()
    # 将训练集特征缩放到-1和1之间
    train_fea_scaled = 2 * (train_fea - scaled_train_fea_min) / (scaled_train_fea_max - scaled_train_fea_min) - 1
    ### 步骤 2: 使用训练集的均值和标准差对验证集和测试集进行Z-score标准化
    train_mean = np.mean(tra, axis=0)
    train_std = np.std(tra, axis=0)
    validation_fea = (val - train_mean) / train_std
    tes_fea = (tes - train_mean) / train_std
    # 这里确保我们使用相同的缩放公式和训练集的极值
    validation_fea_scaled = 2 * (validation_fea - scaled_train_fea_min) / (scaled_train_fea_max - scaled_train_fea_min) - 1
    tes_fea_scaled = 2 * (tes_fea - scaled_train_fea_min) / (scaled_train_fea_max - scaled_train_fea_min) - 1
    
    ### 步骤 3:Z-score标准化训练和验证的标签数据
    train_label = tra_lab.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    # 计算训练集标准化后的最大值和最小值
    scaled_train_lab_max = train_label.max()
    scaled_train_lab_min = train_label.min()
    train_label_scaled = 2 * (train_label - scaled_train_lab_min) / (scaled_train_lab_max - scaled_train_lab_min) - 1

    ### 步骤 4: 使用训练集的均值和标准差对验证集进行Z-score标准化
    train_lab_mean = np.mean(tra_lab, axis=0)
    train_lab_std = np.std(tra_lab, axis=0)
    validation_lab = (val_lab - train_lab_mean) / train_lab_std
    validation_lab_scaled = 2 * (validation_lab - scaled_train_lab_min) / (scaled_train_lab_max - scaled_train_lab_min) - 1
    tes_lab = (tes_lab - train_lab_mean) / train_lab_std
    tes_lab_scaled = 2 * (tes_lab - scaled_train_lab_min) / (scaled_train_lab_max - scaled_train_lab_min) - 1

    #将Dataframe转换为tensor
    train_fea_tensor = torch.tensor(train_fea_scaled.astype('float32').values)
    train_lab_tensor = torch.tensor(train_label_scaled.astype('float32').values)
    va_fea_tensor = torch.tensor(validation_fea_scaled.astype('float32').values)
    va_lab_tensor = torch.tensor(validation_lab_scaled.astype('float32').values)
    tes_fea_tensor = torch.tensor(tes_fea_scaled.astype('float32').values)
    tes_lab_tensor = torch.tensor(tes_lab_scaled.astype('float32').values)
    #将每日8个时间段的forcing数据展平
    train_fea_re = train_fea_tensor.reshape(-1,8,train_fea_tensor.shape[1])
    train_lab_re = train_lab_tensor.reshape(-1,8,train_lab_tensor.shape[1])
    va_fea_re = va_fea_tensor.reshape(-1,8,va_fea_tensor.shape[1])
    va_lab_re = va_lab_tensor.reshape(-1,8,va_lab_tensor.shape[1])
    tes_fea_re = tes_fea_tensor.reshape(-1,8,tes_fea_tensor.shape[1])
    tes_lab_re = tes_lab_tensor.reshape(-1,8,tes_lab_tensor.shape[1])
    # 创建 TensorDataset 和 DataLoader
    train_dataset = TensorDataset(train_fea_re, train_lab_re)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_dataset = TensorDataset(va_fea_re, va_lab_re)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
    tes_dataset = TensorDataset(tes_fea_re, tes_lab_re)
    tes_loader = DataLoader(tes_dataset, batch_size=bs, shuffle=False)
    return train_loader, val_loader, tes_loader, scaled_train_lab_min, scaled_train_lab_max, train_lab_mean, train_lab_std