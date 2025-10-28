import random
import torch
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 逆缩放和逆标准化函数
def inverse_scale(data, min_val, max_val):
    """逆缩放操作，使用PyTorch张量操作"""
    data = data.cpu()
    return (data + 1) * (max_val - min_val) / 2 + min_val

def inverse_standardize(data, mean, std):
    """逆Z-score标准化操作，使用PyTorch张量操作"""
    data = data.cpu()
    return data * std + mean

def calculate_r2(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def calculate_rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse

def cal_mess(y_true, y_pred, datm):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_range = np.mean(np.abs(y_true - datm))
    mess = np.mean(np.abs(y_true - y_pred) / y_range)
    return 1 - mess

def calculate_mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

def split_and_rename(data):
    groups = [2, 2, 3, 3, 3, 2]  # ld, nw, sh, sa, sp, rm
    group_names = ['London', 'NewYork', 'Shanghai', 'Singapore', 'SaoPaulo', 'Rome']
    # 确保数据能被平均分
    if len(data) % 15 != 0:
        raise ValueError("数据长度无法被 15 整除，请检查数据大小。")
    
    chunk_size = len(data) // 15  # 每一份的大小
    chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(15)]
    
    # 按照组合规则进行合并
    renamed_chunks = {}
    start_idx = 0
    for count, name in zip(groups, group_names):
        end_idx = start_idx + count
        renamed_chunks[name] = pd.concat(chunks[start_idx:end_idx], ignore_index=True)
        start_idx = end_idx
    
    return renamed_chunks

def split_and_rename_2070(data):
    groups = [2, 2, 2, 3, 3, 3]  # ld, nw, rm, sh, sa, sp
    group_names = ['London', 'NewYork', 'Rome', 'Shanghai', 'Singapore', 'SaoPaulo']
    # 确保数据能被平均分
    if len(data) % 15 != 0:
        raise ValueError("数据长度无法被 15 整除，请检查数据大小。")
    
    chunk_size = len(data) // 15  # 每一份的大小
    chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(15)]
    
    # 按照组合规则进行合并
    renamed_chunks = {}
    start_idx = 0
    for count, name in zip(groups, group_names):
        end_idx = start_idx + count
        renamed_chunks[name] = pd.concat(chunks[start_idx:end_idx], ignore_index=True)
        start_idx = end_idx
    
    return renamed_chunks