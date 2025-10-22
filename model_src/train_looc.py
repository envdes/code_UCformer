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
import matplotlib.pyplot as plt
import random
import time
from dataloader_looc import get_data
from ucformer import TransformerClimateModel


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


# model hyperparameters
q_dim = 17
kv_dim = 13
heads = 3
dim_head = 128
d_model = heads * dim_head
dropout = 0.1
ffn_hidden = 2048
num_layers = 4
output_dim = 3
lr = 0.00001
bs = 128
start = 189800 #29200*6.5
end = 219000 #29200*7.5
num_epochs = 50
set_seed(41)


train_loader, val_loader, tes_loader, scaled_train_lab_min, scaled_train_lab_max, train_lab_mean, train_lab_std = get_data('/home/jiyang/Project_ucformer/rebuttel/ucformer/data/', bs, start, end)

model = TransformerClimateModel(q_dim, kv_dim, output_dim, d_model, heads, dim_head, num_layers, ffn_hidden, dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 按label输出每个epoch中不同label的平均loss值
epoch_losses = []   #用于保存每个迭代产生loss的平均值，便于画图
TSA_U_loss = []
Q2M_loss = []
DEW_U_loss = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    TSA_loss = 0.0
    Q2M_loss = 0.0
    DEW_loss = 0.0
    running_losses_per_label = torch.zeros(3)
    for inputs, targets in train_loader:
        inputs = inputs.float()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        inputs_1 = inputs[:, :, :17]
        inputs_2 = inputs[:, :, 17:] 
        tsa, tw, q2m = model(inputs_1, inputs_2)
        loss1 = criterion(tsa, targets[:, :, 0:1])
        loss2 = criterion(q2m, targets[:, :, 1:2])
        loss3 = criterion(tw, targets[:, :, 2:3])
        loss = loss1 + loss2 + loss3
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        TSA_loss += loss1.item() * inputs.size(0)
        Q2M_loss += loss2.item() * inputs.size(0)
        DEW_loss += loss3.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_losses.append(epoch_loss)    # 累计每次epoch的loss平均值，为了画loss趋势图
    print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {epoch_loss:.4f}")
    scheduler.step(epoch_loss)
    label_loss1 = TSA_loss / len(train_loader.dataset)
    label_loss2 = Q2M_loss / len(train_loader.dataset)
    label_loss3 = DEW_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}]_TSA_U_loss: {label_loss1:.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}]_Q2M_loss: {label_loss2:.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}]_DEW_U_loss: {label_loss3:.4f}")

torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(), 'loss':loss}, '/home/jiyang/Project_ucformer/rebuttel/ucformer/checkpoint/20250728_UCformer_looc_rm.pth')
 
