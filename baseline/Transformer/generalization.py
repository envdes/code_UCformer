import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
save_path = os.path.join(os.path.dirname(__file__), '..','..', 'checkpoints', 'Transformer.pth')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
import torch
import torch.optim as optim
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from utils.tools import set_seed, inverse_scale, inverse_standardize, split_and_rename_2070, calculate_r2, calculate_rmse, cal_mess
from dataloader_2070 import get_2070data
from Transformer import TransformerClimateModel
import argparse
import pandas as pd


output_dim = 3
input_dim = 30
num_epochs = 100
bs = 32
model_dim = 128
num_heads = 4
num_encoder_layers = 8
lr = 0.00001
feedforward_dim = 2048
dropout_rate = 0.1352
set_seed(41)

parser = argparse.ArgumentParser()
    
parser.add_argument('--datapath', 
                       type=str, 
                       required=True)
args = parser.parse_args()
data_path = args.datapath
tes_loader, scaled_train_fea_max, scaled_train_lab_min, scaled_train_lab_max, train_lab_mean, train_lab_std = get_2070data(data_path, bs)



model = TransformerClimateModel(input_dim, output_dim, model_dim, num_heads, num_encoder_layers, feedforward_dim, dropout_rate)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

checkpoint = torch.load(save_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

criterion = nn.MSELoss()

print("Model loaded successfully")
y_pred = [[] for _ in range(3)]
y_true = [[] for _ in range(3)]
model.eval()

with torch.no_grad():
    for inputs, targets in tes_loader:
        inputs = inputs.float()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        # 逆缩放和逆标准化预测和真实标签
        scaled_train_lab_min = torch.tensor(scaled_train_lab_min, device=device)
        scaled_train_lab_max = torch.tensor(scaled_train_lab_max, device=device)
        train_lab_mean = torch.tensor(train_lab_mean, device=device)
        train_lab_std = torch.tensor(train_lab_std, device=device)
                                        
        outputs = inverse_scale(outputs, scaled_train_lab_min, scaled_train_lab_max)
        outputs = inverse_standardize(outputs, train_lab_mean, train_lab_std)
        targets = inverse_scale(targets , scaled_train_lab_min, scaled_train_lab_max)
        targets = inverse_standardize(targets , train_lab_mean, train_lab_std)

        # 计算损失
        for i in range(3):
            y_pred[i].extend(outputs[:, :, i].cpu().numpy().flatten().tolist())
            y_true[i].extend(targets[:, :, i].cpu().numpy().flatten().tolist())

y_pred_df = pd.DataFrame(y_pred).T
y_true_df = pd.DataFrame(y_true).T
y_pred_df_splits = split_and_rename_2070(y_pred_df)
y_true_df_splits = split_and_rename_2070(y_true_df)
datm_data_path = os.path.join(os.path.dirname(__file__), '..', '..','data', 'datm_2070.csv')
datm_data = pd.read_csv(datm_data_path)
datm_data_splits = split_and_rename_2070(datm_data)

all_T_rmse = []
all_q_rmse = []
all_t_rmse = []
all_T_mess = []
all_q_mess = []
all_t_mess = []
all_T_r2 = []
all_q_r2 = []
all_t_r2 = []
for i, city in enumerate(['London', 'NewYork', 'Rome', 'Shanghai', 'Singapore', 'SaoPaulo']):
    y_pred_city = y_pred_df_splits[city]
    y_true_city = y_true_df_splits[city]
    datm_city = datm_data_splits[city]
    T_rmse = calculate_rmse(y_true_city.iloc[:, 0], y_pred_city.iloc[:, 0])
    q_rmse = calculate_rmse(y_true_city.iloc[:, 1], y_pred_city.iloc[:, 1])
    q_rmse = q_rmse * 1000  # 转换为g/kg
    t_rmse = calculate_rmse(y_true_city.iloc[:, 2], y_pred_city.iloc[:, 2])
    all_T_rmse.append(T_rmse)
    all_q_rmse.append(q_rmse)
    all_t_rmse.append(t_rmse)
    print(f"{city} T_RMSE: {T_rmse:.4f}")
    print(f"{city} q_RMSE: {q_rmse:.4f}")
    print(f"{city} t_RMSE: {t_rmse:.4f}")
    T_r2 = calculate_r2(y_true_city.iloc[:, 0], y_pred_city.iloc[:, 0])
    q_r2 = calculate_r2(y_true_city.iloc[:, 1], y_pred_city.iloc[:, 1])
    t_r2 = calculate_r2(y_true_city.iloc[:, 2], y_pred_city.iloc[:, 2])
    all_T_r2.append(T_r2)
    all_q_r2.append(q_r2)
    all_t_r2.append(t_r2)
    print(f"{city} T_R2: {T_r2:.4f}")
    print(f"{city} q_R2: {q_r2:.4f}")
    print(f"{city} t_R2: {t_r2:.4f}")
    T_mess = cal_mess(y_true_city.iloc[:, 0], y_pred_city.iloc[:, 0], datm_city['a2x3h_Sa_tbot'])
    q_mess = cal_mess(y_true_city.iloc[:, 1], y_pred_city.iloc[:, 1], datm_city['a2x3h_Sa_shum'])
    t_mess = cal_mess(y_true_city.iloc[:, 2], y_pred_city.iloc[:, 2], datm_city['Dew_Point'])
    all_T_mess.append(T_mess)
    all_q_mess.append(q_mess)
    all_t_mess.append(t_mess)
    print(f"{city} T_MESS: {T_mess:.4f}")
    print(f"{city} q_MESS: {q_mess:.4f}")
    print(f"{city} t_MESS: {t_mess:.4f}")
    
mean_T_rmse = sum(all_T_rmse) / len(all_T_rmse)
mean_q_rmse = sum(all_q_rmse) / len(all_q_rmse)
mean_t_rmse = sum(all_t_rmse) / len(all_t_rmse)
mean_T_r2 = sum(all_T_r2) / len(all_T_r2)
mean_q_r2 = sum(all_q_r2) / len(all_q_r2)
mean_t_r2 = sum(all_t_r2) / len(all_t_r2)
all_T_mess = sum(all_T_mess) / len(all_T_mess)
all_q_mess = sum(all_q_mess) / len(all_q_mess)
all_t_mess = sum(all_t_mess) / len(all_t_mess)
all_mess = all_T_mess + all_q_mess + all_t_mess
print(f"Mean_T_RMSE: {mean_T_rmse:.4f}")
print(f"Mean_q_RMSE: {mean_q_rmse:.4f}")
print(f"Mean_t_RMSE: {mean_t_rmse:.4f}")
print(f"Mean_T_R2: {mean_T_r2:.4f}")
print(f"Mean_q_R2: {mean_q_r2:.4f}")
print(f"Mean_t_R2: {mean_t_r2:.4f}")
print(f"All_MESS: {all_mess:.4f}")