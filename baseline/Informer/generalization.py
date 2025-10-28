
import torch
import numpy as np
import pandas as pd
from models.model import Informer  # 或你的Informer类路径
from data.dataloader_2070 import get_2070data    # 保证FourTupleDataset已经是4元组格式
import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
save_path = os.path.join(project_root, 'checkpoints', 'Informer.pth')
from utils.ucformertool import set_seed, split_and_rename_2070, calculate_r2, calculate_rmse, cal_mess
import argparse
import pandas as pd

# ==== 1. 逆归一化函数 ====
# =========== 1. 逆变换函数 ===========
def inverse_scale(data, min_val, max_val):
    device = data.device
    min_val = torch.as_tensor(min_val, dtype=data.dtype, device=device)
    max_val = torch.as_tensor(max_val, dtype=data.dtype, device=device)
    while min_val.ndim < data.ndim:
        min_val = min_val.unsqueeze(0)
        max_val = max_val.unsqueeze(0)
    return (data + 1) * (max_val - min_val) / 2 + min_val

def inverse_standardize(data, mean, std):
    device = data.device
    mean = torch.as_tensor(mean, dtype=data.dtype, device=device)
    std = torch.as_tensor(std, dtype=data.dtype, device=device)
    while mean.ndim < data.ndim:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return data * std + mean

# ==== 2. 参数和设备 ====
class Args:
    seq_len = 8
    label_len = 8
    pred_len = 0   # mapping任务
    enc_in = 30
    dec_in = 3
    c_out = 3
    batch_size = 128
    d_model = 512
    n_heads = 8
    e_layers = 2
    d_layers = 1
    d_ff = 2048
    factor = 5
    dropout = 0.05
    attn = 'prob'
    embed = 'timeF'
    freq = 'h'
    activation = 'gelu'
    output_attention = False
    distil = True
    mix = True
    use_gpu = True
    
parser = argparse.ArgumentParser()
    
parser.add_argument('--datapath', 
                       type=str, 
                       required=True)
args = parser.parse_args()
data_path = args.datapath
args = Args()
device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

# ==== 3. 加载数据（和训练时一致，加载test loader）====

test_loader, scaled_train_fea_max, lab_min, lab_max, lab_mean, lab_std = get_2070data(data_path, args.batch_size)  # 或只返回test_loader，视你的get_data接口而定

# ==== 4. 加载训练好的模型 ====
model = Informer(
    args.enc_in, args.dec_in, args.c_out, args.seq_len, args.label_len, args.pred_len,
    args.factor, args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
    args.dropout, args.attn, args.embed, args.freq, args.activation,
    args.output_attention, args.distil, args.mix, device
).to(device)
state = torch.load(save_path, map_location=device)
model.load_state_dict(state)
model.eval()

# ==== 5. 推理与逆归一化、RMSE计算 ====
y_pred = [[] for _ in range(3)]
y_true = [[] for _ in range(3)]
with torch.no_grad():
    for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)
        # mapping任务pred_len=0，所以dec_inp、y_mark shape与y一致
        outputs = model(batch_x, batch_x_mark, batch_y, batch_y_mark)  # [B, T, 3]
        preds_real = inverse_scale(outputs, lab_min, lab_max)
        preds_real = inverse_standardize(preds_real, lab_mean, lab_std)
        trues_real = inverse_scale(batch_y, lab_min, lab_max)
        trues_real = inverse_standardize(trues_real, lab_mean, lab_std)
        for i in range(3):
            y_pred[i].extend(preds_real[:, :, i].cpu().numpy().flatten().tolist())
            y_true[i].extend(trues_real[:, :, i].cpu().numpy().flatten().tolist())
        # preds.append(outputs.detach().cpu().numpy())
        # trues.append(batch_y.detach().cpu().numpy())
        
y_pred_df = pd.DataFrame(y_pred).T
y_true_df = pd.DataFrame(y_true).T    
        
y_pred_df_splits = split_and_rename_2070(y_pred_df)
y_true_df_splits = split_and_rename_2070(y_true_df)
datm_data_path = os.path.join(project_root, 'data', 'datm_2070.csv')
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