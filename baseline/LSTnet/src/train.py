import torch
import torch.nn as nn
from LSTNet import Model
from dataloader import get_data
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)
# =========================
# 参数设置（根据实际需求调整）
    
class Args:
    window = 8    # timestep数，与你reshape时保持一致
    m = 30        # input dims
    hidCNN = 128   # 卷积隐藏通道
    hidRNN = 128  # GRU隐藏单元
    CNN_kernel = 3  # 卷积核大小
    dropout = 0.1
    seed = 54321
    cuda = True

args = Args()
args.batch_size = 128
args.epochs = 50
args.lr = 0.0001

# =========================
# 获取数据
train_loader, val_loader, test_loader, min_label, max_label, mean_label, std_label = get_data('/home/jiyang/Project_ucformer/rebuttel/ucformer/data/', args.batch_size)

device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

# =========================
# 构建模型
model = Model(args).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# =========================
# 训练函数
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)  # X: [B, T, 30]  Y: [B, T, 3]
        optimizer.zero_grad()
        output = model(X)                  # output: [B, T, 3]
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            loss = criterion(output, Y)
            total_loss += loss.item() * X.size(0)
    return total_loss / len(loader.dataset)

def inverse_scale(data, min_val, max_val):
    # data: [N, ...], min_val/max_val: [output_dim]
    # 必须转为 tensor，并broadcast
    device = data.device
    min_val = torch.as_tensor(min_val, dtype=data.dtype, device=device)
    max_val = torch.as_tensor(max_val, dtype=data.dtype, device=device)
    # 维度适配
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

def evaluate_and_save(model, loader, criterion, device,
                      min_label, max_label, mean_label, std_label,
                      save_csv_path='prediction_vs_truth.csv'):
    model.eval()
    total_loss = 0
    preds = []
    trues = []
    with torch.no_grad():
        for X, Y in tqdm(loader, desc='Evaluating'):
            X = X.to(device)
            Y = Y.to(device)
            output = model(X)
            # 还原真实值
            output_real = inverse_scale(output, min_label, max_label)
            output_real = inverse_standardize(output_real, mean_label, std_label)
            Y_real = inverse_scale(Y, min_label, max_label)
            Y_real = inverse_standardize(Y_real, mean_label, std_label)
            preds.append(output_real.cpu().numpy())
            trues.append(Y_real.cpu().numpy())
            loss = criterion(output_real, Y_real)
            total_loss += loss.item() * X.size(0)
    preds = np.concatenate(preds, axis=0)  # [N, T, D]
    trues = np.concatenate(trues, axis=0)

    # 总体RMSE
    rmse_total = np.sqrt(np.mean((preds - trues) ** 2))
    # 每个目标单独RMSE
    rmse_each = np.sqrt(np.mean((preds - trues) ** 2, axis=(0, 1)))  # 对batch和timestep求均值

    # 保存csv
    N, T, D = preds.shape
    flat_preds = preds.reshape(-1, D)
    flat_trues = trues.reshape(-1, D)
    df = pd.DataFrame(np.concatenate([flat_preds, flat_trues], axis=1),
                      columns=[f'pred_{i+1}' for i in range(D)] + [f'true_{i+1}' for i in range(D)])
    df.to_csv(save_csv_path, index=False)

    print(f'Overall RMSE: {rmse_total:.4f}')
    for i, rmse in enumerate(rmse_each):
        print(f'RMSE of target {i+1}: {rmse:.4f}')
    print(f"Predictions and true values saved to {save_csv_path}")
    return rmse_total, rmse_each

# =========================
# 训练流程
for epoch in range(1, args.epochs + 1):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# =========================
evaluate_and_save(
    model, test_loader, torch.nn.MSELoss(), device,
    min_label, max_label, mean_label, std_label,
    save_csv_path='/home/jiyang/Project_ucformer/rebuttel/LSTnet/output/pred_test_tuned_v3_dbcheck.csv'
)

torch.save(model.state_dict(), "/home/jiyang/Project_ucformer/rebuttel/LSTnet/checkpoint/lstnet_tuned_v3_dbcheck.pth")

