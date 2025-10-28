import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
save_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'UCformer.pth')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
import torch
import torch.optim as optim
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from utils.tools import set_seed, inverse_scale, inverse_standardize
import time
from src.dataloader import get_data
from src.UCformer import TransformerClimateModel
import argparse
    
    
parser = argparse.ArgumentParser()
    
parser.add_argument('--datapath', 
                       type=str, 
                       required=True)
args = parser.parse_args()
data_path = args.datapath


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
bs = 64
# start = 189800 #29200*6.5
# end = 219000 #29200*7.5
num_epochs = 50
set_seed(41)


train_loader, val_loader, tes_loader, scaled_train_lab_min, scaled_train_lab_max, train_lab_mean, train_lab_std = get_data(data_path, bs)

model = TransformerClimateModel(q_dim, kv_dim, output_dim, d_model, heads, dim_head, num_layers, ffn_hidden, dropout)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
best_val_loss = float("inf")
epoch_losses_tr = []
epoch_losses_val = []

for epoch in range(num_epochs):
    # ====== Train ======
    model.train()
    running_loss = 0.0
    TSA_loss = 0.0
    Q2M_loss = 0.0
    DEW_loss = 0.0

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

        bs = inputs.size(0)
        running_loss += loss.item() * bs
        TSA_loss += loss1.item() * bs
        Q2M_loss += loss2.item() * bs
        DEW_loss += loss3.item() * bs

    epoch_loss_tr = running_loss / len(train_loader.dataset)
    epoch_losses_tr.append(epoch_loss_tr)

    label_loss1_tr = TSA_loss / len(train_loader.dataset)
    label_loss2_tr = Q2M_loss / len(train_loader.dataset)
    label_loss3_tr = DEW_loss / len(train_loader.dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}] Train Total: {epoch_loss_tr:.4f} | "
          f"T:{label_loss1_tr:.4f} q:{label_loss2_tr:.4f} t:{label_loss3_tr:.4f}")

    # ====== Validate ======
    model.eval()
    val_running_loss = 0.0
    val_TSA_loss = 0.0
    val_Q2M_loss = 0.0
    val_DEW_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.float()
            inputs, targets = inputs.to(device), targets.to(device)

            inputs_1 = inputs[:, :, :17]
            inputs_2 = inputs[:, :, 17:]
            tsa, tw, q2m = model(inputs_1, inputs_2)

            vloss1 = criterion(tsa, targets[:, :, 0:1])
            vloss2 = criterion(q2m, targets[:, :, 1:2])
            vloss3 = criterion(tw, targets[:, :, 2:3])
            vloss = vloss1 + vloss2 + vloss3

            bs = inputs.size(0)
            val_running_loss += vloss.item() * bs
            val_TSA_loss += vloss1.item() * bs
            val_Q2M_loss += vloss2.item() * bs
            val_DEW_loss += vloss3.item() * bs

    epoch_loss_val = val_running_loss / len(val_loader.dataset)
    epoch_losses_val.append(epoch_loss_val)

    label_loss1_val = val_TSA_loss / len(val_loader.dataset)
    label_loss2_val = val_Q2M_loss / len(val_loader.dataset)
    label_loss3_val = val_DEW_loss / len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}] Valid Total: {epoch_loss_val:.4f} | "
          f"T:{label_loss1_val:.4f} q:{label_loss2_val:.4f} t:{label_loss3_val:.4f}")

    # 若使用 ReduceLROnPlateau，建议用验证集 loss 调度
    if hasattr(scheduler, "step") and scheduler.__class__.__name__.lower().startswith("reducelronplateau"):
        scheduler.step(epoch_loss_val)
    else:
        # 如果你的 scheduler 不是 ReduceLROnPlateau，仍保留原先基于训练 loss 的 step
        scheduler.step()

    # ====== Save best on validation ======
    if epoch_loss_val < best_val_loss:
        best_val_loss = epoch_loss_val
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }, save_path)  # 例如: args.save_dir/best_model.pt
        print(f"✓ New best model saved at epoch {epoch+1} with val_loss {best_val_loss:.4f}")

# # 训练完成后保存最后一个 epoch（可选）
# torch.save({
#     'epoch': num_epochs,
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'best_val_loss': best_val_loss
# }, save_path)  # 例如: args.save_dir/last_model.pt

print("***** Training finished. Best checkpoints saved. *****")
# # 按label输出每个epoch中不同label的平均loss值
# epoch_losses = []   #用于保存每个迭代产生loss的平均值，便于画图
# TSA_U_loss = []
# Q2M_loss = []
# DEW_U_loss = []

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     TSA_loss = 0.0
#     Q2M_loss = 0.0
#     DEW_loss = 0.0
#     running_losses_per_label = torch.zeros(3)
#     for inputs, targets in train_loader:
#         inputs = inputs.float()
#         inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()
#         inputs_1 = inputs[:, :, :17]
#         inputs_2 = inputs[:, :, 17:] 
#         tsa, tw, q2m = model(inputs_1, inputs_2)
#         loss1 = criterion(tsa, targets[:, :, 0:1])
#         loss2 = criterion(q2m, targets[:, :, 1:2])
#         loss3 = criterion(tw, targets[:, :, 2:3])
#         loss = loss1 + loss2 + loss3
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
        
#         running_loss += loss.item() * inputs.size(0)
#         TSA_loss += loss1.item() * inputs.size(0)
#         Q2M_loss += loss2.item() * inputs.size(0)
#         DEW_loss += loss3.item() * inputs.size(0)
    
#     epoch_loss = running_loss / len(train_loader.dataset)
#     epoch_losses.append(epoch_loss)    # 累计每次epoch的loss平均值，为了画loss趋势图
#     print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {epoch_loss:.4f}")
#     scheduler.step(epoch_loss)
#     label_loss1 = TSA_loss / len(train_loader.dataset)
#     label_loss2 = Q2M_loss / len(train_loader.dataset)
#     label_loss3 = DEW_loss / len(train_loader.dataset)
#     print(f"Epoch [{epoch+1}/{num_epochs}]_T_loss: {label_loss1:.4f}")
#     print(f"Epoch [{epoch+1}/{num_epochs}]_q_loss: {label_loss2:.4f}")
#     print(f"Epoch [{epoch+1}/{num_epochs}]_t_loss: {label_loss3:.4f}")

# torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(), 'loss':loss}, save_path)
# print("*****Model saved successfully.******")
