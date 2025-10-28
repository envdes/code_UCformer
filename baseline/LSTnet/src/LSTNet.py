import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.P = args.window  # window即seq_len
        self.m = 30
        self.hidC = args.hidCNN
        self.hidR = args.hidRNN
        self.Ck = args.CNN_kernel
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        # self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m), padding=(self.Ck//2, 0))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(args.dropout)
        self.linear1 = nn.Linear(self.hidR, 3)  # output_dim=3

    def forward(self, x):
        # batch_size = x.size(0)
        # c = x.view(-1, 1, self.P, self.m)
        # c = F.relu(self.conv1(c))
        # c = self.dropout(c)
        # c = torch.squeeze(c, 3)
        # r = c.permute(2, 0, 1).contiguous()
        # output, _ = self.GRU1(r)
        # output = self.dropout(output)
        # output = self.linear1(output)
        # output = output.permute(1, 0, 2)  # [batch, seq_len, output_dim]
        # return output
        batch_size = x.size(0)
        # x: [batch, timestep, m]
        c = x.view(-1, 1, self.P, self.m)
        # 计算需要的左、右padding
        pad_left = (self.Ck - 1) // 2
        pad_right = self.Ck - 1 - pad_left
        # pad格式: (last_dim_right, last_dim_left, ..., first_dim_right, first_dim_left)
        # 这里只在时序维度pad，所以(0,0,pad_left,pad_right)
        c = torch.nn.functional.pad(c, (0, 0, pad_left, pad_right))
        c = torch.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)
        r = c.permute(2, 0, 1).contiguous()
        output, _ = self.GRU1(r)
        output = self.dropout(output)
        output = self.linear1(output)
        output = output.permute(1, 0, 2)
        return output
