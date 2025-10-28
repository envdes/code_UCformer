import torch
from torch import nn

class MLPModel(nn.Module):
    def __init__(self, input_dim=30, negative_slope=0.01):
        super(MLPModel, self).__init__()

        # LeakyReLU
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

        # hidden layers
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_dim, 192),
            self.activation,
            nn.Linear(192, 160),
            self.activation,
            nn.Linear(160, 128),
            self.activation,
            nn.Linear(128, 160),
            self.activation,
            nn.Linear(160, 160),
            self.activation,
            nn.Linear(160, 32),
            self.activation
        )

        # output layers
        self.out1 = nn.Linear(32, 1)
        self.out2 = nn.Linear(32, 1)
        self.out3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.hidden_layers(x)
        y1 = self.out1(x)
        y2 = self.out2(x)
        y3 = self.out3(x)
        out = torch.cat((y1, y2, y3), dim=1)
        return out