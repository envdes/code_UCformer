
import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerClimateModel(nn.Module):
    def __init__(self, input_dim, output_dim, model_dim, num_heads, num_encoder_layers, feedforward_dim, dropout_rate):
        super(TransformerClimateModel, self).__init__()
        self.model_dim = model_dim
        self.input_embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        self.dropout = nn.Dropout(dropout_rate)  # dropout
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout_rate, dim_feedforward=feedforward_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(model_dim, output_dim)
        
    def forward(self, src):
        src = src.permute(1, 0, 2)  # convert (batch_size, seq_length, feature_dim) to (seq_length, batch_size, feature_dim)
        src = self.input_embedding(src) * math.sqrt(self.model_dim)
        src = self.pos_encoder(src)
        src = self.dropout(src)  # dropout
        output = self.transformer_encoder(src)
        output = self.output_layer(output)  # (Seq_len, batch_size, output_dim)
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, output_dim)
        return output
