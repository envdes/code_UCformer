import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import math
from torch.utils.data.sampler import SubsetRandomSampler



class FeatureNet(nn.Module):
    def __init__(self, input_dim_f, output_dim_f):
        super(FeatureNet, self).__init__()
        self.fc1 = nn.Linear(input_dim_f, output_dim_f)
        self.act_layer = nn.GELU()

    def forward(self, x):
        x = self.act_layer(self.fc1(x))
        return x

        
class ResidualAdd(nn.Module):
    def __init__(self, fn,res_scal=1):
        super().__init__()
        self.fn = fn   
        self.res_scal=res_scal
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res*self.res_scal
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000000):
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

class climate_attention(nn.Module):
    def __init__(self, heads, dim_head, d_model, dropout, res_scal=1):
        super(climate_attention, self).__init__()
        d_model = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        # self.to_out= nn.Sequential(nn.Linear(inner_dim, kv_dim), nn.Dropout(dropout))
        self.w_combine = nn.Linear(d_model, d_model)
        self.res_scal = res_scal
    def forward(self, forcing_data, surface_data):
        batch, time, dimension = forcing_data.shape
        _k = surface_data
        q = self.to_q(forcing_data).view(batch, time, self.heads, self.dim_head).permute(0, 2, 1, 3)
        k = self.to_k(surface_data).view(batch, time, self.heads, self.dim_head).permute(0, 2, 1, 3)
        v = self.to_v(surface_data).view(batch, time, self.heads, self.dim_head).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.dim_head) #* self.scale #debug
        attn = self.attend(scores)
        # attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, time, dimension)
        output = self.w_combine(out)
        # x = self.dropout(output)
        # output += _k * self.res_scal
        return output
        
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, ffn_hidden, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, ffn_hidden)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(ffn_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class DecoderFeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super(DecoderFeedForward, self).__init__()
        self.d_model = d_model
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(d_model // 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, heads, dim_head, d_model, dropout, ffn_hidden):
        super(EncoderLayer, self).__init__()
        self.attn = climate_attention(heads, dim_head, d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, y):
        _x = x
        output = self.attn(x, y)

        # output = self.drop1(output)
        x = self.norm1(output + _x)

        _x = x
        x = self.ffn(x)

        # x = self.drop2(x)
        x = self.norm2(x + _x)
        return x
        
class TransformerEncoder(nn.Module):
    def __init__(self, heads, dim_head, d_model, dropout, ffn_hidden, num_layers, output_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(heads, dim_head, d_model, dropout, ffn_hidden) for _ in range(num_layers)
        ])
    def forward(self, x, y):
        for i, layer in enumerate(self.layers):
            if i == 0:
                out = layer(x, y)
            # if i == -1:
            #     out = layer(x, out)
            else:
                out = layer(out, out)
        return out
    
class Decoder(nn.Module):
    def __init__(self, d_model):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.scale = d_model // 3
    def forward(self, x):
        temp_tsa, temp_tw, temp_q2m = torch.split(x, self.scale, dim=2)
        return temp_tsa, temp_tw, temp_q2m

class Decoder_out(nn.Module):
    def __init__(self, dim_head, d_model, dropout):
        super(Decoder_out, self).__init__()
        self.d_model = d_model
        self.scale = d_model // 3
        self.q_out = nn.Linear(self.scale, d_model)
        self.kv_out = nn.Linear(self.scale * 2, d_model)
        self.attend = nn.Softmax(dim = -1)
        self.out = nn.Linear(d_model, 1)
        self.drop = nn.Dropout(dropout)
    def forward(self, q, k, v):
        q = self.q_out(q)
        kv = torch.cat((k,v), dim = 2)
        kv = self.kv_out(kv)
        scores = torch.matmul(q, kv.transpose(1, 2)) / math.sqrt(self.d_model)
        attn = self.attend(scores)
        out = torch.matmul(attn, kv)
        out = self.drop(out)
        out = self.out(out)
        return q, out        
    
class TransformerClimateModel(nn.Module):
    def __init__(self, q_dim, kv_dim, output_dim, d_model, heads, dim_head, num_layers, ffn_hidden, dropout):
        super(TransformerClimateModel, self).__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.d_model = d_model
        self.q_embedding = ResidualAdd(FeatureNet(q_dim, q_dim))
        self.kv_embedding = ResidualAdd(FeatureNet(kv_dim, kv_dim))
        self.q_input = nn.Linear(q_dim, d_model)
        self.kv_input = nn.Linear(kv_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.drop1 = nn.Dropout(dropout)  # dropout
        self.drop2 = nn.Dropout(dropout)
        encoder_layers = EncoderLayer(heads, dim_head, d_model, dropout, ffn_hidden)
        self.transformer_encoder = TransformerEncoder(heads, dim_head, d_model, dropout, ffn_hidden, num_layers, output_dim)
        
        self.de_norm1 = nn.LayerNorm(d_model)
        self.de_norm2 = nn.LayerNorm(d_model)
        self.de_norm3 = nn.LayerNorm(d_model)
        self.de_ffn1 = DecoderFeedForward(d_model, dropout)
        self.de_ffn2 = DecoderFeedForward(d_model, dropout)
        self.de_ffn3 = DecoderFeedForward(d_model, dropout)
        self.decoder = Decoder(d_model)
        self.decoder_out1 = Decoder_out(dim_head, d_model, dropout)
        self.decoder_out2 = Decoder_out(dim_head, d_model, dropout)
        self.decoder_out3 = Decoder_out(dim_head, d_model, dropout)
        
    def forward(self, x, y):
        x = self.q_embedding(x)
        y = self.kv_embedding(y)
        x = self.q_input(x) * math.sqrt(self.d_model)
        x = self.drop1(x)
        y = self.kv_input(y) * math.sqrt(self.d_model)
        y = self.drop2(y)
        x = self.pos_encoder(x)
        y = self.pos_encoder(y)
        encoder_output = self.transformer_encoder(x, y)
        temp_tsa, temp_tw, temp_q2m = self.decoder(encoder_output)
        res_tsa, tsa = self.decoder_out1(temp_tsa, temp_tw, temp_q2m)
        tsa = self.de_norm1(res_tsa + tsa)
        tsa = self.de_ffn1(tsa)
        res_tw, tw = self.decoder_out2(temp_tw, temp_tsa, temp_q2m)
        tw = self.de_norm2(res_tw + tw)
        tw = self.de_ffn2(tw)
        res_q2m, q2m = self.decoder_out3(temp_q2m, temp_tw, temp_tsa)
        q2m = self.de_norm3(res_q2m + q2m)
        q2m = self.de_ffn3(q2m)
        return tsa, tw, q2m