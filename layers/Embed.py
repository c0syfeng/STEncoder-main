import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pywt
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_

def get_w(x):
    B,L,D=x.size()
    x = x.reshape(L,B*D)
    x_np = x.detach().cpu().numpy()
    low_freq, high_freq = pywt.dwt(x_np, 'db1') 
    w_l=torch.from_numpy(low_freq).to(x.device)
    w_h=torch.from_numpy(high_freq).to(x.device)

    w_l=torch.mean(w_l,dim=1,keepdim=True).repeat(1,L)
    w_h=torch.mean(w_h,dim=1,keepdim=True).repeat(1,L)

    return w_l,w_h

    

def wavelet_init(self, layer, weight):
    if isinstance(layer, torch.nn.Linear):
        with torch.no_grad():
            layer.weight.data = weight


class STDec(nn.Module):
    def __init__(self,seq_len,enc_in):
        super(STDec,self).__init__()
        self.projectorl = nn.Linear(seq_len,seq_len,bias=False)
        self.projectorh = nn.Linear(seq_len,seq_len,bias=False) 
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.Convb = nn.Conv1d(in_channels=3*enc_in, out_channels=enc_in,
                               kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        nn.init.kaiming_normal_(self.Convb.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self,x):
        x = x.transpose(1,2)
        x_l = self.projectorl(x)
        x_h = self.projectorh(x)
        x = torch.concat((x,x_l,x_h),dim=1)
        x = self.Convb(x).transpose(1,2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(##########################################################################
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    
class PartitionTokenEmbedding(nn.Module):
    def __init__(self,c_in,d_model,seq_len):
        super(PartitionTokenEmbedding, self).__init__()
        self.PtokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model//2,kernel_size=4,stride=4,bias=False)
        self.DPtokenConv1 = nn.Conv1d(in_channels=c_in, out_channels=d_model//2,kernel_size=6,stride=2,padding=2,bias=False)
        self.DPtokenConv2 = nn.Conv1d(in_channels=d_model//2, out_channels=d_model//2,kernel_size=2,stride=2,bias=False)
        self.projector = nn.Linear(seq_len//4,seq_len,bias=False)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.transpose(1,2)
        x1 = self.PtokenConv(x)
        x2 = self.DPtokenConv1(x)
        x2 = self.DPtokenConv2(x2)
        x = torch.concat((x1,x2),dim=1)
        x = self.projector(F.gelu(x)).transpose(1,2)

        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h', wo_freq=' '):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13  # from 1 to 12, so size is 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        if 'h' not in wo_freq:
            self.hour_embed = Embed(hour_size, d_model)
        if 'd' not in wo_freq:
            self.day_embed = Embed(day_size, d_model)
        if 'w' not in wo_freq:
            self.weekday_embed = Embed(weekday_size, d_model)
        if 'm' not in wo_freq:
            self.month_embed = Embed(month_size, d_model)

        print(f'TemporalEmbedding({embed_type})-wo_freq:', wo_freq)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3]) if hasattr(self, 'hour_embed') else 0.
        weekday_x = self.weekday_embed(x[:, :, 2]) if hasattr(self, 'weekday_embed') else 0.
        day_x = self.day_embed(x[:, :, 1]) if hasattr(self, 'day_embed') else 0.
        month_x = self.month_embed(x[:, :, 0]) if hasattr(self, 'month_embed') else 0.

        return hour_x + weekday_x + day_x + month_x + minute_x


# class TimeFeatureEmbedding(nn.Module):
#     def __init__(self, d_model, embed_type='timeF', freq='h'):
#         super(TimeFeatureEmbedding, self).__init__()
#
#         freq_map = {'h': 4, 't': 5, 's': 6,
#                     'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
#         d_inp = freq_map[freq]
#         self.embed = nn.Linear(d_inp, d_model, bias=False)
#
#     def forward(self, x):
#         return self.embed(x)


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h', wo_freq=' '):
        super(TimeFeatureEmbedding, self).__init__()

        # t > h > w > d > m
        # hour dim=0, week_dim=1, day_dim=2, year_dim=3
        freq_dim_map = {'h':0, 'w':1, 'd':2, 'm':3}
        self.rm_idx = []
        if wo_freq is not None:
            for f in wo_freq:
                if f in 'hwdm':
                    self.rm_idx.append(freq_dim_map[f])

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        self.d_inp = freq_map[freq]
        self.embed = nn.Linear(self.d_inp, d_model, bias=False)

        print('TimeFeatureEmbedding-wo-freq:', wo_freq, self.rm_idx)

    def forward(self, x):
        if len(self.rm_idx) > 0:
            x[:, :,  self.rm_idx] = 0.0
        return self.embed(x)


class DataEmbedding_t(nn.Module):
    def __init__(self,c_in,d_model,seq_len, embed_type='fixed', freq='h', wo_freq=' ', dropout=0.1,):
        super(DataEmbedding_t, self).__init__()

        self.value_embedding = PartitionTokenEmbedding(c_in=c_in, d_model=d_model,seq_len=seq_len)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq, wo_freq=wo_freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq, wo_freq=wo_freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            # print(self.value_embedding(x).shape,  self.temporal_embedding(x_mark).shape)
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
            
        return self.dropout(x)

class DataEmbedding(nn.Module):
    def __init__(self,c_in,d_model, embed_type='fixed', freq='h', wo_freq=' ', dropout=0.1,):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq, wo_freq=wo_freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq, wo_freq=wo_freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            # print(self.value_embedding(x).shape,  self.temporal_embedding(x_mark).shape)
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
            
        return self.dropout(x)



class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq, wo_freq=' ') if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq, wo_freq=' ')
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        # x: [Batch Variate d_model]
        return self.dropout(x)
    
    
class Adaptive_Wavelet_Tranform(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)
    
    def create_adaptive_high_freq_mask(self,high):
        B, C, W = high.shape
        high_s = high** 2

        threshold = torch.quantile(high_s, self.threshold_param,dim=-1,keepdim=True)
        threshold = threshold.view(B,C,-1)
        threshold = threshold.repeat(1,1,W)
        bools = high_s > threshold

        # Initialize adaptive mask
        adaptive_mask = torch.zeros_like(high_s, device=high.device)
        adaptive_mask[bools] = 1

        return adaptive_mask
    
    def forward(self, x_in):
        device_ = x_in.device

        x = x_in.to(torch.float32)
        x = x.transpose(1,2)
        x = x.cpu().numpy()
        low,high = pywt.wavedec(x,'db8', level=1)
        high = torch.from_numpy(high).to(device_)
        adaptive_mask = self.create_adaptive_high_freq_mask(high)
        high_masked = high*adaptive_mask
        
        high_masked = high_masked.cpu().numpy()
        
        x=pywt.waverec([low,high_masked],'db8')
        
        x = torch.from_numpy(x).to(device_)
        
        return x.transpose(1,2)