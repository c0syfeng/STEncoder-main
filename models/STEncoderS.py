import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding,Adaptive_Wavelet_Tranform
from layers.dilated_conv import DilatedConvEncoder
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.channels = configs.enc_in
        self.c_out = configs.c_out
        self.hidden_dims = configs.d_model
        self.repr_dims = configs.d_ff
        self.depth = configs.e_layers

        self.STEncoder = configs.STEncoder

        self.STEncoder_wnorm = configs.STEncoder_wnorm
        self.STEncoder_multiscales = configs.STEncoder_multiscales

        self.enc_embedding = DataEmbedding(configs.enc_in, self.hidden_dims, configs.embed, configs.freq, dropout=configs.dropout)
        self.feature_extractor = DilatedConvEncoder(
            self.hidden_dims,
            [self.hidden_dims] * self.depth + [self.repr_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        self.repr_head = nn.Linear(self.repr_dims, self.repr_dims)

        self.ch_mlps = nn.ModuleList([nn.Linear(self.repr_dims, self.c_out) for _ in self.STEncoder_multiscales])
        self.length_mlp = nn.Linear(self.seq_len, self.pred_len)
        self.trend_decoms = nn.ModuleList([series_decomp(kernel_size=dlen + 1) for dlen in self.STEncoder_multiscales])

        self.input_decom = series_decomp(25)
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        self.AWT = Adaptive_Wavelet_Tranform()

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.dd_model, configs.n_heads),
                    configs.dd_model,
                    configs.dd_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.ee_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.dd_model)
        )
        self.projector = nn.Linear(configs.dd_model, configs.pred_len, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, onlyrepr=False,use_norm=True,attn=False):
        # x: [Batch, Input length, Channel]

        if self.STEncoder_wnorm == 'ReVIN':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            seq_std = x_enc.std(dim=1, keepdim=True).detach()
            short_x = (x_enc - seq_mean) / (seq_std + 1e-5)
            long_x = short_x.clone()
        elif self.STEncoder_wnorm == 'Mean':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            short_x = (x_enc - seq_mean)
            long_x = short_x.clone()
        elif self.STEncoder_wnorm == 'Decomp':
            short_x, long_x = self.input_decom(x_enc)
        elif self.STEncoder_wnorm == 'LastVal':
            seq_last = x_enc[:,-1:,:].detach()
            short_x = (x_enc - seq_last)
            long_x = short_x.clone()
        else:
            raise Exception(f'Not Supported Window Normalization:{self.STEncoder_wnorm}. Use {"{ReVIN | Mean | LastVal | Decomp}"}.')
        
        if attn:
            if use_norm:
              # Normalization from Non-stationary Transformer
              means = short_x_enc.mean(1, keepdim=True).detach()
              short_x_enc = short_x_enc - means
              stdev = torch.sqrt(torch.var(short_x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
              short_x_enc /= stdev

            _, _, N = short_x_enc.shape # B L N

            short_x_enc = self.AWT(short_x_enc)
            short_enc_out = self.enc_embedding_Season(short_x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens

            # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
            short_enc_out, attns = self.encoder(short_enc_out, attn_mask=None)

            # B N E -> B N S -> B S N 
            short_dec_out = self.projector(short_enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

            if use_norm:
                # De-Normalization from Non-stationary Transformer
                season_out = short_dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
                season_out = short_dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
    
            season_out = season_out[:, -self.pred_len:, :]

        enc_out = self.enc_embedding(long_x, x_mark_enc)
        enc_out = enc_out.transpose(1, 2)  # B x Ch x T
        repr = self.repr_dropout(self.feature_extractor(enc_out)).transpose(1, 2) 

        if onlyrepr:
            return None, repr

        len_out = F.gelu(repr.transpose(1, 2)) # B x O x T
        len_out = self.length_mlp(len_out).transpose(1, 2) # (B, I, C) > (B, O, C)

        trend_outs = []
        for trend_decom, ch_mlp in zip(self.trend_decoms, self.ch_mlps):
            _, dec_out = trend_decom(len_out)
            dec_out = F.gelu(dec_out)
            dec_out = ch_mlp(dec_out)  # (B, I, D) > (B, I, C)
            _, trend_out = trend_decom(dec_out)
            trend_outs.append(trend_out)

        trend_outs = torch.stack(trend_outs, dim=-1).sum(dim=-1)
        short_x = self.AWT(short_x)

        season_out = self.Linear(short_x.permute(0, 2, 1)).permute(0, 2, 1)

        if self.STEncoder_wnorm == 'ReVIN':
            pred = (season_out + trend_outs)*(seq_std+1e-5) + seq_mean
        elif self.STEncoder_wnorm == 'Mean':
            pred = season_out + trend_outs + seq_mean
        elif self.STEncoder_wnorm == 'Decomp':
            pred = season_out + trend_outs
        elif self.STEncoder_wnorm == 'LastVal':
            pred = season_out + trend_outs + seq_last
        else:
            raise Exception()

        if self.STEncoder:
            return pred, repr  # [Batch, Output length, Channel]
        else:
            return pred


    def get_embeddings(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, onlyrepr=False):

        if self.STEncoder_wnorm == 'ReVIN':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            seq_std = x_enc.std(dim=1, keepdim=True).detach()
            short_x = (x_enc - seq_mean) / (seq_std + 1e-5)
            long_x = short_x.clone()
        elif self.STEncoder_wnorm == 'Mean':
            seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
            short_x = (x_enc - seq_mean)
            long_x = short_x.clone()
        elif self.STEncoder_wnorm == 'Decomp':
            short_x, long_x = self.input_decom(x_enc)
        elif self.STEncoder_wnorm == 'LastVal':
            seq_last = x_enc[:,-1:,:].detach()
            short_x = (x_enc - seq_last)
            long_x = short_x.clone()
        else:
            raise Exception()

        enc_out = self.enc_embedding(long_x, x_mark_enc)
        enc_out = enc_out.transpose(1, 2)  # B x Ch x T
        repr = self.repr_dropout(self.feature_extractor(enc_out)).transpose(1, 2) # B x Co x T

        return repr


