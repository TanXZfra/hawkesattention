import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding2,DataEmbedding_inverted

# PATCHING 
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride

        self.proj = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                              kernel_size=patch_len, stride=stride, padding=0)

    def forward(self, x, time):
        # x: [B, L, d_model]
        B, L, D = x.shape

        n_patches = math.ceil((L - self.patch_len + 1) / self.stride)
        pad_len = (n_patches - 1) * self.stride + self.patch_len - L
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            time = F.pad(time, (0, 0, 0, pad_len))
        # [B, d_model, L]
        x = x.permute(0, 2, 1)
        # [B, d_model, n_patches]
        x_patch = self.proj(x)
        # [B, n_patches, d_model]
        x_patch = x_patch.transpose(1, 2)
        # [B, L, time_dim] -> [B, n_patches, time_dim]
        patches = time.unfold(1, self.patch_len, self.stride)  # [B, n_patches, time_dim, patch_len]
        time_patch = patches.mean(-1)
        return x_patch, time_patch

# --------------------- TIME KERNEL ATTENTION -------------------------


class TimeKernelAttention(nn.Module):
    def __init__(self, d_model, n_kernel_heads, num_kernels, time_dim, dropout=0.1):
        super(TimeKernelAttention, self).__init__()
        self.d_model = d_model                                
        self.n_kernel_heads = n_kernel_heads                  
        self.num_kernels = num_kernels                        
        self.head_dim = d_model // n_kernel_heads             
        assert self.head_dim * n_kernel_heads == d_model

        self.W_O = nn.Linear(d_model, d_model)                
        self.dropout = nn.Dropout(dropout)
        self.time_dim = time_dim                         

        shape = (n_kernel_heads, self.head_dim, self.time_dim, self.num_kernels)

        # Q 
        self.mu_Q    = nn.Parameter(torch.randn(*shape))
        self.sigma_Q = nn.Parameter(torch.randn(*shape).abs().clamp_min(1e-3))
        self.w_Q     = nn.Parameter(torch.randn(*shape))
        self.gamma_Q = nn.Parameter(torch.zeros(n_kernel_heads, self.head_dim))
        self.W_Q     = nn.Parameter(torch.randn(n_kernel_heads, self.head_dim, self.head_dim))

        # K 
        self.mu_K    = nn.Parameter(torch.randn(*shape))
        self.sigma_K = nn.Parameter(torch.randn(*shape).abs().clamp_min(1e-3))
        self.w_K     = nn.Parameter(torch.randn(*shape))
        self.gamma_K = nn.Parameter(torch.zeros(n_kernel_heads, self.head_dim))
        self.W_K     = nn.Parameter(torch.randn(n_kernel_heads, self.head_dim, self.head_dim))

        # V
        self.mu_V    = nn.Parameter(torch.randn(*shape))
        self.sigma_V = nn.Parameter(torch.randn(*shape).abs().clamp_min(1e-3))
        self.w_V     = nn.Parameter(torch.randn(*shape))
        self.gamma_V = nn.Parameter(torch.zeros(n_kernel_heads, self.head_dim))
        self.W_V     = nn.Parameter(torch.randn(n_kernel_heads, self.head_dim, self.head_dim))

    def gaussian_kernel(self, t, mu, sigma, w):
        """
        t:  [B, L, time_dim]
        mu/sigma/w: [H, D, time_dim, K]
        out:   [B, H, D, L]
        """
        B, L, T = t.shape
        H, D, T_check, K = mu.shape
        assert T == T_check, "time_dim mismatch"

        # [B, 1, 1, L, T, 1]
        t = t.unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        mu    = mu.unsqueeze(0).unsqueeze(3)       # -> [1, H, D, 1, T, K]
        sigma = sigma.unsqueeze(0).unsqueeze(3)
        w     = w.unsqueeze(0).unsqueeze(3)

        # exp(- (t - mu)^2 / 2σ^2) , w -> [B, H, D, L, T, K]
        g = w * torch.exp(- (t - mu)**2 / (2 * sigma**2))

        # sum over time_dim (T) and kernel  (K) -> [B, H, D, L]
        return g.sum(-1).sum(-1)

    def forward(self, P, t):
        """
        - P: [B, L, d_model]   
        - t: [B, L, time_dim] 
        out: [B, L, d_model]
        """
        B, L, D = P.shape

        #  [B, H, L, head_dim]
        P = P.view(B, L, self.n_kernel_heads, self.head_dim).transpose(1, 2)

        Qw = self.gaussian_kernel(t, self.mu_Q, self.sigma_Q, self.w_Q)  # [B, H, D, L]
        Kw = self.gaussian_kernel(t, self.mu_K, self.sigma_K, self.w_K)
        Vw = self.gaussian_kernel(t, self.mu_V, self.sigma_V, self.w_V)

        Q = torch.einsum('bhdl,bhld->bhld', Qw, P) + self.gamma_Q.unsqueeze(0).unsqueeze(2)  # [B,H,L,D]
        K = torch.einsum('bhdl,bhld->bhld', Kw, P) + self.gamma_K.unsqueeze(0).unsqueeze(2)
        V = torch.einsum('bhdl,bhld->bhld', Vw, P) + self.gamma_V.unsqueeze(0).unsqueeze(2)

        Q = torch.einsum('bhld,hdf->bhlf', Q, self.W_Q)
        K = torch.einsum('bhld,hdf->bhlf', K, self.W_K)
        V = torch.einsum('bhld,hdf->bhlf', V, self.W_V)

        # scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,L,L]
        A = self.dropout(F.softmax(scores, dim=-1))  # [B,H,L,L]
        out = torch.matmul(A, V)                    # [B,H,L,head_dim]

        #  [B,L,D]
        out = out.transpose(1, 2).contiguous().reshape(B, L, D)
        return self.W_O(self.dropout(out))

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.patch_flag = configs.patch_flag
        self.class_strategy = configs.class_strategy
        self.d_model=configs.d_model
        self.embed=configs.embed
        self.freq=configs.freq
        self.dropout=configs.dropout

        self.enc_embedding = DataEmbedding2(
            configs.enc_in,          
            configs.d_model,         
            embed_type=configs.embed, 
            freq=configs.freq,     
            dropout=configs.dropout  
        )

        if self.patch_flag:
            self.patch_embed = PatchEmbedding(configs.d_model,
                                              configs.patch_len,
                                              configs.patch_stride)
        
            self.eff_seq_len = math.ceil((self.seq_len
                                     - configs.patch_len + 1)
                                     / configs.patch_stride)
        else:
            self.eff_seq_len = self.seq_len

        self.invert_embedding = DataEmbedding_inverted(
            self.eff_seq_len,
            self.d_model,
            self.embed,
            self.freq,
            self.dropout
        )

        # kernel attention
        self.new_layers = nn.ModuleList([
            TimeKernelAttention(configs.d_model,
                                configs.n_new_heads, configs.num_kernels,configs.time_dim, configs.dropout)
            for _ in range(configs.num_new_layers)
        ])
        # encoder
        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                  output_attention=configs.output_attention),
                    configs.d_model, configs.n_heads),
                configs.d_model, configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            ) for _ in range(configs.e_layers)
        ], norm_layer=torch.nn.LayerNorm(configs.d_model))
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape
        # embedding
        enc_out = self.enc_embedding(x_enc, None)
        # patch
        if self.patch_flag:
            enc_out, x_mark_enc = self.patch_embed(enc_out, x_mark_enc)
        # kernel attention
        for layer in self.new_layers:
            enc_out = layer(enc_out, x_mark_enc)

        # enc
        enc_out=self.invert_embedding(enc_out,x_mark_enc)
        # encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # projection & permute
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]
        # denormalization
        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns    
        else:
            return dec_out[:, -self.pred_len:, :]
