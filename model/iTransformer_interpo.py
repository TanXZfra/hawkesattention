import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding2,DataEmbedding_inverted

# --------------------- PATCHING MODULE -------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        # Conv1d 对 embedding 后的特征进行 patch
        self.proj = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                              kernel_size=patch_len, stride=stride, padding=0)

    def forward(self, x, time):
        # x: [B, L, d_model]
        B, L, D = x.shape
        # 计算需要补零的长度，保证最后一个 patch 满足大小
        n_patches = math.ceil((L - self.patch_len + 1) / self.stride)
        pad_len = (n_patches - 1) * self.stride + self.patch_len - L
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            time = F.pad(time, (0, 0, 0, pad_len))
        # 转换到 [B, d_model, L]
        x = x.permute(0, 2, 1)
        # 卷积得到 [B, d_model, n_patches]
        x_patch = self.proj(x)
        # 还原到 [B, n_patches, d_model]
        x_patch = x_patch.transpose(1, 2)
        # 时间戳 patch 平均: [B, L, time_dim] -> [B, n_patches, time_dim]
        patches = time.unfold(1, self.patch_len, self.stride)  # [B, n_patches, time_dim, patch_len]
        time_patch = patches.mean(-1)
        return x_patch, time_patch

# --------------------- TIME KERNEL ATTENTION -------------------------

class InterpoAttention(nn.Module):
    """
    多头可学习插值注意力 (interpolation + P_i·P_j)。
    现在 time_dim 作为超参，在 __init__ 里完成所有可学习参数的初始化。
    """
    def __init__(self, d_model, n_heads, seq_len, time_dim, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能整除 n_heads"
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.seq_len  = seq_len
        self.dropout  = nn.Dropout(dropout)

        # 普通 Q/K/V/Output 投影
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=True)

        # 将 time_dim 作为超参
        self.time_dim = time_dim

        # 单层时间投影：time_dim -> 1
        self.W_time = nn.Linear(time_dim, 1, bias=True)

        # p_mat, r_mat: [d_model, seq_len]
        self.p_mat = nn.Parameter(torch.randn(d_model, seq_len))
        self.r_mat = nn.Parameter(torch.randn(d_model, seq_len))

        # α, β, γ: [n_heads]
        self.alpha = nn.Parameter(torch.ones(n_heads))
        self.beta  = nn.Parameter(torch.ones(n_heads))
        self.gamma = nn.Parameter(torch.zeros(n_heads))



    def cubic_interp(self, mat, coords):
        """
        Catmull–Rom 三次插值：
        mat:    [d_model, L]
        coords: 任意形状，取值在 [0, L-1]
        返回:
        out:    [..., d_model]
        原理: 对每个实数下标 c，取 i = floor(c)，t = c-i，
        用 p0=mat[:,i-1], p1=mat[:,i], p2=mat[:,i+1], p3=mat[:,i+2]（边界 clamp）
        按 Catmull–Rom 基本公式：
            val = 0.5 * ((2*p1)
                        + (-p0 + p2) * t
                        + (2*p0 - 5*p1 + 4*p2 - p3) * t^2
                        + (-p0 + 3*p1 - 3*p2 + p3) * t^3)
        """

        d, L = mat.shape
        # 1. 扁平化 coords
        flat = coords.reshape(-1)
        N = flat.size(0)

        # 2. 基本下标和 t
        i = torch.floor(flat).long()
        t = (flat - i.float()).unsqueeze(0)  # [1, N]

        # 3. 准备四个邻居的下标 (clamp到[0, L-1])
        im1 = (i - 1).clamp(0, L-1)
        i0  = i.clamp(0, L-1)
        i1  = (i + 1).clamp(0, L-1)
        i2  = (i + 2).clamp(0, L-1)

        # 4. 按这些下标取 mat 值 -> [d, N]
        p0 = mat[:, im1]
        p1 = mat[:, i0 ]
        p2 = mat[:, i1 ]
        p3 = mat[:, i2 ]

        # 5. 计算 Catmull–Rom 插值
        #    所有操作在 [d, N] 上广播 t:[1,N]
        t2 = t * t
        t3 = t2 * t
        term1 = 2*p1
        term2 = (-p0 + p2) * t
        term3 = (2*p0 - 5*p1 + 4*p2 - p3) * t2
        term4 = (-p0 + 3*p1 - 3*p2 + p3) * t3

        val = 0.5 * (term1 + term2 + term3 + term4)  # [d, N]

        # 6. reshape 回原 coords 形状 + d
        out = val.transpose(0,1).reshape(*coords.shape, d)
        return out

    def forward(self, x, t_enc):
        """
        x:     [B, L, d_model]
        t_enc: [B, L, time_dim]
        """
        B, L, _ = x.shape
        H, d    = self.n_heads, self.head_dim

        # 1) Q/K/V -> [B,H,L,d]
        Q = self.W_Q(x).view(B, L, H, d).transpose(1,2)
        K = self.W_K(x).view(B, L, H, d).transpose(1,2)
        V = self.W_V(x).view(B, L, H, d).transpose(1,2)

        # 2) P = x 分头 -> [B,H,L,d]
        P = x.view(B, L, H, d).transpose(1,2)

        # 3) t_enc -> scalar coords in [0,L-1]
        tau = torch.sigmoid(self.W_time(t_enc).squeeze(-1))  # [B,L]
        coords = tau * (L - 1)

        # 4) q_i 插值 -> [B,L,d_model] -> sum -> [B,L]
        q_vec = self.cubic_interp(self.p_mat, coords)
        q_sca = q_vec.sum(-1)

        # 5) r_ij 插值 -> [B,L,L,d_model] -> sum -> [B,L,L]
        delta = (coords.unsqueeze(2) - coords.unsqueeze(1)).abs()
        r_vec = self.linear_interp(self.r_mat, delta)
        r_sca = r_vec.sum(-1)

        # 6) q_i + q_j -> [B,L,L]
        qsum = q_sca.unsqueeze(2) + q_sca.unsqueeze(1)

        # 7) Pij = P_i·P_j -> [B,H,L,L]
        Pij = torch.einsum('bhid,bhjd->bhij', P, P)

        # 8) A_{h,i,j} = [α(q_i+q_j) + β r_ij + γ] * Pij
        a = self.alpha.view(1,H,1,1)
        b = self.beta .view(1,H,1,1)
        c = self.gamma.view(1,H,1,1)
        S = a * qsum.unsqueeze(1) + b * r_sca.unsqueeze(1) + c  # [B,H,L,L]
        A = S * Pij                                           # [B,H,L,L]

        # 9) scores + A -> attn -> out
        scores = (Q @ K.transpose(-2,-1)) / math.sqrt(d) + A
        attn   = self.dropout(F.softmax(scores, dim=-1))
        out    = attn @ V                                    # [B,H,L,d]
        out    = out.transpose(1,2).reshape(B, L, self.d_model)  # [B,L,d_model]

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
            configs.enc_in,            # 原始通道数
            configs.d_model,           # 投影到的维度
            embed_type=configs.embed,  # 时间特征编码类型 (timeF/fixed/learned)
            freq=configs.freq,         # 时间特征频率
            dropout=configs.dropout    # dropout 概率
        )

        if self.patch_flag:
            self.patch_embed = PatchEmbedding(configs.d_model,
                                              configs.patch_len,
                                              configs.patch_stride)
            # 计算 patch 后序列长度
            self.eff_seq_len = math.ceil((self.seq_len
                                     - configs.patch_len + 1)
                                     / configs.patch_stride)
        else:
            self.eff_seq_len = self.seq_len

        # —— 在 __init__ 中一次性初始化 inverted embedding —— 
        # 这样它对应的 nn.Linear 权重在 optimizer 初始化时就已注册
        self.invert_embedding = DataEmbedding_inverted(
            self.eff_seq_len,
            self.d_model,
            self.embed,
            self.freq,
            self.dropout
        )

        # kernel attention
        self.new_layers = nn.ModuleList([
            InterpoAttention(configs.d_model,
                                configs.n_new_heads, configs.seq_len, configs.time_dim, configs.dropout)
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

