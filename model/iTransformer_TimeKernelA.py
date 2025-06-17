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

class TimeKernelAttentionA(nn.Module):
    """
    多头时间高斯核注意力层 (Multi-head Time Kernel Attention)。
    - 普通用线性层生成 Q,K,V
    - 同时计算绝对时间核 g_abs 和 相对时间核 g_rel
    - 按公式 A_ij = α P_ij (g_abs_i + g_abs_j) + β P_ij g_rel_ij + γ
    - 最后用 scores * A 作为最终 attention 得分，再 softmax -> out
    """
    def __init__(self, d_model, n_kernel_heads, num_kernels, time_dim, dropout=0.1):
        super().__init__()
        # 输入维度必须能整除头数
        assert d_model % n_kernel_heads == 0, "d_model 必须能被 n_kernel_heads 整除"
        self.d_model     = d_model                # 总 embedding 维度
        self.n_heads     = n_kernel_heads         # 多头数
        self.head_dim    = d_model // n_kernel_heads
        self.num_kernels = num_kernels            # 每 (head, variate, time_dim) 下 kernel 数量
        self.time_dim    = time_dim               # time_dim 现在作为超参

        # ---- 普通的 Q/K/V 和 输出投影 ----
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

        H, D, T, K = self.n_heads, self.head_dim, self.time_dim, self.num_kernels

        # 绝对时间核参数 μ, σ, w；形状 [H, D, T, K]
        self.mu_abs    = nn.Parameter(torch.randn(H, D, T, K))
        self.sigma_abs = nn.Parameter(torch.randn(H, D, T, K).abs() + 1e-5)
        self.w_abs     = nn.Parameter(torch.randn(H, D, T, K))

        # 相对时间核参数 μ, σ, w；形状同上
        self.mu_rel    = nn.Parameter(torch.randn(H, D, T, K))
        self.sigma_rel = nn.Parameter(torch.randn(H, D, T, K).abs() + 1e-5)
        self.w_rel     = nn.Parameter(torch.randn(H, D, T, K))

        # 融合系数 α, β, γ，每 head 一个
        self.alpha = nn.Parameter(torch.ones(H))
        self.beta  = nn.Parameter(torch.ones(H))
        self.gamma = nn.Parameter(torch.zeros(H))

    def gaussian_kernel(self, t, mu, sigma, w):
        """
        通用高斯核函数：
        - t:      [B, L, T]
        - mu:     [H, D, T, K]
        - sigma:  [H, D, T, K]
        - w:      [H, D, T, K]
        返回:      [B, H, D, L]
        """
        B, L, T = t.shape               # B=batch, L=seq_len, T=time_dim
        H, D, T2, K = mu.shape          # H=heads, D=head_dim, T2, K=num_kernels
        assert T == T2, "time_dim 不匹配"

        # 将 t 扩展到 [B,1,1,L,T,1]
        t_exp = t.view(B,1,1,L,T,1)
        # 将 mu/sigma/w 扩展到 [1,H,D,1,T,K]
        mu_exp    = mu   .view(1,H,D,1,T,K)
        sigma_exp = sigma.view(1,H,D,1,T,K)
        w_exp     = w    .view(1,H,D,1,T,K)

        # 计算 exp(- (t - mu)^2 / 2σ^2) * w -> [B,H,D,L,T,K]
        g = w_exp * torch.exp(- (t_exp - mu_exp)**2 / (2 * sigma_exp**2))
        # 对 time_dim 和 kernel 数量求和 -> [B,H,D,L]
        return g.sum(-1).sum(-1)

    def forward(self, x, t):
        """
        x: [B, L, d_model]
        t: [B, L, time_dim]
        返回: [B, L, d_model]
        """
        B, L, _ = x.shape

        # 1) 普通 Q/K/V 分头 -> [B, H, L, head_dim]
        Q = self.W_Q(x).view(B, L, self.n_heads, self.head_dim).transpose(1,2)
        K = self.W_K(x).view(B, L, self.n_heads, self.head_dim).transpose(1,2)
        V = self.W_V(x).view(B, L, self.n_heads, self.head_dim).transpose(1,2)

        # 2) 计算 Pij = x·x^T，用于公式中 P_ij 项 -> [B, H, L, L]
        Xh  = x.view(B, L, self.n_heads, self.head_dim).transpose(1,2)
        Pij = torch.einsum('bhid,bhjd->bhij', Xh, Xh)

        # 3) 计算绝对时间核 g_abs: [B, H, D, L]
        g_abs = self.gaussian_kernel(t, self.mu_abs, self.sigma_abs, self.w_abs)
        # 4) 计算相对时间核 g_rel：先算 |t_i - t_j| -> [B, L, L, T]
        t_rel = (t.unsqueeze(2) - t.unsqueeze(1)).abs()  # [B,L,L,T]
        # 把它 reshape 成 [B, L*L, T] 调用 gaussian_kernel -> [B,H,D,L*L]
        g_rel = self.gaussian_kernel(
            t_rel.view(B, L*L, self.time_dim),
            self.mu_rel, self.sigma_rel, self.w_rel
        )
        # reshape 回 [B,H,D,L,L]
        g_rel = g_rel.view(B, self.n_heads, self.head_dim, L, L)

        # 5) 对 variate(D) 维做平均
        g_abs_i = g_abs.mean(2)                           # -> [B,H,L]
        g_abs_j = g_abs_i.unsqueeze(-1).expand(-1,-1,-1,L)  # -> [B,H,L,L]
        g_rel   = g_rel.mean(2)                           # -> [B,H,L,L]

        # 6) 按公式合成 A_ij: [B,H,L,L]
        a = self.alpha.view(1,self.n_heads,1,1)
        b = self.beta .view(1,self.n_heads,1,1)
        c = self.gamma.view(1,self.n_heads,1,1)
        A = a * Pij * (g_abs_i.unsqueeze(-1) + g_abs_j) \
          + b * Pij * g_rel \
          + c

        # 7) 常规 scaled dot-product scores
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.head_dim)  # [B,H,L,L]
        # 8) 将 kernel A 与 scores 相乘，再 softmax/dropout
        scores = scores * A
        attn   = self.dropout(F.softmax(scores, dim=-1))

        # 9) 用 attn 加权 V -> [B,H,L,head_dim]
        out = torch.matmul(attn, V)
        # 10) 合并 heads -> [B,L,d_model]
        out = out.transpose(1,2).contiguous().view(B, L, self.d_model)
        # 11) 最后一层线性变换
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
            TimeKernelAttentionA(configs.d_model,
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
