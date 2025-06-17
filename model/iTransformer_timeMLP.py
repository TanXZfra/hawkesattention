import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding2,DataEmbedding_inverted

#  PATCHING 
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
        #[B, d_model, L]
        x = x.permute(0, 2, 1)
        # [B, d_model, n_patches]
        x_patch = self.proj(x)
        # [B, n_patches, d_model]
        x_patch = x_patch.transpose(1, 2)
        #  [B, L, time_dim] -> [B, n_patches, time_dim]
        patches = time.unfold(1, self.patch_len, self.stride)  # [B, n_patches, time_dim, patch_len]
        time_patch = patches.mean(-1)
        return x_patch, time_patch

# --------------------- ATTENTION -------------------------


class MLPTimeAttention(nn.Module):
    def __init__(self, new_d_model, n_new_heads, time_dim, tmlp_width, tmlp_depth, activation, dropout=0.1):
        super().__init__()
        assert new_d_model % n_new_heads == 0
        self.new_d_model   = new_d_model
        self.n_new_heads   = n_new_heads
        self.head_dim  = new_d_model // n_new_heads
        self.time_dim  = time_dim
        self.dropout   = nn.Dropout(dropout)
        self.sqrt_d    = math.sqrt(self.head_dim)

        #  W_space: x @ W_space ,P_i W
        self.W_Q_space = nn.Linear(new_d_model, new_d_model, bias=False)
        self.W_K_space = nn.Linear(new_d_model, new_d_model, bias=False)
        self.W_V_space = nn.Linear(new_d_model, new_d_model, bias=False)
        if activation.lower() == "relu":
            Act = nn.ReLU
        elif activation.lower() == "gelu":
            Act = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        def build_mlp(out_dim):
            layers = []
            in_dim = time_dim
            for _ in range(tmlp_depth):
                layers.append(nn.Linear(in_dim, tmlp_width))
                layers.append(Act())
                layers.append(nn.Dropout(dropout))
                in_dim = tmlp_width
            layers.append(nn.Linear(in_dim, out_dim))
            return nn.Sequential(*layers)

        self.mlp_Q = build_mlp(new_d_model)
        self.mlp_K = build_mlp(new_d_model)
        self.mlp_V = build_mlp(new_d_model)
        self.mlp_A = build_mlp(n_new_heads)

        self.W_O = nn.Linear(new_d_model, new_d_model, bias=True)

    def forward(self, x, t_emb):
        """
        x:     [B, L, new_d_model]  
        t_emb: [B, L, time_dim]
        return:[B, L, new_d_model]
        """
        B, L, _ = x.shape
        H, d    = self.n_new_heads, self.head_dim

        # 1) time MLP -> Qm/Km/Vm: [B,L,new_d_model]  -> [B,H,L,d]
        Qm = self.mlp_Q(t_emb).view(B, L, H, d).transpose(1,2)
        Km = self.mlp_K(t_emb).view(B, L, H, d).transpose(1,2)
        Vm = self.mlp_V(t_emb).view(B, L, H, d).transpose(1,2)

        # 2. SW: x @ W_space -> [B,L,new_d_model] -> [B,H,L,d]
        SWQ = self.W_Q_space(x).view(B, L, H, d).transpose(1,2)
        SWK = self.W_K_space(x).view(B, L, H, d).transpose(1,2)
        SWV = self.W_V_space(x).view(B, L, H, d).transpose(1,2)
        Q = Qm + SWQ
        K = Km + SWK
        V = Vm + SWV

        #  A：pairwise Δt -> [B,L,L,time_dim] -> MLP_A -> [B,L,L,n_heads]
        delta = (t_emb.unsqueeze(2) - t_emb.unsqueeze(1))  # [B,L,L,time_dim]
        A_raw = self.mlp_A(delta)                          # [B,L,L,n_heads]
        A = A_raw.permute(0,3,1,2)                         # [B,H,L,L]

        scores = (Q @ K.transpose(-2,-1) * A) / self.sqrt_d    # [B,H,L,L]
        scores = scores
        attn   = self.dropout(F.softmax(scores, dim=-1))

        # attn @ V -> [B,L,new_d_model]
        out = attn @ V                                      # [B,H,L,d]
        out = out.transpose(1,2).contiguous().view(B, L, self.new_d_model)
        out = self.W_O(self.dropout(out))
        return out
    

class MLPTimeEncoderLayer(nn.Module):
    """
      1) x' = x + Dropout(attn(x, t_emb))
      2) x'' = LayerNorm(x')
      3) y = Dropout(activation(Conv1(x''))) → Dropout(Conv2(y))
      4) output = LayerNorm(x'' + y)
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # position‐wise feed-forward
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)

        # choose activation
        if activation.lower() == "relu":
            self.activation = F.relu
        else:
            self.activation = F.gelu

    def forward(self, x, t_emb, attn_mask=None):
        attn_out = self.attention(x, t_emb)      # [B, L, d_model]
        x2 = x + self.dropout(attn_out)

        x2 = self.norm1(x2)

        #    conv1 expects [B, d_model, L]
        y = x2.transpose(1, 2)                   # [B, d_model, L]
        y = self.conv1(y)                        # [B, d_ff, L]
        y = self.activation(y)
        y = self.dropout(y)
        y = self.conv2(y)                        # [B, d_model, L]
        y = self.dropout(y)
        y = y.transpose(1, 2)                    # [B, L, d_model]

        out = self.norm2(x2 + y)
        return out

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
        self.new_d_model=configs.new_d_model
        self.embed=configs.embed
        self.freq=configs.freq
        self.dropout=configs.dropout

        self.enc_embedding = DataEmbedding2(
            configs.enc_in,            
            configs.new_d_model,           
            embed_type=configs.embed,  
            freq=configs.freq,       
            dropout=configs.dropout  
        )

        if self.patch_flag:
            self.patch_embed = PatchEmbedding(configs.new_d_model,
                                              configs.patch_len,
                                              configs.patch_stride)
            # cal patch len
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
        # self.new_layers = nn.ModuleList([
        #     MLPTimeAttention(configs.d_model,
        #                         configs.n_new_heads,configs.time_dim, configs.tmlp_width,configs.tmlp_depth,configs.activation,configs.dropout)
        #     for _ in range(configs.num_new_layers)
        # ])

        self.new_layers=nn.ModuleList([MLPTimeEncoderLayer(
            attention=MLPTimeAttention(configs.new_d_model,configs.n_new_heads,configs.time_dim,configs.tmlp_width,configs.tmlp_depth,configs.activation,
                                       configs.dropout
            ),d_model=configs.new_d_model,d_ff=configs.d_ff,dropout=configs.dropout,activation=configs.activation)
            for _ in range(configs.num_new_layers)])
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
            print("patching")
            enc_out, x_mark_enc = self.patch_embed(enc_out, x_mark_enc)
        # kernel attention
        # print("enc_out shape:", enc_out.shape)
        # print("x_mark_enc shape:", x_mark_enc.shape)
        for layer in self.new_layers:
            enc_out = layer(enc_out, x_mark_enc)

        # inverted
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
