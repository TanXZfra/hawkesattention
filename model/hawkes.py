import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder2, DecoderLayer2
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding2,DataEmbedding_inverted

#  PATCHING 
# Useless, just ignore
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




class HawkesAttention(nn.Module):
    def __init__(self, new_d_model, n_new_heads, time_dim, tmlp_width, tmlp_depth,  activation,dropout=0.1):
        super().__init__()
        assert new_d_model % n_new_heads == 0, "d_model must be divisible by n_heads"
        self.new_d_model = new_d_model          # total embedding dimension
        self.n_new_heads = n_new_heads             # number of attention heads
        self.head_dim = new_d_model // n_new_heads    # dimension per head
        self.time_dim = time_dim                  # time embedding dimension(auto,no need to set)
        self.scale = math.sqrt(self.head_dim)      
        self.dropout= nn.Dropout(dropout)          

        if activation.lower() == "relu":
            Act = nn.ReLU
        elif activation.lower() == "gelu":
            Act = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        # Linear projections for Q, K, V (shared across heads)
        self.W_Q = nn.Linear(new_d_model, new_d_model, bias=False)
        self.W_K = nn.Linear(new_d_model, new_d_model, bias=False)
        self.W_V = nn.Linear(new_d_model, new_d_model, bias=False)

        # Factory to build a single-head φ MLP (maps time_dim -> 1 scalar)
        def build_phi_single():
            layers = []
            in_d = time_dim
            # hidden layers
            for _ in range(tmlp_depth):
                layers.append(nn.Linear(in_d, tmlp_width))
                layers.append(Act())
                layers.append(nn.Dropout(dropout))
                in_d = tmlp_width
            # final output layer produces one scalar for this head
            layers.append(nn.Linear(in_d, 1))
            return nn.Sequential(*layers)

        # Create per-head φ_Q, φ_K, φ_V MLP lists
        self.phi_Q = nn.ModuleList([build_phi_single() for _ in range(n_new_heads)])
        self.phi_K = nn.ModuleList([build_phi_single() for _ in range(n_new_heads)])
        self.phi_V = nn.ModuleList([build_phi_single() for _ in range(n_new_heads)])

    def forward(self, Q, K, V, t_Q, t_K, hawkes_self_attn_mask=True):
        """
        Q: [B, Lq, d_model]  Query token embeddings
        K: [B, Lk, d_model]  Key token embeddings
        V: [B, Lk, d_model]  Value token embeddings
        t_Q: [B, Lq, time_dim]  Query time embeddings
        t_K: [B, Lk, time_dim]  Key time embeddings
        t_V: [B, Lk, time_dim]  Value time embeddings
        causal_mask: [Lq, Lk]  Causal mask for self-attention (optional)
        returns out: [B, Lq, d_model]
        """
        B, Lq, _ = Q.shape
        _, Lk, _ = K.shape

        # Q0/K0/V0: [B, Lq/Lk, d_model] -> [B, Lq/Lk, H, head_dim] -> [B, H, Lq/Lk, head_dim]
        Q0 = self.W_Q(Q).view(B, Lq, self.n_new_heads, self.head_dim).transpose(1, 2)
        K0 = self.W_K(K).view(B, Lk, self.n_new_heads, self.head_dim).transpose(1, 2)
        V0 = self.W_V(V).view(B, Lk, self.n_new_heads, self.head_dim).transpose(1, 2)

        # Δt_{j,i}: [B, Lq, Lk, time_dim]
        tQ = t_Q.unsqueeze(2)  # [B, Lq, 1, time_dim]
        tK = t_K.unsqueeze(1)  # [B, 1, Lk, time_dim]
        delta = tQ - tK        # [B, Lq, Lk, time_dim]

        # For each head h, run its φ MLP on Δt to get modulation scalars [B, Lq, Lk] -> [B, H, Lq, Lk]
        phiQ = torch.stack([self.phi_Q[h](delta).squeeze(-1) for h in range(self.n_new_heads)], dim=1)
        phiK = torch.stack([self.phi_K[h](delta).squeeze(-1) for h in range(self.n_new_heads)], dim=1)
        phiV = torch.stack([self.phi_V[h](delta).squeeze(-1) for h in range(self.n_new_heads)], dim=1)

        # Apply φ scalars: [B, H, Lq, Lk, head_dim]
        Q_mod = phiQ.unsqueeze(-1) * Q0.unsqueeze(3)
        K_mod = phiK.unsqueeze(-1) * K0.unsqueeze(2)
        V_mod = phiV.unsqueeze(-1) * V0.unsqueeze(2)

        # [B,H,T,L]
        scores = (Q_mod * K_mod).sum(-1) / self.scale

        # masking for self attention
        if hawkes_self_attn_mask:
            # Causal mask for hawkes self-attention: [Lk, Lq]
            # scores: [B, H, Lq, Lk]
            # causal_mask: [Lk, Lq] -> [1, 1, Lk, Lq]
            # Mask out future positions in self-attention
            # scores: [B, H, Lq, Lk] -> [B, H, Lq, Lk]
            causal_mask = torch.triu(torch.ones(Lk, Lq, dtype=torch.bool, device=scores.device),
                            diagonal=0)     
            mask = causal_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(~mask, float('-inf'))


         # Compute attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Compute output: [B, H, Lq, head_dim]
        out = (attn.unsqueeze(-1) * V_mod).sum(-2)

        # Merge heads: [B, Lq, d_model]
        out = out.transpose(1, 2).reshape(B, Lq, self.new_d_model)
        return out



class HawkesEncoderLayer(nn.Module):
    """
    This encoder structure comes from the iTransformer paper, with vanilla attention replaced by HawkesAttention
    x' = x_enc + Dropout(att(x_enc, t_enc, t_dec))
    x'' = LayerNorm(x')
    y = Dropout(act(Conv1(x''))) → Dropout(Conv2(y))
     out = LayerNorm(x'' + y)
    """
    def __init__(self, attention,new_d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * new_d_model
        self.attn = attention
        self.norm1 = nn.LayerNorm(new_d_model)
        self.norm2 = nn.LayerNorm(new_d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(new_d_model, d_ff, kernel_size= 1)
        self.conv2 = nn.Conv1d(d_ff, new_d_model, kernel_size= 1)
        self.activation = F.relu if activation.lower()=="relu" else F.gelu

    def forward(self, Q, K, V, t_Q, t_K, hawkes_self_attn_mask=True):
        h = self.attn(Q, K, V, t_Q, t_K, hawkes_self_attn_mask)
        x2 = Q + self.dropout(h)
        x2 = self.norm1(x2)
        y = x2.transpose(1,2)
        y = self.activation(self.conv1(y))
        y = self.dropout(y)
        y = self.conv2(y)                
        y = self.dropout(y).transpose(1,2)      
        return self.norm2(x2 + y)

class Projector(nn.Module):
    def __init__(self, seq_len, pred_len, new_d_model, enc_in):
        super(Projector, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.new_d_model = new_d_model
        self.enc_in = enc_in

        # [B, seq_len, new_d_model] -> [B, pred_len, new_d_model]
        self.map_to_pred_len = nn.Linear(seq_len, pred_len)

        # [B, pred_len, new_d_model] -> [B, pred_len, enc_in]
        self.map_to_enc_in = nn.Linear(new_d_model, enc_in)

    def forward(self, x):
        # x: [B, seq_len, new_d_model]
        x = x.permute(0, 2, 1)  # [B, new_d_model, seq_len]
        x = self.map_to_pred_len(x)  # [B, new_d_model, pred_len]
        x = x.permute(0, 2, 1)  # [B, pred_len, new_d_model]
        x = self.map_to_enc_in(x)  # [B, pred_len, enc_in]
        return x


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

        self.new_layers=nn.ModuleList([HawkesEncoderLayer(
            attention=HawkesAttention(configs.new_d_model,configs.n_new_heads,configs.time_dim,configs.tmlp_width,configs.tmlp_depth,configs.activation,
                                       configs.dropout
            ),new_d_model=configs.new_d_model,d_ff=configs.d_ff,dropout=configs.dropout,activation=configs.activation)
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
        
        self.new_projector = Projector(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            new_d_model=self.new_d_model,
            enc_in=configs.enc_in
        )


        self.decoder = Decoder2(
            layers=[
                DecoderLayer2(
                    self_attention=HawkesAttention(configs.new_d_model, configs.n_new_heads, configs.time_dim,
                                                   configs.tmlp_width, configs.tmlp_depth, configs.activation,
                                                   configs.dropout),
                    cross_attention=HawkesAttention(configs.new_d_model, configs.n_new_heads, configs.time_dim,
                                                    configs.tmlp_width, configs.tmlp_depth, configs.activation,
                                                    configs.dropout),
                    d_model=configs.new_d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.d_layers)
            ],
            norm_layer=nn.LayerNorm(configs.new_d_model)
        )

        self.decoder_projector=nn.Linear(self.new_d_model, configs.enc_in)



    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape
        # embedding
        enc_out = self.enc_embedding(x_enc, None)
        x_dec = self.enc_embedding(x_dec, None)
        # patch
        if self.patch_flag:
            #print("patching")
            enc_out, x_mark_enc = self.patch_embed(enc_out, x_mark_enc)
        # kernel attention
        # print("enc_out shape:", enc_out.shape)
        # print("x_mark_enc shape:", x_mark_enc.shape)
        for layer in self.new_layers:
            enc_out = layer(enc_out, enc_out, enc_out, x_mark_enc,x_mark_enc, hawkes_self_attn_mask=False)

        # inverted
        #enc_out=self.invert_embedding(enc_out,x_mark_enc)
        # # encoder
        #enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # projection & permute

        #dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        #dec_out = self.new_projector(enc_out)

        # decoder
        dec_out = self.decoder(
            x=x_dec,  # Ground truth for self-attention
            cross=enc_out,  # Cross-attention input from encoder
            t_x=x_mark_dec,  # Time features for self-attention
            t_cross=x_mark_enc,  # Time features for cross-attention
            x_mask=True,
            cross_mask=False
        )
        dec_out = self.decoder_projector(dec_out)


        # denormalization
        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out= self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :]   
        else:
            return dec_out[:, -self.pred_len:, :]
