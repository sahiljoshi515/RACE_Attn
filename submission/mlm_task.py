import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch 
from torch.utils.data import DataLoader
import itertools
import math
import torch.nn as nn
from datasets import load_dataset
import time
from tqdm import tqdm
import torch.nn.functional as F
from transformers import BertTokenizerFast, DataCollatorForLanguageModeling
from tokenizers import BertWordPieceTokenizer
import matplotlib.pyplot as plt
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# ------------ CONSTANTS ------------
SAMPLE_SIZE = 20000  # Reduced dataset size

BERT_CONFIG = {
    "vocab_size": 30522,    # Vocabulary size
    "context_length": 512, # Context length
    "emb_dim": 384,         # Embedding dimension
    "n_heads": 4,          # Number of attention heads
    "n_layers": 6,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False,       # Query-Key-Value bias
    "K": 1,
    "L": 1,
    "M": 5
}

# ------------------------------------

# ------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj= nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape   
        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        K = self.W_key(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        V = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)

        out = F.scaled_dot_product_attention(Q, K, V, is_causal=False)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(self.dropout(out))
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att  = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"], num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff    = nn.Sequential(
                        nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
                        nn.GELU(),
                        nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"])
                     )
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.att(x); x = self.drop(x) + h
        h = x
        x = self.norm2(x)
        x = self.ff(x); x = self.drop(x) + h
        return x

# ----------------------------------------------------------

# ---------------------- Angular Model ----------------------
class AngularAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj= nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_query(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        K = self.W_key(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        V = self.W_value(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)

        Q = F.normalize(Q, dim=-1, eps=1e-6)
        K = F.normalize(K, dim=-1, eps=1e-6)

        cos_sim = (Q @ K.transpose(-2,-1)).clamp(-0.999,0.999)
        scores  = 1 - torch.acos(cos_sim)/torch.pi
        W = scores.clamp(min=1e-6).pow(16.0) # Change exponent to adjust attention sharpness
        W = W / (W.sum(-1,keepdim=True)+1e-6)
        W = self.dropout(W)

        out = (W @ V).transpose(1,2).contiguous().view(B, T, -1)
        return self.out_proj(out)

class AngularBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att  = AngularAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            dropout=cfg["drop_rate"], num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff    = nn.Sequential(
                        nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
                        nn.GELU(),
                        nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"])
                     )
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.att(x); x = self.drop(x) + h
        h = x
        x = self.norm2(x)
        x = self.ff(x); x = self.drop(x) + h
        return x
# ------------------------

# ------------- RACE ------------------
class BatchedACE(nn.Module):
    """
    Non-causal BatchedACE with optional shared planes.
    Inputs:
      Khf, Vhf, Qhf : [M, B, T, H, d_k]
    """
    def __init__(self, d_k, K, L, M, device='cpu', share_planes: bool = False):
        super().__init__()
        self.d_k, self.K, self.L, self.M = d_k, K, L, M
        self.R = 1 << K
        self.share_planes = share_planes

        if share_planes:
            # Shared planes [L, K, d_k] --> [d_k, (L*K)]
            planes = torch.randn(L, K, d_k, device=device)
            self.register_buffer('planes_T', planes.view(L * K, d_k).T)   # [d_k, L*K]
        else:
            # Independent planes [M, L, K, d_k] --> [M, d_k, (L*K)]
            planes = torch.randn(M, L, K, d_k, device=device)
            planes = planes.view(M, L * K, d_k).transpose(1, 2)           # [M, d_k, L*K]
            self.register_buffer('planes_T', planes)

        # Prototypes (corners of {-1,+1}^K): [K, R]
        corners = torch.tensor(list(itertools.product([-1., +1.], repeat=K)), device=device)
        self.register_buffer('protos_T', corners.T)                        # [K, R]

        # learnable temperature
        self.logit_temp = nn.Parameter(torch.log(torch.tensor(1.0)))

    def forward(self, Khf, Vhf, Qhf, eps: float = 1e-6):
        # Khf, Vhf, Qhf: [M, B, T, H, d_k]
        M, B, T, H, dk = Khf.shape
        assert M == self.M and dk == self.d_k
        S = self.L * self.R
        scale = self.logit_temp.exp().clamp(1e-2, 10.0) # uncomment when you make temp learnable

        if self.share_planes:
            # Collapse M·B·H → N
            N = M * B * H
            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)  # [N,T,dk]
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(N, T, dk)

            # Projections to L*K
            projK = Kh2 @ self.planes_T                                     # [N,T,L*K]
            projQ = Qh2 @ self.planes_T                                     # [N,T,L*K]
        else:
            # Keep ensembles separate; collapse only B·H
            BH = B * H
            Kh2 = Khf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)  # [M,BH,T,dk]
            Qh2 = Qhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)
            V2  = Vhf.permute(0, 1, 3, 2, 4).contiguous().view(M, BH, T, dk)

            # One GEMM per ensemble
            projK = torch.einsum('mbtd,mds->mbts', Kh2, self.planes_T)        # [M,BH,T,L*K]
            projQ = torch.einsum('mbtd,mds->mbts', Qh2, self.planes_T)
            # Merge M,BH → N
            projK = projK.contiguous().view(M * BH, T, self.L * self.K)       # [N,T,L*K]
            projQ = projQ.contiguous().view(M * BH, T, self.L * self.K)
            V2    = V2.view(M * BH, T, dk)
            N     = M * BH

        # Reshape to [N,T,L,K] and soft-hash → probs over R buckets
        projK = projK.view(N, T, self.L, self.K)
        projQ = projQ.view(N, T, self.L, self.K)

        logitsK = (projK.tanh().div(scale) @ self.protos_T)                   # [N,T,L,R]
        logitsQ = (projQ.tanh().div(scale) @ self.protos_T)                   # [N,T,L,R]
        probsK  = F.softmax(logitsK, dim=-1)                                   # [N,T,L,R]
        probsQ  = F.softmax(logitsQ, dim=-1)                                   # [N,T,L,R]

        # -------- Non-causal bucket summaries over the full sequence --------
        # Collapse buckets L,R → S
        probsK_S = probsK.contiguous().view(N, T, S)                           # [N,T,S]
        probsQ_S = probsQ.contiguous().view(N, T, S)                           # [N,T,S]

        # Weighted sums across time:
        #   b_sum = probsK^T @ V   → [N,S,dk]
        b_sum = probsK_S.transpose(1, 2).bmm(V2)                               # [N,S,dk]
        #   A = sum_t probsK_t     → [N,S]
        A = probsK_S.sum(dim=1)                                                # [N,S]
        #   E = b_sum / (A + eps)  → [N,S,dk]
        E = b_sum / (A.unsqueeze(-1) + eps)                                    # [N,S,dk]

        # Query lookup per time (no prefix): [N,T,S] @ [N,S,dk] → [N,T,dk]
        out2 = probsQ_S.bmm(E)                                                 # [N,T,dk]

        # Unflatten back to [M,B,T,H,dk]
        out = out2.view(M, B, H, T, dk).permute(0, 1, 3, 2, 4)                 # [M,B,T,H,dk]
        return out

class LinearAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False, eps=1e-6):
        super().__init__()
        assert d_out % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads
        self.eps = eps

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def kernel(self, x):
        # φ(x): positive-valued kernel feature map
        return F.elu(x) + 1  # [B, H, T, D]

    def forward(self, x):
        B, T, _ = x.size()

        # Linear projections
        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        K = self.W_key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)    # [B, H, T, D]
        V = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]

        # Apply kernel φ
        Q = self.kernel(Q)  # [B, H, T, D]
        K = self.kernel(K)  # [B, H, T, D]

        # Compute KV^T: [B, H, D, D]
        KV = torch.einsum('bhtd,bhte->bhde', K, V)  # [B, H, D, D]

        # Compute normalization factor: Z = Q * sum(K)
        K_sum = K.sum(dim=2)  # [B, H, D]
        Z = torch.einsum('bhtd,bhd->bht', Q, K_sum) + self.eps  # [B, H, T]

        # Compute output: Q @ (KV)
        context = torch.einsum('bhtd,bhde->bhte', Q, KV)  # [B, H, T, D]
        out = context / Z.unsqueeze(-1)  # [B, H, T, D]

        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # [B, T, H*D]
        return self.out_proj(out)

class LinearBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att  = LinearAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            dropout=cfg["drop_rate"], num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff    = nn.Sequential(
                        nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
                        nn.GELU(),
                        nn.Linear(4*cfg["emb_dim"],cfg["emb_dim"])
                        )
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x) + h
        h = x
        x = self.norm2(x)
        x = self.ff(x); x = self.drop(x) + h
        return x

class RACEAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout,
                 num_heads, L, K, N_M, qkv_bias=False, device='cpu'):
        super().__init__()
        assert d_in % num_heads == 0
        self.H   = num_heads
        self.d_k = d_in // num_heads
        self.M   = N_M

        self.q_proj = nn.Linear(d_in, d_in, bias=qkv_bias)
        self.k_proj = nn.Linear(d_in, d_in, bias=qkv_bias)
        self.v_proj = nn.Linear(d_in, d_in, bias=qkv_bias)
        self.out    = nn.Linear(d_in, d_out)
        self.drop   = nn.Dropout(dropout)
        self.ace = BatchedACE(self.d_k, K, L, N_M, device=device)

    def forward(self, x):
        B, T, _ = x.shape
        H, d_k, M = self.H, self.d_k, self.M

        # 1) project & reshape for ACE
        Q = self.q_proj(x).view(B, T, H, d_k)
        K = self.k_proj(x).view(B, T, H, d_k)
        V = self.v_proj(x).view(B, T, H, d_k)

        # shape --> [M, B, T, H, d_k] by explicit unsqueeze
        def pack(Z):
            Zm = Z.unsqueeze(0).expand(M, -1, -1, -1, -1)
            return Zm

        Khf = pack(K)
        Vhf = pack(V)
        Qhf = pack(Q)

        # 2) run ACE
        out_hm = self.ace(Khf, Vhf, Qhf)  # [M,B,T,H,d_k]

        # 3) average ensembles & merge heads
        out = out_hm.mean(dim=0)          # [B,T,H,d_k]
        out = out.permute(0,2,1,3).reshape(B, T, H * d_k)

        # 4) final proj + dropout
        return self.drop(self.out(out))



class RACEBlock(nn.Module):
    def __init__(self, cfg, device='cpu'):
        super().__init__()
        self.att   = RACEAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"], qkv_bias=cfg["qkv_bias"],
            L=cfg["L"], K=cfg["K"], N_M=cfg["M"]
        )
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff    = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"])
        )
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x) + h
        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x


class LinformerAttention(nn.Module):
    """
    Linformer-style attention: project K,V along sequence length T -> k (k << T),
    then do standard scaled dot-product attention with softmax over k.

    Shapes:
      x: (B, T, d_in)
      returns: (B, T, d_out)
    """
    def __init__(
        self,
        d: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool,
        k_proj_dim: int,      # low-rank sequence dim
        max_seq_len: int    # allocate E up to this T, slice at runtime
    ):
        super().__init__()
        assert d % num_heads == 0, "d_out must be divisible by num_heads"
        self.h = num_heads
        self.dk = d // num_heads
        self.k_proj_dim = k_proj_dim
        self.max_seq_len = max_seq_len

        # token projections
        self.W_query = nn.Linear(d,  d, bias=qkv_bias)
        self.W_key   = nn.Linear(d,  d, bias=qkv_bias)
        self.W_value = nn.Linear(d,  d, bias=qkv_bias)

        # learnable sequence projections E_k, E_v: [T_max, k]
        self.E_k = nn.Parameter(torch.empty(max_seq_len, k_proj_dim))
        self.E_v = nn.Parameter(torch.empty(max_seq_len, k_proj_dim))

        nn.init.xavier_uniform_(self.E_k)
        nn.init.xavier_uniform_(self.E_v)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        assert T <= self.max_seq_len, f"T={T} exceeds max_seq_len={self.max_seq_len}"
        h, dk, k = self.h, self.dk, self.k_proj_dim

        # Linear projections -> (B, h, T, dk)
        Q = self.W_query(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        K = self.W_key(  x).view(B, T, h, dk).transpose(1, 2).contiguous()
        V = self.W_value(x).view(B, T, h, dk).transpose(1, 2).contiguous()

        # Sequence down-projection (T -> k) using E_k/E_v sliced to current T
        Ek = self.E_k[:T]  # (T, k)
        Ev = self.E_v[:T]  # (T, k)

        # K_proj, V_proj: (B, h, k, dk)
        # Contract over sequence axis
        K_proj = torch.einsum("bhtd,tk->bhkd", K, Ek)
        V_proj = torch.einsum("bhtd,tk->bhkd", V, Ev)

        # Scaled dot-product attention over compressed length k
        # scores: (B, h, T, k)
        scale = 1.0 / math.sqrt(dk)
        scores = torch.einsum("bhtd,bhkd->bhtk", Q, K_proj) * scale
        attn = F.softmax(scores, dim=-1)

        # Context: (B, h, T, dk)
        ctx = torch.einsum("bhtk,bhkd->bhtd", attn, V_proj)

        # Merge heads -> (B, T, d_out)
        out = ctx.transpose(1, 2).contiguous().view(B, T, h * dk)
        return self.out_proj(self.dropout(out))


class LinformerBlock(nn.Module):
    """
    Drop-in analogue of your LinearBlock but using LinformerAttention.
    Non-causal, no kernel; just K,V low-rank sequence projection.
    """
    def __init__(self, cfg):
        super().__init__()
        drop = cfg["drop_rate"]
        qkv_bias = cfg["qkv_bias"]
        k_proj_dim = 128

        self.att  = LinformerAttention(
            d=cfg["emb_dim"], dropout=drop, num_heads=cfg["n_heads"], qkv_bias=qkv_bias,
            k_proj_dim=k_proj_dim, max_seq_len=cfg["context_length"]
        )
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff    = nn.Sequential(
                        nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
                        nn.GELU(),
                        nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
                     )
        self.drop  = nn.Dropout(drop)

    def forward(self, x):
        # Attn sublayer
        h = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x) + h

        # FFN sublayer
        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x

def favorplus_features(x, proj, eps=1e-6):
    """
    FAVOR+ positive random features for softmax kernel.
    x:    [B,H,T,D]
    proj: [H,M,D]  (one matrix per head; rows ~ N(0, I))
    ->    [B,H,T,M]  (non-negative)
    """
    # x @ W^T  -> [B,H,T,M]
    xw = torch.einsum('bhtd,hmd->bhtm', x, proj)

    # stabilize across feature dimension
    xw = xw - xw.max(dim=-1, keepdim=True).values

    # exp( xW^T - ||x||^2/2 )
    exp_part  = torch.exp(xw)                         # [B,H,T,M]
    x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)    # [B,H,T,1]
    base      = torch.exp(-0.5 * x_norm_sq)           # [B,H,T,1]
    return exp_part * base + eps                      # strictly positive


class FavorPlusAttention(nn.Module):
    """
    Non-causal FAVOR+ (Performer) attention (softmax kernel via positive RF).
    - Pad-mask aware (mask: 1=keep, 0=pad).
    - Saves pre-projection context in self.last_ctx (B,T,d) and per-head in self.last_ctx_heads (B,H,T,dk).
    """
    def __init__(self, d, h, m_features=256, drop=0.0, qkv_bias=False, seed=None):
        super().__init__()
        assert d % h == 0, "Embedding dim must be divisible by n_heads"
        self.h  = h
        self.dk = d // h
        self.m  = m_features

        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

        # Draw Gaussian projection matrices per head: [H,M,Dk]
        if seed is not None:
            torch.manual_seed(seed)
        proj = torch.randn(h, m_features, self.dk)     # ~ N(0,1)
        self.register_buffer("proj", proj)             # no grad; moves with device

        # For inspection/plots
        self.ctx = None
        self.eps = 1e-6

    def forward(self, x):
        """
        x:    (B, T, d)
        mask: (B, T) with 1/True = keep, 0/False = pad
        return: (B, T, d)
        """
        B, T, d = x.shape
        h, dk, m = self.h, self.dk, self.m

        # Projections -> (B,H,T,dk)
        Q = self.q(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        K = self.k(x).view(B, T, h, dk).transpose(1, 2).contiguous()
        V = self.v(x).view(B, T, h, dk).transpose(1, 2).contiguous()

        # Scale like softmax attention: exp(q·k / sqrt(dk)) ≡ use q/√dk inside features
        Qs = Q / math.sqrt(dk)
        Ks = K / math.sqrt(dk)


        # Positive random features
        phiQ = favorplus_features(Qs, self.proj, eps=self.eps)/ math.sqrt(m)   # [B,H,T,M]
        phiK = favorplus_features(Ks, self.proj, eps=self.eps) / math.sqrt(m)  # [B,H,T,M]


        # Global (non-causal) aggregation over time
        # KV   = sum_t phiK_t^T ⊗ V_t  -> (B,H,M,dk)
        # Ksum = sum_t phiK_t          -> (B,H,M)
        KV   = torch.einsum("bhtm,bhtd->bhmd", phiK, V)
        Ksum = phiK.sum(dim=2)

        # Per-query readout
        # num = phiQ @ KV   -> (B,H,T,dk)
        # den = phiQ · Ksum -> (B,H,T,1)
        num = torch.einsum("bhtm,bhmd->bhtd", phiQ, KV)
        den = torch.einsum("bhtm,bhm->bht",   phiQ, Ksum).unsqueeze(-1) + self.eps
        out_heads = num / den                          # (B,H,T,dk)

        # Save pre-projection context for visualization
        merged = out_heads.transpose(1, 2).contiguous().view(B, T, h * dk)
        self.ctx = merged

        # Standard output path
        merged = self.drop(merged)
        return self.o(merged)
    

class PerformerBlock(nn.Module):
    """
    Residual block with FAVOR+ attention + FFN, mirroring your LinearBlock.
    """
    def __init__(self, cfg):
        super().__init__()
        self.att = FavorPlusAttention(
            d=cfg["emb_dim"],
            h=cfg["n_heads"],
            m_features=cfg.get("m_features", 256),
            drop=cfg["drop_rate"],
            qkv_bias=cfg.get("qkv_bias", False),
            seed=cfg.get("favor_seed", None),
        )
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.ff    = nn.Sequential(
                        nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
                        nn.GELU(),
                        nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"])
                     )
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Pre-norm + attention
        h = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x) + h

        # FFN
        h = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x) + h
        return x


# --------------------------------------

# ------------ MODEL DEFINITION -----------

class MLMModel(nn.Module):
    def __init__(self, cfg, attn_type="gpt", device="cpu"):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb= nn.Dropout(cfg["drop_rate"])
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])

        # (Optional) keep a full-vocab head for convenience / non-MLM tasks
        self.out_head   = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        if attn_type == "bert":
            AttnBlock = TransformerBlock
        elif attn_type == "angular":
            AttnBlock = AngularBlock
        elif attn_type == "race":
            AttnBlock = lambda c: RACEBlock(c, device)
        elif attn_type == "linear":
            AttnBlock = LinearBlock
        elif attn_type == "linformer":
            AttnBlock = LinformerBlock
        elif attn_type == "performer":
            AttnBlock = PerformerBlock
        else:
            raise ValueError(attn_type)

        self.blocks = nn.Sequential(*[AttnBlock(cfg) for _ in range(cfg["n_layers"])])

    def forward_hidden(self, x):
        """Return final hidden states [B,T,d] (use this for MLM masked-only loss)."""
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # [1,T]
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.drop_emb(h)
        h = self.blocks(h)
        h = self.final_norm(h)
        return h  # [B,T,d]

    def forward(self, x):
        """Legacy full-vocab logits for non-MLM use (computes [B,T,V])."""
        h = self.forward_hidden(x)
        return self.out_head(h)


# ------------------------------------

def load_small_tinystories():
    dataset = load_dataset("roneneldan/TinyStories")
    raw_texts = dataset["train"].select(range(SAMPLE_SIZE))
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True, truncation=True, max_length=BERT_CONFIG["context_length"], padding="max_length")

    tokenized = raw_texts.map(tokenize_function, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = (len(concatenated["input_ids"]) // BERT_CONFIG["context_length"]) * BERT_CONFIG["context_length"]
        result = {
            k: [t[i:i + BERT_CONFIG["context_length"]] for i in range(0, total_len, BERT_CONFIG["context_length"])]
            for k, t in concatenated.items()
        }
        return result

    grouped = tokenized.map(group_texts, batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    def tuple_collate_fn(features):
        # Let HuggingFace collator do the masking and batch creation
        batch = data_collator(features)
        return batch["input_ids"], batch["labels"]

    train_size = int(0.9 * len(grouped))
    train_dataset = grouped.select(range(train_size))
    val_dataset = grouped.select(range(train_size, len(grouped)))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=tuple_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=tuple_collate_fn)

    return train_loader, val_loader


# ------------ TRAINING ------------

class LinearWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Linear warmup to base LR for `warmup_steps` optimizer updates,
    then linear decay to 0 by `total_steps`. Step this *per optimizer step*.
    """
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = max(1, int(warmup_steps))
        self.total_steps  = max(self.warmup_steps + 1, int(total_steps))
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1  # count optimizer steps
        lrs = []
        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                lr = base_lr * (step / self.warmup_steps)
            else:
                progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lr = base_lr * (1.0 - progress)
            lrs.append(lr)
        return lrs


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, cfg, grad_accum_steps=1):
    import math, os, time
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    train_times,  val_times  = [], []
    train_ppls,   val_ppls   = [], []

    K, L, M = cfg.get("K", None), cfg.get("L", None), cfg.get("M", None)
    out_path = f"trial_K{K}_L{L}_M{M}_MLM.txt"

    # ---- schedule based on *optimizer steps* (not micro-steps) ----
    steps_per_epoch   = len(train_loader)                                   # micro-steps
    updates_per_epoch = math.ceil(steps_per_epoch / grad_accum_steps)       # optimizer steps
    total_updates     = num_epochs * updates_per_epoch
    warmup_updates    = max(1, int(0.01 * total_updates))                   # 1% warmup

    scheduler = LinearWarmupLR(
        optimizer,
        warmup_steps=warmup_updates,
        total_steps=total_updates,
    )

    def _log(fp, msg):
        print(msg); fp.write(msg + "\n"); fp.flush()

    with open(out_path, "a", encoding="utf-8") as f:
        _log(f, f"Epochs: {num_epochs}")
        _log(f, "-" * 72)

        global_update = 0
        for epoch in range(1, num_epochs + 1):
            # === TRAIN ===
            t0 = time.time()
            model.train()
            total_loss = 0.0
            total_correct = 0
            total_tokens  = 0

            optimizer.zero_grad(set_to_none=True)
            accum_count = 0

            for input_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
                input_ids = input_ids.to(device)
                labels    = labels.to(device)

                # (1) forward to hidden states for full sequence
                h    = model.forward_hidden(input_ids)  # [B,T,d]
                mask = (labels != -100)                 # [B,T], True on masked positions

                # (2) select masked positions
                if mask.any():
                    h_mask = h[mask]                    # [Nmask, d]
                    y_mask = labels[mask]               # [Nmask]

                    # (3) weight tying: use embedding matrix as output weights
                    W = model.tok_emb.weight            # [V, d]
                    logits_mask = F.linear(h_mask, W)   # [Nmask, V]

                    loss = F.cross_entropy(logits_mask, y_mask)
                    # accuracy on masked tokens
                    preds = logits_mask.argmax(dim=-1)
                    total_correct += (preds == y_mask).sum().item()
                    total_tokens  += y_mask.numel()
                else:
                    # keep graph valid if a batch happens to have no masked tokens
                    loss = h.sum() * 0.0

                # ---- accumulation: scale loss and delay optimizer step ----
                (loss / grad_accum_steps).backward()
                accum_count += 1
                total_loss  += loss.item()

                if accum_count == grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()    # step scheduler *per optimizer step*
                    optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    global_update += 1

            # flush remainder if last batch didn't align with boundary
            if accum_count > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

            train_time = time.time() - t0
            train_times.append(train_time)
            tr_l = total_loss / max(1, len(train_loader))
            tr_a = (total_correct / total_tokens) if total_tokens > 0 else 0.0
            tr_p = math.exp(tr_l)  # masked-token perplexity
            train_losses.append(tr_l); train_accs.append(tr_a); train_ppls.append(tr_p)

            # === VALIDATION ===
            t1 = time.time()
            model.eval()
            val_loss_total = 0.0
            val_correct = 0
            val_tokens  = 0
            with torch.no_grad():
                for input_ids, labels in val_loader:
                    input_ids = input_ids.to(device)
                    labels    = labels.to(device)

                    h    = model.forward_hidden(input_ids)  # [B,T,d]
                    mask = (labels != -100)

                    if mask.any():
                        h_mask = h[mask]
                        y_mask = labels[mask]
                        W = model.tok_emb.weight
                        logits_mask = F.linear(h_mask, W)
                        loss = F.cross_entropy(logits_mask, y_mask)
                        val_loss_total += loss.item()
                        preds = logits_mask.argmax(dim=-1)
                        val_correct += (preds == y_mask).sum().item()
                        val_tokens  += y_mask.numel()
                    else:
                        pass

            val_time = time.time() - t1
            val_times.append(val_time)
            va_l = val_loss_total / max(1, len(val_loader))
            va_a = (val_correct / val_tokens) if val_tokens > 0 else 0.0
            va_p = math.exp(va_l)
            val_losses.append(va_l); val_accs.append(va_a); val_ppls.append(va_p)

            _log(
                f,
                (f"Ep{epoch:2d} | "
                 f"train_loss {tr_l:.3f}, acc {tr_a:.3f}, ppl {tr_p:.1f} ({train_time:.1f}s) | "
                 f"val_loss {va_l:.3f}, acc {va_a:.3f}, ppl {va_p:.1f} ({val_time:.1f}s) | "
                 f"updates {global_update}/{total_updates}")
            )

        _log(f, "-" * 72)
        _log(f, f"Log saved to: {os.path.abspath(out_path)}")

    return {
        "train_loss": train_losses, "val_loss": val_losses,
        "train_ppl": train_ppls,   "val_ppl": val_ppls,
        "train_acc": train_accs,   "val_acc": val_accs,
        "train_time": train_times, "val_time": val_times,
    }




def start_experiment():
    device = "cuda:1"
    train_loader, val_loader = load_small_tinystories()
    print(len(train_loader), len(val_loader))
    num_epochs = 100
    # ------------------ TRAINING MODELS ----------------
    # print("Training BERT model...")
    # torch.manual_seed(123)
    # model_bert = torch.compile(MLMModel(BERT_CONFIG, "bert"))
    # model_bert.to(device)
    # optimizer_bert = torch.optim.AdamW(model_bert.parameters(), lr=6e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)

    # metrics_bert = train_model_simple(
    #     model_bert, train_loader, val_loader, optimizer_bert, device,
    #     num_epochs=num_epochs, cfg=BERT_CONFIG)
    
    # print("Training RACE model...")
    # torch.manual_seed(123)
    # model_race = torch.compile(MLMModel(BERT_CONFIG, "race"))
    # model_race.to(device)
    # print(sum(p.numel() for p in model_race.parameters() if p.requires_grad))
    # optimizer_race = torch.optim.AdamW(model_race.parameters(), lr=6e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)

    # metrics_race = train_model_simple(
    #     model_race, train_loader, val_loader, optimizer_race, device,
    #     num_epochs=num_epochs, cfg=BERT_CONFIG)
    
    print("Training PERFORMER model...")
    torch.manual_seed(123)
    model_performer = torch.compile(MLMModel(BERT_CONFIG, "performer"))
    model_performer.to(device)
    optimizer_performer = torch.optim.AdamW(model_performer.parameters(), lr=6e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)

    metrics_performer = train_model_simple(
        model_performer, train_loader, val_loader, optimizer_performer, device,
        num_epochs=num_epochs, cfg=BERT_CONFIG)
    
    # print("Training LinearAttention...")
    # torch.manual_seed(123)
    # model_linear = torch.compile(MLMModel(BERT_CONFIG, "linear"))
    # model_linear.to(device)
    # optimizer_linear = torch.optim.AdamW(model_linear.parameters(), lr=6e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)

    # metrics_linear = train_model_simple(
    #     model_linear, train_loader, val_loader, optimizer_linear, device,
    #     num_epochs=num_epochs, cfg=BERT_CONFIG)

    # print("Training Linformer...")
    # torch.manual_seed(123)
    # model_linformer = torch.compile(MLMModel(BERT_CONFIG, "linformer"))
    # model_linformer.to(device)
    # optimizer_linformer = torch.optim.AdamW(model_linformer.parameters(), lr=6e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)

    # metrics_linformer = train_model_simple(
    #     model_linformer, train_loader, val_loader, optimizer_linformer, device,
    #     num_epochs=num_epochs, cfg=BERT_CONFIG)

    # print("Training Angular model...")
    # torch.manual_seed(123)
    # model_angular = torch.compile(MLMModel(BERT_CONFIG, "angular"))
    # model_angular.to(device)
    # optimizer_angular = torch.optim.AdamW(model_angular.parameters(), lr=6e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)

    # metrics_angular = train_model_simple(
    #     model_angular, train_loader, val_loader, optimizer_angular, device,
    #     num_epochs=num_epochs, cfg=BERT_CONFIG)



start_experiment()