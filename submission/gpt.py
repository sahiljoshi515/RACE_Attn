from torch import nn
import torch 
import torch.nn.functional as F
from torch.autograd import Function

linear_pref = "" # Basline CPU kernel: This is basically the CPU kernel found at - https://github.com/idiap/fast-transformers/blob/master/fast_transformers/causal_product/causal_product_cpu.cpp
# ------------ GPT Model ----------------
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
        self.register_buffer("mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        K = self.W_key(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)
        V = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1,2)

        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)
    
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
    
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        _ , seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# ----------------------------------------------------------

# ---------------------- Angular Model ----------------------
class AngularAttention(nn.Module):
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
        self.register_buffer("mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_query(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        K = self.W_key(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        V = self.W_value(x).view(B,T,self.num_heads,self.head_dim).transpose(1,2)

        Q = F.normalize(Q, dim=-1, eps=1e-6)
        K = F.normalize(K, dim=-1, eps=1e-6)

        cos_sim = (Q @ K.transpose(-2,-1)).clamp(-0.999,0.999)
        scores  = 1 - torch.acos(cos_sim)/torch.pi
        scores.masked_fill_(self.mask[:T,:T], 0.0)
        W = scores.clamp(min=1e-6).pow(8) # Change exponent to adjust attention sharpness
        W = W / (W.sum(-1,keepdim=True)+1e-6)
        W = self.dropout(W)

        out = (W @ V).transpose(1,2).reshape(B, T, -1)
        return self.out_proj(out)

class AngularBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att  = AngularAttention(
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
    
class AngularModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[AngularBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        _ , seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
# ----------------------------------------------------------------------

class CausalDotProductFn(Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        """
        Q, K, V: [B, H, T, D] (float32 CPU)
        returns: out [B, H, T, Dv] (Dv==D in your code)
        """
        # Extension is float32 CPU-only per your code; ensure dtype/device
        assert Q.dtype == torch.float32 and K.dtype == torch.float32 and V.dtype == torch.float32
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
        assert Q.device.type == 'cpu' and K.device.type == 'cpu' and V.device.type == 'cpu'

        out = torch.empty(Q.size(0), Q.size(1), Q.size(2), V.size(3),
                          dtype=Q.dtype, device=Q.device)
        # The C++ fn signature: (queries, keys, values, product) -> None
        linear_pref.causal_dot_product(Q, K, V, out)
        ctx.save_for_backward(Q, K, V)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        grad_out: [B, H, T, Dv]
        returns grads for (Q, K, V)
        """
        Q, K, V = ctx.saved_tensors
        # Allocate grads
        gQ = torch.zeros_like(Q)
        gK = torch.zeros_like(K)
        gV = torch.zeros_like(V)

        # C++ backward signature:
        # (queries, keys, values, grad_out, grad_queries, grad_keys, grad_values) -> None
        linear_pref.causal_dot_backward(Q, K, V, grad_out.contiguous(), gQ, gK, gV)
        return gQ, gK, gV
    
class CausalLinearAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False, eps=1e-6):
        super().__init__()
        assert d_out % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_out // num_heads
        self.eps = eps

        self.W_query = nn.Linear(d_in,  d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in,  d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in,  d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def kernel(x):
        # φ(x): positive feature map (ELU+1 is the common choice)
        return F.elu(x, inplace=False) + 1.0
    
    def forward(self, x):
        B, T, _ = x.shape
        # Linear projections -> [B, H, T, D]
        Q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Feature map φ on Q,K
        Q = self.kernel(Q)  # [B,H,T,D]
        K = self.kernel(K)  # [B,H,T,D]

        Z = 1/(torch.einsum("nlhi,nlhi->nlh", Q, K.cumsum(1)) + self.eps)
        unnorm = CausalDotProductFn.apply(Q.contiguous(), K.contiguous(), V.contiguous())  # [B,H,T,D]
        out = unnorm / Z.unsqueeze(-1)                     # [B,H,T,D]

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.out_proj(out)
        return self.dropout(out)

class CausalLinearBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att  = CausalLinearAttention(
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