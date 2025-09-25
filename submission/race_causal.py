import torch
from torch import nn
import itertools
import torch.nn.functional as F
import math
from torch.autograd import Function
from race_ext import race_pref

class RACEPrefixMeanFlatFn(Function):
    @staticmethod
    def forward(ctx, probsK_flat: torch.Tensor, V_flat: torch.Tensor, eps: float):
        E_flat = race_pref.race_prefix_mean_flat(probsK_flat, V_flat, float(eps))
        ctx.save_for_backward(probsK_flat, V_flat)
        ctx.eps = float(eps)
        return E_flat

    @staticmethod
    def backward(ctx, gradE_flat):
        probsK_flat, V_flat = ctx.saved_tensors
        gradE_flat = gradE_flat.contiguous()
        gW_flat, gV_flat = race_pref.race_prefix_mean_flat_bw(
            probsK_flat, V_flat, gradE_flat, ctx.eps
        )
        return gW_flat, gV_flat, None  # None for eps

class BatchedACE(nn.Module):
    """
    Causal (LM) BatchedACE that optionally uses a single shared sketch
    or independent sketches per ensemble.
    Inputs:
      Khf, Vhf, Qhf : [M, B, T, H, d_k]
    """
    def __init__(self, d_k, K, L, M, device='cpu', share_planes: bool = True):
        super().__init__()
        self.d_k, self.K, self.L, self.M = d_k, K, L, M
        self.R = 1 << K
        self.share_planes = share_planes

        if share_planes:
            # Shared planes [L, K, d_k] --> [d_k, (L*K)]
            planes = torch.randn(L, K, d_k, device=device)
            self.register_buffer(
              'planes_T',
              planes.view(L*K, d_k).T
            )  # [d_k, L*K]
        else:
            # Independent planes [M, L, K, d_k] --> [M, d_k, (L*K)]
            planes = torch.randn(M, L, K, d_k, device=device)
            planes = planes.view(M, L*K, d_k).transpose(1,2)
            self.register_buffer('planes_T', planes)
            # planes_T: [M, d_k, L*K]

        # flatten protos [R, K] --> [K, R]
        corners = torch.tensor(
            list(itertools.product([-1., +1.], repeat=K)),
            device=device
        )
        self.register_buffer('protos_T', corners.T)  # [K, R]

        # # learnable temperature
        # self.logit_temp = nn.Parameter(torch.log(torch.tensor(1.0)))

    def forward(self, Khf, Vhf, Qhf):
        # [M, B, T, H, d_k]
        M, B, T, H, dk = Khf.shape
        assert M == self.M and dk == self.d_k
        scale = math.sqrt(dk)
        # scale = self.logit_temp.exp().clamp(1e-2, 10.0) # uncomment when you make scale learnable

        # collapse dims → [?, T, d_k]
        if self.share_planes:
            # full collapse over M·B·H
            N = M * B * H
            Kh2 = Khf.permute(0,1,3,2,4).contiguous().view(N, T, dk)
            Qh2 = Qhf.permute(0,1,3,2,4).contiguous().view(N, T, dk)
            V2  = Vhf.permute(0,1,3,2,4).contiguous().view(N, T, dk)

            # single GEMM per projection
            projK = Kh2 @ self.planes_T                # [N, T, L*K]
            projQ = Qh2 @ self.planes_T                # [N, T, L*K]
        else:
            # collapse only batch+head per ensemble: [M, BH, T, d_k]
            BH = B * H
            Kh2 = Khf.permute(0,1,3,2,4).contiguous().view(M, BH, T, dk)
            Qh2 = Qhf.permute(0,1,3,2,4).contiguous().view(M, BH, T, dk)
            V2  = Vhf.permute(0,1,3,2,4).contiguous().view(M, BH, T, dk)

            # one batched GEMM across ensembles
            projK = torch.einsum('mbtd, mds -> mbts', Kh2, self.planes_T)
            projQ = torch.einsum('mbtd, mds -> mbts', Qh2, self.planes_T)
            # merge M,BH → N
            projK = projK.contiguous().view(M*BH, T, self.L*self.K)
            projQ = projQ.contiguous().view(M*BH, T, self.L*self.K)
            V2    = V2.view(M*BH, T, dk)
            N     = M * BH

        # reshape --> [N, T, L, K]
        projK = projK.view(N, T, self.L, self.K)
        projQ = projQ.view(N, T, self.L, self.K)

        # soft‑hash & bucket‑protos
        logitsK = (projK.tanh().div(scale) @ self.protos_T)  # [N, T, L, R]
        probsK  = F.softmax(logitsK, dim=-1)
        logitsQ = (projQ.tanh().div(scale) @ self.protos_T)
        probsQ  = F.softmax(logitsQ, dim=-1)

        # 2) causal prefix sums
        A_pref = probsK.cumsum(dim=1)                                    # [N, T, L, R]
        B_pref = (probsK.unsqueeze(-1) * V2.unsqueeze(2).unsqueeze(3)).cumsum(dim=1)                                       # [N, T, L, R, d_k]
        E_pref = B_pref.div(A_pref.unsqueeze(-1).add(1e-6))              # [N, T, L, R, d_k]

        S      = self.L * self.R

        # If a user wants to run this on CPU - they can uncomment the following and comment out the #2 on top.
        # probsK_flat = probsK.permute(0, 2, 3, 1).contiguous().view(N * S, T)

        # # V2: [N,T,dk] -> duplicate across S then align time -> [NS, T, dk]
        # V_flat = (
        #     V2.unsqueeze(2).unsqueeze(3)           # [N,T,1,1,dk]
        #         .expand(N, T, self.L, self.R, dk)               # [N,T,L,R,dk]
        #         .permute(0, 2, 3, 1, 4)               # [N,L,R,T,dk]
        #         .contiguous()
        #         .view(N * S, T, dk)
        # )

        # # 2) prefix mean
        # E_flat = RACEPrefixMeanFlatFn.apply(probsK_flat, V_flat, 1e-6)  # [NS, T, dk]
        # # back to [N,T,L,R,dk]
        # E_pref = (
        #     E_flat.view(N, self.L, self.R, T, dk)
        #             .permute(0, 3, 1, 2, 4)
        #             .contiguous()
        # )
        # 3) lookup via one batched bmm
        out2 = torch.bmm(
            probsQ.view(N*T, 1, S),            # [N*T, 1, S]
            E_pref.contiguous().view(N*T, S, dk)            # [N*T, S, d_k]
        ).view(N, T, dk)                       # --> [N, T, d_k]

        # 4) reshape back --> [M, B, T, H, d_k]
        out = out2.view(M, B, H, T, dk).permute(0,1,3,2,4)
        return out

class RACEAttention(nn.Module):
    """Multi‑head wrapper around BatchedACE."""
    def __init__(self, d, h, K, L, M, drop=0.1,
                 qkv_bias=False, device='cpu'):
        super().__init__()
        assert d % h == 0
        self.h, self.d_k, self.M = h, d//h, M
        self.q = nn.Linear(d, d, bias=qkv_bias)
        self.k = nn.Linear(d, d, bias=qkv_bias)
        self.v = nn.Linear(d, d, bias=qkv_bias)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        self.ace = BatchedACE(
                       self.d_k, K, L, M, device=device
                   )

    def forward(self, x):
        B, T, _ = x.shape
        # split heads
        Q = self.q(x).view(B, T, self.h, self.d_k)
        K = self.k(x).view(B, T, self.h, self.d_k)
        V = self.v(x).view(B, T, self.h, self.d_k)
        # pack M ensembles
        pack = lambda z: z.unsqueeze(0).expand(self.M, -1, -1, -1, -1)
        Khf, Vhf, Qhf = pack(K), pack(V), pack(Q)
        # single-shot causal ACE
        out_h = self.ace(Khf, Vhf, Qhf)       # [M, B, T, H, d_k]
        # merge ensembles & heads
        out   = out_h.mean(0).permute(0,2,1,3).contiguous().view(B, T, -1)
        return self.drop(self.o(out))


class RACEBlock(nn.Module):
    def __init__(self, cfg, device='cpu'):
        super().__init__()
        self.att = RACEAttention(
                       d     = cfg["emb_dim"],
                       h     = cfg["n_heads"],
                       K     = cfg["K"],
                       L     = cfg["L"],
                       M     = cfg["M"],
                       drop  = cfg["drop_rate"],
                       qkv_bias    = cfg["qkv_bias"],
                       device      = device
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
        return self.drop(x) + h