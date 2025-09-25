import argparse, time, sys, math, itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import contextlib
from torch.backends.cuda import sdp_kernel

# ---------------------- attention kernels (compatible shapes) ----------------------
def softmax_attention(Q, K, V, eps=1e-6):
    # Q,K,V: [B,H,T,D] -> [B,H,T,D]
    with sdp_kernel(
        enable_flash=True, 
        enable_math=False, 
        enable_mem_efficient=False
    ):
        out = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p = 0.0, 
            is_causal = False
        )

    return out

def angular_attention(Q, K, V, eps=1e-6, exponent=8.0):
    # Normalize along feature dim, cosine sim -> angular score in [0,1], sharpen, row-normalize.
    Qn = F.normalize(Q, dim=-1, eps=1e-6)
    Kn = F.normalize(K, dim=-1, eps=1e-6)
    cos_sim = (Qn @ Kn.transpose(-2,-1)).clamp(-0.999,0.999)          # [B,H,T,T]
    scores  = 1.0 - torch.acos(cos_sim)/torch.pi                       # [B,H,T,T]
    W = scores.clamp(min=1e-6).pow(exponent)                           # [B,H,T,T]
    W = W / (W.sum(-1, keepdim=True) + 1e-6)
    return W @ V                                                       # [B,H,T,D]

def linformer_attention(Q, K, V, k_proj_dim=128, eps=1e-6, Ek=None, Ev=None):
    # K,V compressed along sequence by learned matrices E_k,E_v: [T,k]
    B,H,T,D = Q.shape
    dk = D
    if Ek is None or Ev is None:
        Ek = torch.empty(T, k_proj_dim, device=Q.device, dtype=Q.dtype, requires_grad=True)
        Ev = torch.empty(T, k_proj_dim, device=Q.device, dtype=Q.dtype, requires_grad=True)
        nn.init.xavier_uniform_(Ek); nn.init.xavier_uniform_(Ev)
    K_proj = torch.einsum("bhtd,tk->bhkd", K, Ek)                      # [B,H,k,D]
    V_proj = torch.einsum("bhtd,tk->bhkd", V, Ev)                      # [B,H,k,D]
    scale  = 1.0 / math.sqrt(dk)
    scores = torch.einsum("bhtd,bhkd->bhtk", Q, K_proj) * scale        # [B,H,T,k]
    attn   = F.softmax(scores, dim=-1)
    return torch.einsum("bhtk,bhkd->bhtd", attn, V_proj)               # [B,H,T,D]

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

def race_attention(Q, K, V, ace_module: BatchedACE, eps=1e-6):
    # Wrap ACE to accept [B,H,T,D] like others.
    # Pack to [M,B,T,H,D], run, average M, then permute back to [B,H,T,D].
    B,H,T,D = Q.shape
    M = ace_module.M
    def pack(Z): return Z.permute(0,2,1,3).unsqueeze(0).expand(M, -1, -1, -1, -1)  # [M,B,T,H,D]
    out_hm = ace_module(pack(K), pack(V), pack(Q))                                  # [M,B,T,H,D]
    out = out_hm.mean(dim=0).permute(0,2,1,3)                                       # [B,H,T,D]
    return out

def linear_attention(Q, K, V, eps=1e-6):
    '''
    Linear attention with phi(x) = ELU(x) + 1. Non-causal global variant.
    Q,K,V: [B,H,T,D] -> [B,H,T,D]
    '''
    phiQ = F.elu(Q) + 1
    phiK = F.elu(K) + 1
    KV   = torch.einsum('bhtd,bhte->bhde', phiK, V)         # [B,H,D,D]
    Ksum = phiK.sum(dim=2)                                  # [B,H,D]
    num  = torch.einsum('bhtd,bhde->bhte', phiQ, KV)        # [B,H,T,D]
    den  = torch.einsum('bhtd,bhd->bht',   phiQ, Ksum).unsqueeze(-1) + eps
    return num / den


def favorplus_features(x, proj, eps=1e-6):
    """
    FAVOR+ positive random features for softmax kernel.
    x:    [B,H,T,D]
    proj: [H,M,D]  (one matrix per head)
    ->    [B,H,T,M]  (non-negative)
    """
    # x @ W^T  -> [B,H,T,M]
    xw = torch.einsum('bhtd,hmd->bhtm', x, proj)
    # stabilize across feature dimension
    xw = xw - xw.max(dim=-1, keepdim=True).values

    # exp( xW^T - ||x||^2/2 )
    exp_part = torch.exp(xw)
    x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)          # [B,H,T,1]
    base = torch.exp(-0.5 * x_norm_sq)
    return exp_part * base + eps

def linear_attention_favorplus(Q, K, V, proj, eps=1e-6):
    """
    Non-causal FAVOR+ linear attention (softmax kernel via positive random features).
    Q,K,V: [B,H,T,D]  ->  [B,H,T,D]
    proj:  [H,M,D]    (random feature projection per head)
    """
    B, H, T, D = Q.shape
    assert K.shape == (B, H, T, D) and V.shape[:3] == (B, H, T), "Shape mismatch"
    assert proj.shape[0] == H and proj.shape[2] == D, "proj must be [H,M,D] with matching H,D"

    # standard query scaling (like softmax attention)
    Q = Q / math.sqrt(D)

    # positive random features
    phiQ = favorplus_features(Q, proj, eps=eps)              # [B,H,T,M]
    phiK = favorplus_features(K, proj, eps=eps)              # [B,H,T,M]

    # global (non-causal) aggregation
    KV   = torch.einsum('bhtm,bhtd->bhmd', phiK, V)          # [B,H,M,D]
    Ksum = phiK.sum(dim=2)                                   # [B,H,M]

    num  = torch.einsum('bhtm,bhmd->bhtd', phiQ, KV)         # [B,H,T,D]
    den  = torch.einsum('bhtm,bhm->bht',   phiQ, Ksum).unsqueeze(-1) + eps
    return num / den

def orthogonal_random_matrix(M, D, device=None, dtype=None):
    """
    Build an MxD block-orthogonal matrix; rows are approximately N(0, I) after scaling.
    """
    q, _ = torch.linalg.qr(torch.randn(D, D, device=device, dtype=dtype), mode='reduced')
    mat = q.T  # rows orthonormal
    if M <= D:
        out = mat[:M]
    else:
        reps = [mat] * (M // D) + [mat[:(M % D)]]
        out = torch.cat(reps, dim=0)
    # scale so rows ~ N(0, I)
    return out * math.sqrt(D)

def make_favorplus_projections(H, M, D, device=None, dtype=None, orthogonal=True):
    """
    Create per-head projection matrix proj: [H,M,D].
    Use once (or re-draw occasionally during training).
    """
    if orthogonal:
        mats = [orthogonal_random_matrix(M, D, device=device, dtype=dtype) for _ in range(H)]
        proj = torch.stack(mats, dim=0)                      # [H,M,D]
    else:
        proj = torch.randn(H, M, D, device=device, dtype=dtype) * (1.0 / math.sqrt(D))
    # final per-row scaling to keep variance ~1/D
    return proj / math.sqrt(D)
# ---------------------- Benchmark harness ----------------------

def alloc_qkv(B,H,T,D,dtype,device):
    Q = torch.randn(B,H,T,D, dtype=dtype, device=device, requires_grad=True)
    K = torch.randn(B,H,T,D, dtype=dtype, device=device, requires_grad=True)
    V = torch.randn(B,H,T,D, dtype=dtype, device=device, requires_grad=True)
    return Q,K,V

def median_ms(vals):
    vals = sorted(vals)
    return vals[len(vals)//2] if vals else float('nan')

def is_cuda_device(dev: str) -> bool:
    return dev.startswith("cuda")

@contextlib.contextmanager
def maybe_autocast(device, dtype):
    if is_cuda_device(device) and dtype in (torch.float16, torch.bfloat16):
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        yield

def bench_one(method, T, B, H, D, device, dtype, warmup, iters, knobs):
    Q,K,V = alloc_qkv(B,H,T,D,dtype,device)
    state = {}

    # --- build per-method state ONCE (not inside the timed step) ---
    if method == 'linformer':
        k = knobs.get('linformer_k', 128)
        Ek = torch.empty(T, k, device=device, dtype=dtype, requires_grad=True)
        Ev = torch.empty(T, k, device=device, dtype=dtype, requires_grad=True)
        nn.init.xavier_uniform_(Ek); nn.init.xavier_uniform_(Ev)
        state['Ek'], state['Ev'] = Ek, Ev
    elif method == 'race':
        L  = knobs.get('race_L', 1)
        Kb = knobs.get('race_Kbits', 1)
        M  = knobs.get('race_M', 1)
        state['ace'] = BatchedACE(D, Kb, L, M, device=device)
    elif method == 'favor':
        # Build FAVOR+ projections once
        m = 256
        state['proj'] = make_favorplus_projections(H, m, D, device=device, dtype=dtype)

    def step():
        with maybe_autocast(device, dtype):
            if method == 'softmax':
                out = softmax_attention(Q,K,V)
            elif method == 'linear':
                out = linear_attention(Q,K,V)
            elif method == 'angular':
                out = angular_attention(Q,K,V, exponent=knobs.get('angular_exp', 8.0))
            elif method == 'linformer':
                out = linformer_attention(Q,K,V, k_proj_dim=knobs.get('linformer_k', 128),
                                          Ek=state['Ek'], Ev=state['Ev'])
            elif method == 'race':
                out = race_attention(Q,K,V, ace_module=state['ace'])
            elif method == 'favor':
                out = linear_attention_favorplus(Q,K,V, proj=state['proj'])
            else:
                raise ValueError(method)
            loss = out.float().sum()
        loss.backward()
        return out

    # Warmup
    for _ in range(warmup):
        step()
        if is_cuda_device(device): torch.cuda.synchronize()
        Q.grad = K.grad = V.grad = None
        for t in state.values():
            if isinstance(t, torch.Tensor) and t.grad is not None:
                t.grad = None

    # Timed
    times = []
    for _ in range(iters):
        if is_cuda_device(device): torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        if is_cuda_device(device): torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0)*1e3)

        Q.grad = K.grad = V.grad = None
        for t in state.values():
            if isinstance(t, torch.Tensor) and t.grad is not None:
                t.grad = None
    return median_ms(times)

def main():
    p = argparse.ArgumentParser(description='Kernel-only benchmark for 5 attention methods (synthetic Q/K/V)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype',  default='bf16', choices=['bf16', 'fp16', 'fp32']) # change to fp32 for favor
    p.add_argument('--B', type=int, default=1)
    p.add_argument('--H', type=int, default=4)
    p.add_argument('--D', type=int, default=128, help='per-head dim for Q/K/V')
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--iters',  type=int, default=5)
    p.add_argument('--exponents', type=int, nargs='+',
                   default=[11, 13, 15, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
                   help='Sequence length exponents: T = 2^e')
    p.add_argument('--methods', nargs='+',
                   default=['race', 'linear', 'linformer'],
                   choices=['race', 'linear', 'linformer'],)
    p.add_argument('--angular_exp', type=float, default=6.0)
    p.add_argument('--linformer_k', type=int, default=128)
    p.add_argument('--race_L', type=int, default=1)
    p.add_argument('--race_Kbits', type=int, default=1)
    p.add_argument('--race_M', type=int, default=1)
    p.add_argument('--out', default='five_attention_time_vs_length.png')
    p.add_argument('--csv', default='five_attention_time_vs_length.csv')
    args = p.parse_args()

    device = args.device
    if args.dtype == 'auto':
        dtype = torch.float32
    elif args.dtype == 'fp32': dtype = torch.float32
    elif args.dtype == 'fp16': dtype = torch.float16
    elif args.dtype == 'bf16': dtype = torch.bfloat16

    print(f'device={device} dtype={dtype} B={args.B} H={args.H} D={args.D} methods={args.methods}')

    lengths = [1 << e for e in args.exponents]
    results = {m: [] for m in args.methods}

    knobs = dict(angular_exp=args.angular_exp,
                 linformer_k=args.linformer_k,
                 race_L=args.race_L, race_Kbits=args.race_Kbits, race_M=args.race_M)

    for T in lengths:
        print(f'\n=== T = 2^{int(math.log2(T))} ({T}) ===')
        for m in args.methods:
            try:
                t_ms = bench_one(m, T, args.B, args.H, args.D, device, dtype,
                                 warmup=args.warmup, iters=args.iters, knobs=knobs)
                results[m].append((T, t_ms))
                print(f'  {m:9s}  median fwd time: {t_ms:8.2f} ms')
            except RuntimeError as err:
                msg = str(err).lower()
                print(f'  {m:9s}  -> ERROR: {msg}', file=sys.stderr)
                if 'out of memory' in msg:
                    results[m].append((T, float("nan")))
                else:
                    raise

    # Save CSV
    import csv
    with open(args.csv, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['Sequence Length'] + args.methods
        w.writerow(header)
        for T in lengths:
            row = [T]
            for m in args.methods:
                ts_map = dict(results[m])
                row.append(ts_map.get(T, float('nan')))
            w.writerow(row)
    print(f'Wrote {args.csv}')

    # Plot log–log
    plt.figure(figsize=(9.5,5.5))
    for m in args.methods:
        ts_map = dict(results[m])
        ys = [ts_map.get(T, float('nan')) for T in lengths]
        plt.plot(lengths, ys, marker='o', label=m)
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)  [forward + backward]')
    plt.title('Kernel-only attention timing vs sequence length')
    plt.grid(True, which='both', linestyle='--', linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    print(f'Wrote {args.out}')

if __name__ == '__main__':
    main()
