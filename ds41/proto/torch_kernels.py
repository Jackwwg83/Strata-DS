"""PyTorch equivalents of the TileLang kernels in ref/kernel.py (DeepSeek V4.1 Flash inference).

Same function names and signatures, so this module can stand in for `kernel` when TileLang is not
available (older GPUs, CPU tests). Each function reproduces the kernel's arithmetic: block scales,
power-of-two scale rounding, clamps, and the casts to FP8 / FP4. Accumulation order inside the
GEMMs and the attention differs from the tiled kernels, so results match to rounding, not bit for
bit. tests/test_torch_kernels.py checks the rounding rules on CPU; on a GPU with TileLang,
`compare_with_tilelang()` measures the difference against the real kernels.
"""
import os

import torch

FP8_MAX = 448.0
# DS41_TK_FP32_GEMM=1 accumulates the FP8 GEMM in fp32 instead of a bf16 GEMM: same math, another summation
# order. Used to measure how far two equally valid implementations drift apart (the engine comparison baseline).
FP32_GEMM = os.environ.get("DS41_TK_FP32_GEMM") == "1"
FP4_MAX = 6.0


def round_pow2_scale(a: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(a)) for positive normal fp32 values (kernel.py fast_round_scale)."""
    m, e = torch.frexp(a)
    e = torch.where(m == 0.5, e - 1, e)       # exact powers of two keep their exponent
    return torch.ldexp(torch.ones_like(a), e)


def round_to_e2m1(v: torch.Tensor) -> torch.Tensor:
    """Round fp32 values already clamped to [-6, 6] onto the FP4 E2M1 grid, ties to even."""
    a = v.abs()
    q = torch.full_like(a, 6.0)
    q = torch.where(a <= 5.0, torch.full_like(a, 4.0), q)
    q = torch.where(a < 3.5, torch.full_like(a, 3.0), q)
    q = torch.where(a <= 2.5, torch.full_like(a, 2.0), q)
    q = torch.where(a < 1.75, torch.full_like(a, 1.5), q)
    q = torch.where(a <= 1.25, torch.full_like(a, 1.0), q)
    q = torch.where(a < 0.75, torch.full_like(a, 0.5), q)
    q = torch.where(a <= 0.25, torch.zeros_like(a), q)
    return torch.copysign(q, v)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """Block-wise FP8 E4M3 quantization; inplace=True writes quantize-dequantize back into x."""
    n = x.size(-1)
    assert n % block_size == 0
    xf = x.float().unflatten(-1, (n // block_size, block_size))
    amax = xf.abs().amax(-1).clamp_min(1e-4)
    s = amax * (1.0 / FP8_MAX)
    if scale_fmt is not None:
        s = round_pow2_scale(s)
    q = (xf / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((q.float() * s.unsqueeze(-1)).flatten(-2).to(x.dtype))
        return x
    return q.flatten(-2), s.to(scale_dtype)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """Block-wise FP4 E2M1 with E8M0 (power-of-two) or E4M3 scales."""
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    assert inplace, "only the in-place (quantize-dequantize) form is used by model.py"
    n = x.size(-1)
    assert n % block_size == 0
    xf = x.float().unflatten(-1, (n // block_size, block_size))
    amax = xf.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(FP4_MAX * 2.0 ** -9)
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(FP4_MAX * 2.0 ** -126)
        s = round_pow2_scale(amax * (1.0 / FP4_MAX))
    q = round_to_e2m1((xf / s.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX))
    x.copy_((q * s.unsqueeze(-1)).flatten(-2).to(x.dtype))
    return x


def _expand_block_scale(scale: torch.Tensor, rows: int, cols: int, block: int) -> torch.Tensor:
    return scale.float().repeat_interleave(block, 0)[:rows].repeat_interleave(block, 1)[:, :cols]


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    """C = A @ B^T with per-block FP8 scaling. FP8 values times power-of-two scales are exact in
    bf16, so the product is a bf16 GEMM with fp32 accumulation."""
    k = a.size(-1)
    m = a.numel() // k
    n = b.size(0)
    a_deq = (a.float().view(m, k // block_size, block_size) * a_s.float().view(m, -1, 1)).view(m, k)
    b_deq = b.float() * _expand_block_scale(b_s, n, k, block_size)
    if FP32_GEMM:
        c = (a_deq @ b_deq.t()).to(torch.bfloat16)
    else:
        c = a_deq.to(torch.bfloat16) @ b_deq.to(torch.bfloat16).t()
    return c.view(*a.shape[:-1], n).to(torch.get_default_dtype())


def fp4_gemm(*args, **kwargs):
    raise NotImplementedError("routed experts run through EXL3 in this prototype; fp4_gemm is unused")


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale, chunk=256):
    """Attention of each query over its own gathered KV positions (index -1 = empty), with a
    per-head sink logit added to the softmax denominator only."""
    b, m, h, d = q.shape
    out = torch.empty_like(q)
    sink = attn_sink.float()
    for bi in range(b):
        for m0 in range(0, m, chunk):
            m1 = min(m, m0 + chunk)
            idx = topk_idxs[bi, m0:m1].long()
            valid = idx >= 0
            g = kv[bi][idx.clamp_min(0)] * valid.unsqueeze(-1)              # [c, k, d] bf16, empty -> 0
            s = torch.einsum("chd,ckd->chk", q[bi, m0:m1].float(), g.float()) * softmax_scale
            s = s.masked_fill(~valid.unsqueeze(1), float("-inf"))
            mx = s.amax(-1).clamp_min(-1e30)                                 # kernel starts at -1e30
            p = torch.exp(s - mx.unsqueeze(-1))
            denom = p.sum(-1) + torch.exp(sink.unsqueeze(0) - mx)
            o = torch.einsum("chk,ckd->chd", p.to(torch.bfloat16).float(), g.float())
            out[bi, m0:m1] = (o / denom.unsqueeze(-1)).to(q.dtype)
    return out


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    x = mixes.float()
    pre = torch.sigmoid(x[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(x[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (x[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def compare_with_tilelang(seed: int = 0) -> dict:
    """Run every kernel here and in ref/kernel.py on the same random inputs (GPU + TileLang)."""
    import kernel as tl  # ref/kernel.py, needs tilelang
    torch.set_default_dtype(torch.bfloat16)      # the kernels write C in the default dtype
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = {}

    def rel(a, b):
        a, b = a.float(), b.float()
        return float((a - b).norm() / b.norm().clamp_min(1e-30))

    def case(name, fn):
        try:
            r[name] = fn()
        except Exception as ex:          # e.g. a kernel that needs more shared memory than this GPU has
            r[name] = f"tilelang failed: {type(ex).__name__}: {str(ex)[:120]}"

    x = torch.randn(64, 5120, device="cuda", generator=g).bfloat16() * 3
    y = torch.randn(64, 512, device="cuda", generator=g).bfloat16()
    case("act_quant_inplace", lambda: rel(act_quant(x.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True),
                                          tl.act_quant(x.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True)))
    case("fp4_e4m3_g16", lambda: rel(fp4_act_quant(y.clone(), 16, True, torch.float8_e4m3fn),
                                     tl.fp4_act_quant(y.clone(), 16, True, torch.float8_e4m3fn)))
    case("fp4_e8m0_g32", lambda: rel(fp4_act_quant(y.clone(), 32, True), tl.fp4_act_quant(y.clone(), 32, True)))
    w = (torch.randn(2304, 5120, device="cuda", generator=g, dtype=torch.float32) * 0.02).to(torch.float8_e4m3fn)
    ws = torch.full((72, 160), 1.0, device="cuda", dtype=torch.float32).to(torch.float8_e8m0fnu)
    xa, xs = act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    case("fp8_gemm", lambda: rel(fp8_gemm(xa, xs, w, ws, torch.float8_e8m0fnu, 32),
                                 tl.fp8_gemm(xa, xs, w, ws, torch.float8_e8m0fnu, 32)))
    q = torch.randn(1, 128, 64, 512, device="cuda", generator=g).bfloat16()
    kvt = torch.randn(1, 700, 512, device="cuda", generator=g).bfloat16()
    idx = torch.randint(-1, 700, (1, 128, 640), device="cuda", generator=g).int()
    sink = torch.randn(64, device="cuda", generator=g, dtype=torch.float32)
    case("sparse_attn", lambda: rel(sparse_attn(q, kvt, sink, idx, 512 ** -0.5),
                                    tl.sparse_attn(q, kvt, sink, idx, 512 ** -0.5)))
    mixes = torch.randn(1, 64, 24, device="cuda", generator=g, dtype=torch.float32)
    sc = torch.rand(3, device="cuda", dtype=torch.float32) + 0.5
    base = torch.randn(24, device="cuda", dtype=torch.float32)
    case("hc_sinkhorn", lambda: max(rel(u, v) for u, v in zip(hc_split_sinkhorn(mixes, sc, base),
                                                              tl.hc_split_sinkhorn(mixes, sc, base))))
    return r
