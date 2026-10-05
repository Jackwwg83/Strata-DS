"""CPU tests for proto/torch_kernels.py and for the official model.py running on them.

Run: python -m pytest proto/tests -q
"""
import math
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "ref"))

import torch_kernels as tk  # noqa: E402

sys.modules["kernel"] = tk      # model.py imports `kernel`; give it the PyTorch kernels
import model as M  # noqa: E402

E2M1_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]   # index = encoding; even index = even mantissa


def nearest_even_e2m1(v: float) -> float:
    a = abs(v)
    best = min(range(8), key=lambda i: (abs(E2M1_GRID[i] - a), i % 2))
    return math.copysign(E2M1_GRID[best], v)


def test_e2m1_rounding_matches_bruteforce_including_ties():
    vals = [i / 64 for i in range(-6 * 64, 6 * 64 + 1)] + [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    t = torch.tensor(vals)
    got = tk.round_to_e2m1(t).tolist()
    want = [nearest_even_e2m1(v) for v in vals]
    assert got == want


def test_pow2_scale_is_ceil_log2():
    a = torch.cat([torch.rand(1000) * 10 + 1e-6, torch.tensor([0.25, 0.5, 1.0, 2.0, 3.0, 1e-3])])
    got = tk.round_pow2_scale(a)
    want = torch.tensor([2.0 ** math.ceil(math.log2(float(x))) for x in a])
    assert torch.equal(got, want)


def test_act_quant_inplace_is_idempotent_and_close():
    torch.manual_seed(0)
    x = (torch.randn(8, 256) * 5).bfloat16()
    y = tk.act_quant(x.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True)
    z = tk.act_quant(y.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True)
    assert torch.equal(y, z)
    rel = (y.float() - x.float()).norm() / x.float().norm()
    assert rel < 0.05


def test_act_quant_scales_are_powers_of_two_and_cover_amax():
    x = (torch.randn(4, 64) * 3).bfloat16()
    q, s = tk.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    sf = s.float()
    assert torch.equal(sf, tk.round_pow2_scale(sf))
    amax = x.float().unflatten(-1, (2, 32)).abs().amax(-1)
    assert torch.all(amax / sf <= 448.0)


def test_fp4_quant_both_scale_kinds():
    torch.manual_seed(1)
    x = torch.randn(16, 512).bfloat16()
    for block, sd in ((16, torch.float8_e4m3fn), (32, torch.float8_e8m0fnu)):
        y = tk.fp4_act_quant(x.clone(), block, True, sd)
        rel = (y.float() - x.float()).norm() / x.float().norm()
        assert 0 < rel < 0.2
        # every value divided by its block scale lands on the E2M1 grid
        assert torch.equal(tk.fp4_act_quant(y.clone(), block, True, sd), y)


def test_fp8_gemm_matches_dequantized_reference():
    torch.manual_seed(2)
    a = torch.randn(5, 128)
    b = torch.randn(96, 128) * 0.1
    aq, as_ = tk.act_quant(a.bfloat16(), 32, "ue8m0", torch.float8_e8m0fnu)
    bs = torch.full((3, 4), 2.0 ** -3).to(torch.float8_e8m0fnu)
    bq = (b / 2.0 ** -3).clamp(-448, 448).to(torch.float8_e4m3fn)
    ref = (aq.float().view(5, 4, 32) * as_.float().view(5, 4, 1)).view(5, 128) @ (bq.float() * 2.0 ** -3).t()
    got = tk.fp8_gemm(aq, as_, bq, bs, torch.float8_e8m0fnu, 32).float()
    assert (got - ref).norm() / ref.norm() < 1e-2


def test_sparse_attn_matches_naive_softmax_with_sink():
    torch.manual_seed(3)
    b, m, h, d, n, k = 1, 5, 4, 64, 20, 7
    q = torch.randn(b, m, h, d).bfloat16()
    kv = torch.randn(b, n, d).bfloat16()
    idx = torch.randint(-1, n, (b, m, k)).int()
    idx[0, 0] = -1                                   # a query with nothing to attend to -> zeros
    sink = torch.randn(h)
    got = tk.sparse_attn(q, kv, sink, idx, d ** -0.5).float()
    for i in range(m):
        sel = [int(j) for j in idx[0, i] if j >= 0]
        if not sel:
            assert torch.all(got[0, i] == 0)
            continue
        kk = kv[0, sel].float()
        s = (q[0, i].float() @ kk.t()) * d ** -0.5               # [h, len]
        logits = torch.cat([s, sink.unsqueeze(1)], dim=1)
        p = logits.softmax(-1)[:, :-1]
        ref = p @ kk
        assert (got[0, i] - ref).norm() / ref.norm() < 2e-2


def test_sinkhorn_comb_is_doubly_stochastic():
    mixes = torch.randn(2, 3, 24)
    pre, post, comb = tk.hc_split_sinkhorn(mixes, torch.tensor([1.0, 1.0, 1.0]), torch.zeros(24))
    assert torch.allclose(comb.sum(-1), torch.ones(2, 3, 4), atol=1e-3)
    assert torch.allclose(comb.sum(-2), torch.ones(2, 3, 4), atol=1e-3)
    assert torch.all((post > 0) & (post < 2)) and torch.all(pre > 0)


def _tiny_model():
    torch.manual_seed(4)
    torch.set_default_dtype(torch.bfloat16)
    args = M.ModelArgs(max_batch_size=1, max_seq_len=64, expert_dtype=None, dspark_block_size=0,
                       swiglu_limit=10.0, route_scale=1.5)
    model = M.Transformer(args)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dtype == torch.float8_e8m0fnu:
                p.copy_(torch.full(p.shape, 2.0 ** -6).to(p.dtype))
            elif p.dtype == torch.float8_e4m3fn:
                p.copy_((torch.randn(p.shape) * 8).clamp(-448, 448).to(p.dtype))
            elif name.endswith("norm.weight"):
                p.fill_(1.0)
            elif ".hc_" in name and name.endswith("_scale"):
                p.fill_(0.1)
            else:
                p.copy_((torch.randn(p.shape) * 0.02).to(p.dtype))
    return args, model


def test_official_model_runs_on_torch_kernels_and_decode_matches_prefill():
    """Prefill 33 tokens in one pass vs prefill 32 then decode 1: the last logits must agree."""
    args, model = _tiny_model()
    tokens = torch.randint(0, args.vocab_size, (1, 33))
    _, logits_full, _ = model(tokens, 0)
    _, _, _ = model(tokens[:, :32], 0)
    _, logits_step, _ = model(tokens[:, 32:33], 32)
    assert torch.isfinite(logits_full).all()
    rel = (logits_step.float() - logits_full.float()).norm() / logits_full.float().norm()
    assert rel < 0.05, rel
