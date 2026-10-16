#!/usr/bin/env python3
"""KL divergence between two --logprobs-out files of the same --force-list run: per position
KL(P_ref || P_x) = sum_v p_ref(v) (log p_ref(v) - log p_x(v)), top-1 agreement, and the forced token's nll in each.
Summaries overall and per text kind; 95% intervals by bootstrap over the 200 segments (10,000 resamples, seed 1).
Usage: kld_stats.py REF.bin X.bin META.json OUT.json"""
import json, sys
import numpy as np

ref_path, x_path, meta_path, out_path = sys.argv[1:5]
meta = json.load(open(meta_path))
V = None
REC = None


def header(f):
    magic, vocab = np.frombuffer(f.read(8), dtype=np.int32)
    assert magic == 0x31504C44
    return int(vocab)


fa, fx = open(ref_path, "rb"), open(x_path, "rb")
V = header(fa)
assert header(fx) == V
REC = 12 + 2 * V
seg, kl, top1, nll_a, nll_x = [], [], [], [], []
CH = 32   # rows per read: about 100 MB in flight (the engine holds most of the RAM meanwhile)
while True:
    ba, bx = fa.read(REC * CH), fx.read(REC * CH)
    if not ba:
        break
    assert len(ba) == len(bx)
    n = len(ba) // REC
    ra = np.frombuffer(ba, dtype=np.uint8).reshape(n, REC)
    rx = np.frombuffer(bx, dtype=np.uint8).reshape(n, REC)
    ha, hx = ra[:, :12].copy().view(np.int32), rx[:, :12].copy().view(np.int32)
    assert (ha == hx).all(), "the two runs scored different positions"
    la = ra[:, 12:].copy().view(np.float16).astype(np.float32)
    lx = rx[:, 12:].copy().view(np.float16).astype(np.float32)
    # renormalize after fp16 rounding (each row's probabilities sum to 1 again)
    la -= np.log(np.exp(la).sum(1, keepdims=True))
    lx -= np.log(np.exp(lx).sum(1, keepdims=True))
    pa = np.exp(la)
    kl.append((pa * (la - lx)).sum(1))
    top1.append(la.argmax(1) == lx.argmax(1))
    tgt = ha[:, 2]
    nll_a.append(-la[np.arange(n), tgt])
    nll_x.append(-lx[np.arange(n), tgt])
    seg.append(ha[:, 0])
seg, kl, top1 = np.concatenate(seg), np.concatenate(kl), np.concatenate(top1)
nll_a, nll_x = np.concatenate(nll_a), np.concatenate(nll_x)
kinds = np.array([meta[s]["kind"] for s in seg])
rng = np.random.default_rng(1)


def summary(mask):
    s_ids = np.unique(seg[mask])
    per = {k: [] for k in ("kl", "top1", "dnll")}
    for s in s_ids:
        m = mask & (seg == s)
        per["kl"].append(kl[m].mean()); per["top1"].append(top1[m].mean()); per["dnll"].append((nll_x[m] - nll_a[m]).mean())
    per = {k: np.array(v) for k, v in per.items()}
    boot = rng.integers(0, len(s_ids), size=(10000, len(s_ids)))
    ci = {k: [float(np.percentile(v[boot].mean(1), 2.5)), float(np.percentile(v[boot].mean(1), 97.5))]
          for k, v in per.items()}
    return {"segments": int(len(s_ids)), "positions": int(mask.sum()),
            "kl_mean": float(kl[mask].mean()), "kl_median": float(np.median(kl[mask])),
            "kl_p99": float(np.percentile(kl[mask], 99)), "kl_mean_ci95": ci["kl"],
            "top1_agreement": float(top1[mask].mean()), "top1_ci95": ci["top1"],
            "nll_ref": float(nll_a[mask].mean()), "nll_x": float(nll_x[mask].mean()),
            "dnll": float((nll_x[mask] - nll_a[mask]).mean()), "dnll_ci95": ci["dnll"]}


res = {"ref": ref_path, "x": x_path, "all": summary(np.ones(len(kl), bool))}
for k in ("code_py", "code_cpp", "zh", "en", "chat"):
    res[k] = summary(kinds == k)
json.dump(res, open(out_path, "w"), indent=1)
a = res["all"]
print(f"{x_path} vs {ref_path}: KL mean {a['kl_mean']:.5f} {a['kl_mean_ci95']} median {a['kl_median']:.5f} "
      f"p99 {a['kl_p99']:.4f} | top1 {a['top1_agreement']:.4f} {a['top1_ci95']} | dnll {a['dnll']:+.4f} {a['dnll_ci95']}")
