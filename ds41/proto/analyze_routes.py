"""Turn recorded routing (proto/ds41_proto.py output) into cache and speed numbers.

Outputs <run>/analysis.json and prints a summary. Every result names what it is computed from.
  1. skew: share of expert accesses that go to the hottest x% of (layer, expert) pairs
  2. static profile hit rate vs cache size, leave-one-doc-out (Strata paper, figure 3 method)
  3. tiers at 128 GB RAM: share of accesses served by VRAM / RAM / SSD per bitrate and GPU size
  4. adaptive VRAM cache (decayed counts, periodic swaps, as Strata does) vs static, per token
  5. distinct experts per layer inside a window of k consecutive tokens (speculative verify cost)
  6. decode tok/s estimate from session-1 measured constants (labelled as an estimate)

Usage: python analyze_routes.py <run_dir> [--measured bench/results/2026-10-05-4090-7950x]
"""
import argparse
import glob
import json
import os

import numpy as np

L, E, K = 40, 384, 6
GIB = 2 ** 30
EXPERT_BYTES = {3.0: 13_315_596, 2.5: 11_103_756, 2.0: 8_891_916, 1.59: 101.4 * GIB / (L * E)}
# Budgets for 128 GB RAM: host keeps OS/runtime/engram cache/staging (about 24 GiB) out of the arena
RAM_ARENA_GIB = 128 * 1e9 / GIB - 24
VRAM_EXPERT_GIB = {16: 3.3, 24: 10.8}        # after ~8 GiB dense, KV, workspace, runtime (kit budget)


def load_runs(run_dir):
    seqs = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "routes", "doc_*.npz"))):
        seqs[os.path.basename(p)[4:-4]] = ("prefill", np.load(p)["routes"].astype(np.int64))
    for p in sorted(glob.glob(os.path.join(run_dir, "routes", "gen_*.npz"))):
        z = np.load(p)
        if len(z["decode_routes"]):
            seqs["gen_" + os.path.basename(p)[4:-4]] = ("decode", z["decode_routes"].astype(np.int64))
    return seqs


def pair_counts(routes):
    """routes [T, L, K] -> counts [L*E] of (layer, expert) accesses."""
    flat = (np.arange(L)[None, :, None] * E + routes).reshape(-1)
    return np.bincount(flat, minlength=L * E)


def skew(counts):
    c = np.sort(counts)[::-1].astype(np.float64)
    cum = np.cumsum(c) / c.sum()
    return {f"top_{int(x * 100)}pct_pairs_share": float(cum[int(len(c) * x) - 1]) for x in (0.01, 0.05, 0.1, 0.2, 0.5)}


def hit_rate_curve(seqs, sizes):
    """Leave-one-sequence-out: rank pairs by counts of the other sequences, cache the top n."""
    names = list(seqs)
    per = {n: pair_counts(seqs[n][1]) for n in names}
    total = sum(per.values())
    curve = {s: [] for s in sizes}
    for n in names:
        prof = total - per[n]
        order = np.argsort(-prof, kind="stable")
        held = per[n][order]
        cum = np.concatenate([[0], np.cumsum(held)]) / held.sum()
        for s in sizes:
            curve[s].append(cum[min(s, len(cum) - 1)])
    return {s: float(np.mean(v)) for s, v in curve.items()}


def tier_shares(seqs):
    """For each bitrate and GPU size: share of accesses served from VRAM, RAM and SSD when the
    hottest pairs (leave-one-out profile) go to VRAM first, then RAM, the rest to the SSD."""
    names = list(seqs)
    per = {n: pair_counts(seqs[n][1]) for n in names}
    total = sum(per.values())
    out = {}
    for bpw, eb in EXPERT_BYTES.items():
        for gpu, vgib in VRAM_EXPERT_GIB.items():
            nv = int(vgib * GIB // eb)
            nr = int(RAM_ARENA_GIB * GIB // eb)
            v = r = s = 0.0
            for n in names:
                order = np.argsort(-(total - per[n]), kind="stable")
                held = per[n][order].astype(np.float64)
                t = held.sum()
                v += held[:nv].sum() / t
                r += held[nv:nv + nr].sum() / t
                s += held[nv + nr:].sum() / t
            k = len(names)
            out[f"{bpw}bpw_{gpu}GB"] = {"vram_experts": nv, "ram_experts": min(nr, L * E - nv),
                                       "vram_share": v / k, "ram_share": r / k, "ssd_share": s / k}
    return out


def adaptive_hit(seq, cap, profile, every=4, decay=0.7, max_swaps=96, ratio=1.5):
    """Strata-style adaptive VRAM cache on one token sequence (routes [T, L, K]).
    Start from the profile's top `cap` pairs; count accesses; every `every` tokens decay the counts
    and swap in up to `max_swaps` missing pairs whose count beats the coldest resident by `ratio`."""
    resident = np.zeros(L * E, bool)
    resident[np.argsort(-profile, kind="stable")[:cap]] = True
    cnt = np.zeros(L * E)
    hits = tot = 0
    for t in range(seq.shape[0]):
        pairs = (np.arange(L)[:, None] * E + seq[t]).reshape(-1)
        hits += resident[pairs].sum()
        tot += len(pairs)
        np.add.at(cnt, pairs, 1)
        if (t + 1) % every == 0:
            cnt *= decay
            cand = np.where(~resident & (cnt >= 2))[0]
            if len(cand):
                cand = cand[np.argsort(-cnt[cand])][:max_swaps]
                res_idx = np.where(resident)[0]
                cold = res_idx[np.argsort(cnt[res_idx])][:len(cand)]
                for c, o in zip(cand, cold):
                    if cnt[c] > ratio * cnt[o] + 1e-9:
                        resident[c], resident[o] = True, False
    return hits / max(tot, 1)


def window_distinct(seqs, ks=(1, 2, 3, 4, 6, 8)):
    """Mean distinct experts per layer within windows of k consecutive tokens, over 6*k."""
    res = {}
    for k in ks:
        ratios = []
        for kind, r in seqs.values():
            for t0 in range(0, r.shape[0] - k + 1, k):
                w = r[t0:t0 + k]                                  # [k, L, K]
                d = [len(np.unique(w[:, l])) for l in range(L)]
                ratios.append(np.mean(d) / (K * k))
        res[k] = float(np.mean(ratios))
    return res


def decode_estimate(tiers, measured):
    """tok/s per forward pass (no speculation): GPU dense + GPU experts run beside CPU experts;
    SSD reads add on top. Constants are session-1 measurements; the combination is a model."""
    cpu_gbps = measured.get("cpu_expert_gbps", 44.0)
    ssd_gbps = measured.get("nvme_13m_gbps", 6.5)
    gpu_dense_ms = measured.get("gpu_dense_ms", 8.9)
    gpu_expert_us = measured.get("gpu_expert_us", 40.0)
    out = {}
    for key, t in tiers.items():
        bpw = float(key.split("bpw")[0])
        eb = EXPERT_BYTES[bpw]
        acc = L * K
        cpu_ms = acc * t["ram_share"] * eb / (cpu_gbps * 1e9) * 1e3
        gpu_ms = gpu_dense_ms + acc * t["vram_share"] * gpu_expert_us / 1e3
        ssd_ms = acc * t["ssd_share"] * eb / (ssd_gbps * 1e9) * 1e3
        ms = max(cpu_ms, gpu_ms) + ssd_ms
        out[key] = {"cpu_ms": round(cpu_ms, 1), "gpu_ms": round(gpu_ms, 1), "ssd_ms": round(ssd_ms, 1),
                    "tok_s": round(1000 / ms, 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--measured", default="")
    a = ap.parse_args()
    seqs = load_runs(a.run_dir)
    allr = np.concatenate([r for _, r in seqs.values()])
    out = {"sequences": {n: {"kind": k, "tokens": int(r.shape[0])} for n, (k, r) in seqs.items()},
           "tokens_total": int(allr.shape[0])}
    counts = pair_counts(allr)
    out["skew_all"] = skew(counts)
    out["skew_per_layer_top10pct_median"] = float(np.median(
        [skew(counts[l * E:(l + 1) * E])["top_10pct_pairs_share"] for l in range(L)]))
    out["pairs_never_used"] = int((counts == 0).sum())
    sizes = [0, 100, 250, 500, 850, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 15360]
    out["static_hit_curve"] = hit_rate_curve(seqs, sizes)
    out["tiers_128gb"] = tier_shares(seqs)

    total = sum(pair_counts(r) for _, r in seqs.values())
    adapt = {}
    for cap in (260, 850):
        a_h, s_h = [], []
        for n, (kind, r) in seqs.items():
            prof = total - pair_counts(r)
            a_h.append(adaptive_hit(r, cap, prof))
            order = np.argsort(-prof, kind="stable")[:cap]
            res = np.zeros(L * E, bool)
            res[order] = True
            pairs = (np.arange(L)[None, :, None] * E + r).reshape(-1)
            s_h.append(res[pairs].mean())
        adapt[cap] = {"static": float(np.mean(s_h)), "adaptive": float(np.mean(a_h))}
    out["adaptive_vs_static"] = adapt
    out["window_distinct_ratio"] = window_distinct(seqs)

    measured = {}
    if a.measured:
        try:
            m = a.measured
            cpu = json.load(open(os.path.join(m, "expert_cpu_auto.json")))["rows"]
            measured["cpu_expert_gbps"] = max(r["expert_gbps"] for r in cpu if r["mode"] == "distinct")
            measured["nvme_13m_gbps"] = json.load(open(os.path.join(m, "nvme_summary.json")))["rand_13m_qd4"]["gbps"]
            g = json.load(open(os.path.join(m, "gpu_pcie.json")))
            measured["gpu_dense_ms"] = 8.5e9 / (g["gpu_read_sum_gbps"] * 1e9) * 1e3
            measured["gpu_expert_us"] = json.load(open(os.path.join(m, "expert_gpu.json")))["one_expert_one_token_us"]
        except Exception as ex:
            measured["error"] = str(ex)
    out["measured_constants"] = measured
    out["decode_estimate_no_spec"] = decode_estimate(out["tiers_128gb"], measured)

    # Same tiers, but VRAM runs Strata's adaptive cache. SSD keeps the statically coldest pairs, so
    # the SSD share is unchanged; the adaptive VRAM hits come out of the RAM share.
    adaptive_tiers = {}
    for key, t in out["tiers_128gb"].items():
        cap = t["vram_experts"]
        hits = []
        for n, (kind, r) in seqs.items():
            hits.append(adaptive_hit(r, cap, total - pair_counts(r)))
        v = float(np.mean(hits))
        adaptive_tiers[key] = dict(t, vram_share=v, ram_share=max(0.0, 1 - v - t["ssd_share"]))
    out["tiers_128gb_adaptive"] = adaptive_tiers
    out["decode_estimate_no_spec_adaptive"] = decode_estimate(adaptive_tiers, measured)
    json.dump(out, open(os.path.join(a.run_dir, "analysis.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
