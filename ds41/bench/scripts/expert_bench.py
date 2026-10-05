"""Real DeepSeek V4.1 Flash EXL3 3bpw experts: CPU and GPU kernel speed, and CPU-vs-GPU parity.

Kernels come from exllamav3 (MIT). One full layer of 384 routed experts is loaded from the
checkpoint shards, so the working set (4.76 GiB) is far larger than the CPU caches.

Parts:
  cpu     CPU MoE kernel (exl3_moe_cpu_forward): GB/s of expert weights per thread count,
          tokens per call, and experts per call. Set EXL3_MOE_CPU_MAX_ISA to cap the ISA tier.
  cpubg   Same CPU kernel while another process streams pinned host memory to the GPU
          (RAM bandwidth contention, as in Strata's decode with pcie_frac > 0).
  gpu     EXL3 GEMV (decode rows) and reconstruct+GEMM (prefill rows) per projection.
  parity  CPU kernel output vs a GPU reference built from LinearEXL3 for the same experts.

Usage: python expert_bench.py --model-dir D --layer 10 --part cpu --out results/x [--tag avx2]
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import statistics
import time

import torch
from safetensors import safe_open

H, F, TOPK = 5120, 2304, 6
SWIGLU_LIMIT = 10.0


def load_layer(model_dir, layer):
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    prefix = f"layers.{layer}.ffn.experts."
    files = sorted({f for k, f in idx.items() if k.startswith(prefix)})
    t = {}
    for f in files:
        with safe_open(os.path.join(model_dir, f), "pt") as s:
            for k in s.keys():
                if k.startswith(prefix):
                    t[k] = s.get_tensor(k)
    n = 1 + max(int(k[len(prefix):].split(".")[0]) for k in t)

    def get(e, w, part):
        return t[f"{prefix}{e}.{w}.{part}"]

    expert_bytes = sum(get(0, w, p).numel() * get(0, w, p).element_size()
                       for w in ("w1", "w2", "w3") for p in ("trellis", "suh", "svh", "mul1"))
    return n, get, expert_bytes


def swizzle(tr):
    """Band-contiguous trellis layout, same transform as exllamav3 MoeArena.rehome."""
    tk, tn, ps = tr.shape
    if ps // 16 == 8:
        return tr.contiguous()
    d = torch.empty(tr.numel(), dtype=tr.dtype)
    d.view(tn // 8, tk, 8, ps).copy_(tr.view(tk, tn // 8, 8, ps).permute(1, 0, 2, 3))
    return d.view(tr.shape)


def make_cpu_layer(ext, n, get, swz):
    lists = []
    for w in ("w1", "w3", "w2"):          # gate, up, down
        tr = [swizzle(get(e, w, "trellis")) if swz else get(e, w, "trellis").contiguous() for e in range(n)]
        lists += [tr, [get(e, w, "suh").contiguous() for e in range(n)],
                  [get(e, w, "svh").contiguous() for e in range(n)]]
    return ext.exl3_moe_cpu_make_layer(*lists, [], [], [], 0, SWIGLU_LIMIT, 1 if swz else 0)


def pick(n, m, k, mode, rng):
    if mode == "shared":
        row = rng.sample(range(n), k)
        return [row[:] for _ in range(m)]
    ids = rng.sample(range(n), min(n, m * k))
    return [ids[i * k:(i + 1) * k] for i in range(m)]


def cpu_call(ext, handle, n, m, k, mode, threads, rng, reps=40):
    times, uniq = [], []
    x = torch.randn(m, H).half()
    w = torch.full((m, k), 1.5 / k).half()
    out = torch.empty(m, H, dtype=torch.float32)
    for r in range(reps + 5):
        sel = torch.tensor(pick(n, m, k, mode, rng), dtype=torch.int64)
        t0 = time.perf_counter()
        ext.exl3_moe_cpu_forward(handle, x, sel, w, out, threads)
        dt = time.perf_counter() - t0
        if r >= 5:
            times.append(dt)
            uniq.append(len(set(sel.flatten().tolist())))
    return statistics.median(times), statistics.mean(uniq)


def bg_h2d(stop, result):
    torch.cuda.init()
    h = torch.empty(1 << 30, dtype=torch.uint8, pin_memory=True)
    d = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    s = torch.cuda.Stream()
    moved, t0 = 0, time.perf_counter()
    with torch.cuda.stream(s):
        while not stop.is_set():
            d.copy_(h, non_blocking=True)
            s.synchronize()
            moved += h.numel()
    result.value = moved / (time.perf_counter() - t0) / 1e9


def part_cpu(a, ext, n, get, eb, bg=False):
    flags = {"avx2": ext.exl3_moe_cpu_has_avx2(), "avx512_bw": ext.exl3_moe_cpu_has_avx512_bw(),
             "avx512_vnni": ext.exl3_moe_cpu_has_avx512_vnni(),
             "avx512_vbmi": ext.exl3_moe_cpu_has_avx512_vbmi()}
    swz = flags["avx512_bw"] and os.environ.get("EXL3_MOE_CPU_SWIZZLE", "1") != "0"
    handle = make_cpu_layer(ext, n, get, swz)
    rng = random.Random(1234)
    ncpu = os.cpu_count()
    thread_list = sorted({max(1, ncpu // 4), ncpu // 2, ncpu})
    rows = []
    stop, res, proc = None, None, None
    if bg:
        ctx = mp.get_context("spawn")
        stop, res = ctx.Event(), ctx.Value("d", 0.0)
        proc = ctx.Process(target=bg_h2d, args=(stop, res))
        proc.start()
        time.sleep(3)
        thread_list = [ncpu // 2, ncpu]
    for th in thread_list:
        cpu_call(ext, handle, n, 1, TOPK, "distinct", th, rng, reps=10)   # warm the pool
        cases = [(1, k, "distinct") for k in (1, 2, 3, 6)] + \
                [(m, TOPK, "distinct") for m in (2, 4, 8)] + \
                [(m, TOPK, "shared") for m in (2, 4, 8)]
        if bg:
            cases = [(1, TOPK, "distinct"), (4, TOPK, "distinct")]
        for m, k, mode in cases:
            t, u = cpu_call(ext, handle, n, m, k, mode, th, rng)
            rows.append({"threads": th, "tokens": m, "experts_per_token": k, "mode": mode,
                         "unique_experts": round(u, 2), "ms": round(t * 1e3, 3),
                         "expert_gbps": round(u * eb / t / 1e9, 2),
                         "us_per_unique_expert": round(t / u * 1e6, 1)})
            print(rows[-1], flush=True)
    bg_gbps = None
    if bg:
        stop.set()
        proc.join()
        bg_gbps = round(res.value, 2)
    ext.exl3_moe_cpu_free_layer(handle)
    return {"isa_flags": flags, "isa_cap": os.environ.get("EXL3_MOE_CPU_MAX_ISA", "auto"),
            "swizzled": swz, "expert_bytes": eb, "num_experts": n, "rows": rows,
            "bg_h2d_gbps": bg_gbps}


def make_gpu_linears(e, get):
    from exllamav3.modules.quant.exl3 import LinearEXL3

    def lin(w, k, nout):
        return LinearEXL3(None, k, nout, suh=get(e, w, "suh").cuda(), svh=get(e, w, "svh").cuda(),
                          trellis=get(e, w, "trellis").cuda(), mul1=get(e, w, "mul1").cuda(),
                          key=f"e{e}.{w}")
    return lin("w1", H, F), lin("w3", H, F), lin("w2", F, H)


def cuda_time(fn, reps):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 1e3 / reps


def part_gpu(a, n, get, eb):
    g1, g3, g2 = make_gpu_linears(0, get)
    proj_bytes = eb // 3
    rows = []
    for m in (1, 2, 4, 6, 8, 16):
        for name, lin, k, nout in (("w1", g1, H, F), ("w2", g2, F, H)):
            x = torch.randn(m, k, device="cuda").half()
            t = cuda_time(lambda: lin.forward(x, {}), 200)
            rows.append({"path": "gemv", "proj": name, "rows": m, "us": round(t * 1e6, 2),
                         "weight_gbps": round(proj_bytes / t / 1e9, 1)})
    for m in (64, 256, 1024, 2048, 4096):
        for name, lin, k, nout in (("w1", g1, H, F), ("w2", g2, F, H)):
            x = torch.randn(m, k, device="cuda").half()
            t = cuda_time(lambda: lin.forward(x, {"reconstruct": True}), 20)
            rows.append({"path": "reconstruct_gemm", "proj": name, "rows": m,
                         "us": round(t * 1e6, 1), "tflops": round(2 * m * k * nout / t / 1e12, 2)})
    for r in rows:
        print(r, flush=True)

    # Whole expert, one token: what one GPU-resident expert costs in decode
    x = torch.randn(1, H, device="cuda").half()

    def expert():
        g = g1.forward(x, {}, torch.float).clamp_(max=SWIGLU_LIMIT)
        u = g3.forward(x, {}, torch.float).clamp_(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        return g2.forward((torch.nn.functional.silu(g) * u).half(), {}, torch.float)
    t = cuda_time(expert, 200)
    return {"expert_bytes": eb, "rows": rows,
            "one_expert_one_token_us": round(t * 1e6, 2),
            "one_expert_one_token_gbps": round(eb / t / 1e9, 1)}


def part_parity(a, ext, n, get, eb):
    rng = random.Random(7)
    m = 4
    sel = pick(n, m, TOPK, "distinct", rng)
    x = torch.randn(m, H).half()
    wts = torch.rand(m, TOPK)
    wts = (wts / wts.sum(1, keepdim=True) * 1.5).half()
    out = torch.empty(m, H, dtype=torch.float32)
    swz = ext.exl3_moe_cpu_has_avx512_bw()
    handle = make_cpu_layer(ext, n, get, swz)
    ext.exl3_moe_cpu_forward(handle, x, torch.tensor(sel), wts, out, os.cpu_count())
    ext.exl3_moe_cpu_free_layer(handle)

    ref_before = torch.zeros(m, H, device="cuda")   # DS order: route weight applied before w2
    ref_after = torch.zeros(m, H, device="cuda")    # weight applied after w2
    xc = x.cuda()
    for i in range(m):
        for j, e in enumerate(sel[i]):
            g1, g3, g2 = make_gpu_linears(e, get)
            xi = xc[i:i + 1].contiguous()
            g = g1.forward(xi, {}, torch.float).clamp(max=SWIGLU_LIMIT)
            u = g3.forward(xi, {}, torch.float).clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
            h = torch.nn.functional.silu(g) * u
            wv = wts[i, j].float().item()
            ref_before[i] += g2.forward((h * wv).half().contiguous(), {}, torch.float)[0]
            ref_after[i] += wv * g2.forward(h.half().contiguous(), {}, torch.float)[0]
    cpu = out.cuda()

    def cmp(ref):
        d = cpu - ref
        return {"rel_l2": float(d.norm() / ref.norm()), "max_abs": float(d.abs().max()),
                "ref_rms": float(ref.pow(2).mean().sqrt()),
                "cos": float(torch.nn.functional.cosine_similarity(cpu.flatten(), ref.flatten(), 0))}
    r = {"tokens": m, "selected": sel, "vs_ref_weight_before_w2": cmp(ref_before),
         "vs_ref_weight_after_w2": cmp(ref_after),
         "ref_before_vs_after_rel_l2": float((ref_before - ref_after).norm() / ref_before.norm())}
    print(json.dumps(r, indent=1))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--part", choices=["cpu", "cpubg", "gpu", "parity"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    from exllamav3.ext import exllamav3_ext as ext
    n, get, eb = load_layer(a.model_dir, a.layer)
    print(f"layer {a.layer}: {n} experts, {eb} bytes each", flush=True)
    if a.part == "cpu":
        r = part_cpu(a, ext, n, get, eb)
    elif a.part == "cpubg":
        r = part_cpu(a, ext, n, get, eb, bg=True)
    elif a.part == "gpu":
        r = part_gpu(a, n, get, eb)
    else:
        r = part_parity(a, ext, n, get, eb)
    r["layer"] = a.layer
    os.makedirs(a.out, exist_ok=True)
    name = f"expert_{a.part}{'_' + a.tag if a.tag else ''}.json"
    json.dump(r, open(os.path.join(a.out, name), "w"), indent=1)


if __name__ == "__main__":
    main()
