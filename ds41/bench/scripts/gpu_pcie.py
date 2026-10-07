"""PCIe H2D/D2H bandwidth, GPU memory bandwidth, and pinned host memory limits.

Usage: python gpu_pcie.py <out_dir> [--pin-max-gib N]
Writes <out_dir>/gpu_pcie.json.
"""
import argparse
import json
import os
import time

import torch

EXPERT_SLOT = 13_316_352          # one 3bpw DS V4.1 expert, 256 B aligned components
MB = 1 << 20


def timed_copy(dst, src, reps):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    best = 1e30
    for _ in range(reps):
        s.record()
        dst.copy_(src, non_blocking=True)
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / 1e3)
    return src.numel() * src.element_size() / best / 1e9


def pcie(results):
    dev = torch.device("cuda")
    rows = []
    for size in [1 * MB, EXPERT_SLOT, 64 * MB, 256 * MB, 1024 * MB]:
        h = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        d = torch.empty(size, dtype=torch.uint8, device=dev)
        rows.append({"bytes": size,
                     "h2d_pinned_gbps": round(timed_copy(d, h, 20), 2),
                     "d2h_pinned_gbps": round(timed_copy(h, d, 20), 2)})
        del h, d
    results["pcie_by_size"] = rows

    # Pageable source, 256 MiB
    hp = torch.empty(256 * MB, dtype=torch.uint8)
    d = torch.empty(256 * MB, dtype=torch.uint8, device=dev)
    results["h2d_pageable_256m_gbps"] = round(timed_copy(d, hp, 10), 2)
    del hp, d

    # Stream of 40 expert-sized copies on one copy stream, like a prefill ring
    n = 40
    h = torch.empty(n * EXPERT_SLOT, dtype=torch.uint8, pin_memory=True)
    d = torch.empty(n * EXPERT_SLOT, dtype=torch.uint8, device=dev)
    stream = torch.cuda.Stream()
    best = 1e30
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.stream(stream):
            for i in range(n):
                a, b = i * EXPERT_SLOT, (i + 1) * EXPERT_SLOT
                d[a:b].copy_(h[a:b], non_blocking=True)
        stream.synchronize()
        best = min(best, time.perf_counter() - t0)
    results["h2d_40_expert_copies_gbps"] = round(n * EXPERT_SLOT / best / 1e9, 2)
    del h, d


def gpu_mem(results):
    dev = torch.device("cuda")
    n = 2 << 30
    x = torch.empty(n, dtype=torch.uint8, device=dev)
    y = torch.empty(n, dtype=torch.uint8, device=dev)
    x.random_(0, 255)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = 1e30
    for _ in range(10):
        s.record(); y.copy_(x); e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / 1e3)
    results["gpu_copy_gbps_rw"] = round(2 * n / best / 1e9, 1)
    xf = x.view(torch.float16)
    best = 1e30
    for _ in range(10):
        s.record(); xf.sum(dtype=torch.float32); e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / 1e3)
    results["gpu_read_sum_gbps"] = round(n / best / 1e9, 1)
    del x, y, xf
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    results["gpu_free_gib_idle"] = round(free / 2**30, 2)
    results["gpu_total_gib"] = round(total / 2**30, 2)


def pinned_limits(results, pin_max_gib):
    rows = []
    avail_kb = int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1])
    cap = min(pin_max_gib, int(avail_kb / 2**20 * 0.8))
    for gib in [4, 16, 32, 64, 96, 112]:
        if gib > cap:
            rows.append({"gib": gib, "skipped": f"above cap {cap} GiB"})
            continue
        t0 = time.perf_counter()
        try:
            h = torch.empty(gib << 30, dtype=torch.uint8, pin_memory=True)
            h[:: 1 << 20].fill_(1)
            rows.append({"gib": gib, "ok": True, "alloc_s": round(time.perf_counter() - t0, 2)})
            del h
        except Exception as ex:   # record the failure; this is the data we want
            rows.append({"gib": gib, "ok": False, "error": str(ex)[:200]})
            break
    results["pinned_alloc"] = rows
    results["ulimit_memlock"] = os.popen("ulimit -l").read().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--pin-max-gib", type=int, default=112)
    a = ap.parse_args()
    results = {"device": torch.cuda.get_device_name(0)}
    pcie(results)
    gpu_mem(results)
    pinned_limits(results, a.pin_max_gib)
    os.makedirs(a.out, exist_ok=True)
    json.dump(results, open(os.path.join(a.out, "gpu_pcie.json"), "w"), indent=1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
