#!/usr/bin/env python3
"""Roofline of one decode token on the laptop: for each resource, the bytes a token moves through it, the time those
bytes take at the measured bandwidth (a lower bound), and the time the step logs measure for that part.
Usage: roofline.py D1_DIR PACK_DIR RUN_GLOB(e.g. 'code-r*')
Inputs: D1_DIR/bw.jsonl (pinned H2D, device copy), membw.jsonl (RAM read), ssd.jsonl (expert-sized O_DIRECT reads),
the runs' --step-log files (no-stall steps 33-255: tiers warm), PACK_DIR/index.txt (dense tensors) and experts.bin."""
import glob, json, os, statistics, sys

d1, pack, pattern = sys.argv[1:4]
bw = [json.loads(l) for l in open(f"{d1}/bw.jsonl")]
big = max(bw, key=lambda r: r["bytes"])
gpu_mem, h2d = big["gpu_mem_GBps"], big["h2d_GBps"]
ram = max(json.loads(l).get("read_gbps", 0) for l in open(f"{d1}/membw.jsonl"))
ssd = {r["threads"]: r for r in map(json.loads, open(f"{d1}/ssd.jsonl"))}
expert_bytes = os.path.getsize(f"{pack}/experts.bin") / 15360
dense = 0
for l in open(f"{pack}/index.txt"):
    p = l.split()
    if l.startswith("#") or len(p) < 3 or p[0] == "embed.weight":   # the embedding: one row per token
        continue
    dense += int(p[-1])
cols = None
rows = []
for f in sorted(glob.glob(f"{d1}/{pattern}.steps")):
    lines = open(f).read().splitlines()
    cols = {c: i for i, c in enumerate(lines[0].lstrip("# ").split())}
    data = [list(map(float, x.split())) for x in lines[1:] if x.strip()]
    rows += [d for d in data[32:] if d[cols["total_ms"]] <= 100]
m = lambda c: statistics.fmean(r[cols[c]] for r in rows)
vram, ram_e, ssd_e, pref = m("vram_hits"), m("ram_cpu"), m("ssd"), m("prefetched")
ssd_bw = ssd.get(8, ssd[max(ssd)])["GBps"]
table = [
    ("GPU memory: dense weights + VRAM experts", dense + vram * expert_bytes, gpu_mem, m("wait_ms"),
     "measured: the worker's wait for the GPU (wait_ms)"),
    ("RAM: experts the CPU computes", ram_e * expert_bytes, ram, m("cpu_experts_ms"), "measured: cpu_experts_ms"),
    ("SSD: experts read per token", ssd_e * expert_bytes, ssd_bw, m("admit_ms"), "measured: admit_ms (8 threads bandwidth)"),
    ("PCIe: prefetched experts copied to the GPU", pref * expert_bytes, h2d, None, "overlaps the GPU's work"),
]
print(f"runs {pattern}: {len(rows)} steps; mean step {m('total_ms'):.1f} ms; expert {expert_bytes / 2**20:.2f} MiB; "
      f"per token: VRAM experts {vram:.0f}, CPU (RAM) {ram_e:.1f}, SSD {ssd_e:.1f}, prefetched {pref:.1f}")
print(f"bandwidths: GPU memory {gpu_mem} GB/s, pinned H2D {h2d} GB/s, RAM read {ram} GB/s, SSD {ssd_bw} GB/s")
out = []
for name, b, gbps, meas, note in table:
    lb = b / 1e9 / gbps * 1e3
    out.append({"part": name, "GB_per_token": round(b / 1e9, 3), "GBps": gbps, "lower_bound_ms": round(lb, 2),
                "measured_ms": None if meas is None else round(meas, 2), "note": note})
    print(f"{name:45s} {b / 1e9:7.3f} GB  /{gbps:7.1f} GB/s = {lb:6.2f} ms   measured {'' if meas is None else f'{meas:.2f} ms'}  ({note})")
json.dump({"pattern": pattern, "steps": len(rows), "mean_step_ms": m("total_ms"), "parts": out},
          open(f"{d1}/roofline-{pattern.replace('*', 'x')}.json", "w"), indent=1)
