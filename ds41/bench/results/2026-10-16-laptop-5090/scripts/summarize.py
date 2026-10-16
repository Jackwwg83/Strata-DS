#!/usr/bin/env python3
"""rel1 folder -> summary.tsv (one row per run), steps.tsv (every decode step of every run, with a run column).
Usage: summarize.py RUN_DIR OUT_DIR"""
import re, sys, statistics
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
rows, steps_out, header = [], [], None
for log in sorted(src.glob("*.log")):
    run = log.stem
    if run.startswith(("serve", "ctest")):
        continue
    steps = src / f"{run}.steps"
    if not steps.exists():
        continue
    t = log.read_text(errors="replace")
    pf = re.search(r"prefill_tokens (\d+) ms ([\d.]+) tok_s ([\d.]+)", t)
    dec = re.search(r"decode_ms_per_token ([\d.]+)", t)
    fill = re.search(r"filled in ([\d.]+) s", t)
    tier = re.search(r"RAM tier (\d+) experts \(([\d.]+) GiB", t)
    if not (pf and dec and fill and tier):
        continue   # a run still going
    vram = re.search(r"(\d+) VRAM expert slots \(([\d.]+) GiB\)", t)
    lines = steps.read_text().splitlines()
    header = header or lines[0].lstrip("# ").split()
    data = [list(map(float, l.split())) for l in lines[1:] if l.strip()]
    for d in data:
        steps_out.append([run] + d)
    col = {name: i for i, name in enumerate(header)}
    calm = [d for d in data if d[col["total_ms"]] <= 100]
    mean = lambda rows_, c: statistics.fmean(r[col[c]] for r in rows_)
    rows.append({
        "run": run, "prompt_tokens": int(pf.group(1)), "prefill_s": round(float(pf.group(2)) / 1000, 1),
        "prefill_tok_s": float(pf.group(3)), "ram_tier_fill_s": float(fill.group(1)),
        "ram_tier_experts": int(tier.group(1)), "ram_tier_gib": float(tier.group(2)),
        "vram_expert_slots": int(vram.group(1)), "decode_steps": len(data),
        "decode_ms_per_token": float(dec.group(1)), "decode_tok_s": round(1000 / float(dec.group(1)), 1),
        "stalls_over_100ms": len(data) - len(calm),
        "no_stall_ms_per_token": round(mean(calm, "total_ms"), 1),
        "gpu_wait_ms": round(mean(calm, "wait_ms"), 1), "cpu_experts_ms": round(mean(calm, "cpu_experts_ms"), 1),
        "ssd_reads_per_token": round(mean(data, "ssd"), 2), "ssd_admit_ms": round(mean(calm, "admit_ms"), 1)})
keys = list(rows[0])
(out / "summary.tsv").write_text("\t".join(keys) + "\n" + "".join("\t".join(str(r[k]) for k in keys) + "\n"
                                                                    for r in rows))
(out / "steps.tsv").write_text("run\t" + "\t".join(header) + "\n" + "".join(
    "\t".join([s[0]] + [f"{v:g}" for v in s[1:]]) + "\n" for s in steps_out))
for r in rows:
    print(r["run"], r["prefill_tok_s"], r["decode_ms_per_token"], r["decode_tok_s"], r["no_stall_ms_per_token"],
          r["stalls_over_100ms"], r["ssd_reads_per_token"])
