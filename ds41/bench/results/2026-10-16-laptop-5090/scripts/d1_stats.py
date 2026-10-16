#!/usr/bin/env python3
"""Summaries of the measurement protocol (d1.sh). Usage: d1_stats.py D1_DIR OUT.json
Cold: per condition the runs' decode tokens/s over all 256 steps (median, 95% bootstrap interval of the median,
min-max), the no-stall mean, prefill tokens/s, stalls per run, and step time p50/p95/p99 over all steps of all runs.
Warm: per request kind decode tokens/s and time to first token (median, interval), gaps p50/p95/p99.
Sustained: tokens/s of each request, the first and last 5, and the hardware samples of the window (GPU power,
temperature, SM clock, throttle reasons, CPU temperature) with the GPU energy per generated token."""
import collections, glob, json, os, random, re, statistics, sys

d1, out = sys.argv[1:3]
rng = random.Random(1)


def boot_median(xs, n=10000):
    if len(xs) < 2:
        return [xs[0], xs[0]] if xs else [None, None]
    ms = sorted(statistics.median(rng.choices(xs, k=len(xs))) for _ in range(n))
    return [round(ms[int(0.025 * n)], 2), round(ms[int(0.975 * n)], 2)]


def pct(xs, p):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(p / 100 * len(xs)))], 2)


res = {"cold": {}, "warm": {}, "sustained": {}}
cold = collections.defaultdict(list)
for log in glob.glob(f"{d1}/*-r[0-9].log"):
    name = os.path.basename(log)[:-4]
    cond = name.rsplit("-r", 1)[0]
    t = open(log).read()
    dec = re.search(r"decode_ms_per_token ([\d.]+)", t)
    pf = re.search(r"prefill_tokens (\d+) ms ([\d.]+) tok_s ([\d.]+)", t)
    steps = [list(map(float, l.split())) for l in open(f"{d1}/{name}.steps") if l.strip() and not l.startswith("#")]
    if not dec or not steps:
        continue
    tot = [s[1] for s in steps]
    calm = [x for x in tot if x <= 100]
    cold[cond].append({"run": name, "decode_tok_s": 1000 / float(dec.group(1)), "nostall_tok_s": 1000 / statistics.fmean(calm),
                       "prefill_tok_s": float(pf.group(3)), "prefill_s": float(pf.group(2)) / 1000, "stalls": len(tot) - len(calm),
                       "steps": tot})
for cond, runs in sorted(cold.items()):
    d = [r["decode_tok_s"] for r in runs]
    allsteps = [x for r in runs for x in r["steps"]]
    res["cold"][cond] = {"runs": len(runs), "decode_tok_s_median": round(statistics.median(d), 2), "decode_ci95": boot_median(d),
                         "decode_min_max": [round(min(d), 2), round(max(d), 2)],
                         "nostall_tok_s_median": round(statistics.median(r["nostall_tok_s"] for r in runs), 2),
                         "prefill_tok_s_median": round(statistics.median(r["prefill_tok_s"] for r in runs), 1),
                         "prefill_s_median": round(statistics.median(r["prefill_s"] for r in runs), 1),
                         "stalls_per_run": round(statistics.fmean(r["stalls"] for r in runs), 1),
                         "step_ms_p50_p95_p99": [pct(allsteps, 50), pct(allsteps, 95), pct(allsteps, 99)]}
warm = [json.loads(l) for l in open(f"{d1}/warm.jsonl")]
by = collections.defaultdict(list)
for r in warm:
    if r["label"] != "warmup":
        by[r["label"].rsplit("-r", 1)[0]].append(r)
for k, rs in sorted(by.items()):
    d = [r["decode_tok_s"] for r in rs]
    tt = [r["ttft_s"] for r in rs]
    gaps = [g for r in rs for g in r["gaps_ms"]]
    res["warm"][k] = {"requests": len(rs), "prompt_tokens": rs[0]["prompt_tokens"],
                      "completion_tokens_median": statistics.median(r["completion_tokens"] for r in rs),
                      "decode_tok_s_median": round(statistics.median(d), 2), "decode_ci95": boot_median(d),
                      "decode_min_max": [round(min(d), 2), round(max(d), 2)],
                      "ttft_s_median": round(statistics.median(tt), 2), "ttft_min_max": [round(min(tt), 2), round(max(tt), 2)],
                      "gap_ms_p50_p95_p99": [pct(gaps, 50), pct(gaps, 95), pct(gaps, 99)]}
sus = [json.loads(l) for l in open(f"{d1}/sustained.jsonl")]
marks = {}
for l in open(f"{d1}/marks.txt"):
    ts, what, name = l.split()
    marks[(what, name)] = float(ts)
t0, t1 = marks[("start", "sustained")], marks[("end", "sustained")]
tele = [l.rstrip("\n").split("\t") for l in open(f"{d1}/tele.tsv")]
head, rows = tele[0], [r for r in tele[1:] if len(r) == len(tele[0])]
col = {c: i for i, c in enumerate(head)}
win = [r for r in rows if t0 <= float(r[col["t"]]) <= t1]
f = lambda r, c: float(r[col[c]])
energy = sum(f(r, "gpu_w") for r in win) * 1.0          # 1-second samples: joules
tokens = sum(r["completion_tokens"] for r in sus)
d = [r["decode_tok_s"] for r in sus]
res["sustained"] = {"requests": len(sus), "seconds": round(t1 - t0), "tokens": tokens,
                    "decode_tok_s_median": round(statistics.median(d), 2), "first5_mean": round(statistics.fmean(d[:5]), 2),
                    "last5_mean": round(statistics.fmean(d[-5:]), 2), "per_request": [round(x, 2) for x in d],
                    "gpu_w_mean": round(statistics.fmean(f(r, "gpu_w") for r in win), 1),
                    "gpu_c_mean_max": [round(statistics.fmean(f(r, "gpu_c") for r in win), 1), max(f(r, "gpu_c") for r in win)],
                    "sm_mhz_mean": round(statistics.fmean(f(r, "sm_mhz") for r in win)),
                    "throttle_samples": sum(1 for r in win if int(r[col["throttle"]], 16) != 0), "samples": len(win),
                    "throttle_reasons_seen": sorted({r[col["throttle"]] for r in win}),
                    "cpu_c_mean_max": [round(statistics.fmean(f(r, "cpu_c") for r in win), 1), max(f(r, "cpu_c") for r in win)],
                    "gpu_joules_per_token": round(energy / tokens, 2)}
json.dump(res, open(out, "w"), indent=1)
for k, v in res["cold"].items():
    print(f"cold {k:12s} n={v['runs']} decode {v['decode_tok_s_median']} {v['decode_ci95']} [{v['decode_min_max']}] nostall "
          f"{v['nostall_tok_s_median']} prefill {v['prefill_tok_s_median']} stalls {v['stalls_per_run']} steps {v['step_ms_p50_p95_p99']}")
for k, v in res["warm"].items():
    print(f"warm {k:6s} n={v['requests']} prompt {v['prompt_tokens']} decode {v['decode_tok_s_median']} {v['decode_ci95']} "
          f"[{v['decode_min_max']}] ttft {v['ttft_s_median']} gaps {v['gap_ms_p50_p95_p99']}")
s = res["sustained"]
print("sustained", {k: v for k, v in s.items() if k != "per_request"})
