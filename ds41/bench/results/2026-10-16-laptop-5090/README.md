# 2026-10-15/16: DeepSeek V4.1 Flash on an RTX 5090 Laptop, 64 GB: speed, quality, install

The raw data behind [docs/DEEPSEEK_LAPTOP.md](../../../../docs/DEEPSEEK_LAPTOP.md). Laptop: MECHREVO Canglong, RTX 5090
Laptop GPU 24 GB (175 W limit, Dynamic Boost on), Ryzen 9 9955HX3D, 60.5 GiB usable RAM, 2 TB YMTC PC411 NVMe, Linux
7.1, driver 610.57, CUDA 13.3, BIOS N.1.18MRO43. Model: `vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw` at
`eca94a388a70841858feed8f057a9862e897aba4`.

## What ran, in order

| When | What | Source | Files |
| --- | --- | --- | --- |
| 10-15 15:45-18:05 | KL divergence: 200 segments x 256 tokens, teacher-forced decode, no skip (twice), route skip only, defaults | 742536c0 (dev build) | `quality/kld*`, `scripts/q1.sh`, `scripts/kld_stats.py` |
| 10-15 17:47 | decision rules written | | `quality/criteria.md` |
| 10-15 18:02-10-16 07:37 | HumanEval+ (164) and GSM8K (first 200), greedy, thinking off; needle in a haystack (60 prompts, 32K-256K); no skip vs defaults | 742536c0 | `quality/*.jsonl`, `quality/paired.json`, `quality/evalplus.json` |
| 10-16 00:52-03:54 | the doc's steps from scratch in a fresh clone (97438fe) | 97438fe | `install/` |
| 10-16 07:38-09:51 | the measurement protocol on that install: the doc's step 3, bandwidths, nsys, 45 cold runs, 20 warm API requests, 30 minutes sustained, 1-second hardware samples | 97438fe | `speed/` |
| 10-16 10:04 | two live requests for the demo video (thinking off) | 97438fe | `demo/` |

The engine source is the same in 742536c0 and 97438fe (97438fe adds READMEs and data). Every server was checked from
its own engine log to run the intended setting (`tail skip ON` lines: 0 without skips, 2 with). The first batch of
default-setting evals reached a no-skip server that had outlived its script; its 6 answers were discarded.

## Results

Speed (medians of 5; intervals are 95% bootstrap intervals of the median):

| Condition | Decode tokens/s | Prefill tokens/s |
| --- | ---: | ---: |
| cold, short prompts (4 kinds) | 30.0-32.0 | - |
| cold 4K / 32K / 64K / 128K / 256K | 26.1 / 25.6 / 25.5 / 25.2 / 24.4 | 244 / 844 / 859 / 904 / 752 |
| warm API, zh / en / code / 18K file | 39.3 / 37.2 / 33.2 / 34.7 | - |
| sustained 30 min (32 x 2,048 tokens) | 37.6 (first 5: 38.3, last 5: 38.0) | - |

Quality, defaults vs no skip: HumanEval+ 90.2% vs 91.5% (-1.2 points [-3.7, +1.2], McNemar p 0.63); GSM8K 98.0% vs
97.5%; needle 60/60 vs 60/60; KL divergence 0.0316 [0.0295, 0.0337] vs a noise floor of 0.0119, same top token
91.8% (floor 94.9%); by text kind from 0.022 (Python) to 0.047 (Chinese).

Install: 3 h (download 2 h 57 min at about 32 MB/s); the pack is byte-identical to the development pack.

## Files

| Path | What |
| --- | --- |
| `speed/cold-runs.tsv`, `speed/steps.tsv` | every cold run and every decode step (`--step-log` columns) |
| `speed/warm.jsonl`, `speed/sustained.jsonl` | every API request: tokens, time to first token, decode tokens/s, the gaps between tokens, the text |
| `speed/tele.tsv`, `speed/run-order.txt` | 1-second hardware samples; the run order and each run's start and end (to align them) |
| `speed/stats.json`, `speed/roofline.json`, `speed/bandwidth.jsonl`, `speed/nsys.txt` | the summaries, the roofline, the measured bandwidths, the nsys idle and per-layer summaries |
| `speed/step3.jsonl` | the first request through the doc's start script |
| `quality/kld.json`, `quality/kld200.list`, `quality/kld200.meta.json` | KL results; the 200 segments (token ids) and where each comes from |
| `quality/{humaneval,gsm8k,niah}_{noskip,default}.jsonl` | every answer; `evalplus.json` the sandbox results; `paired.json` the paired statistics |
| `quality/criteria.md`, `quality/read20.md` | the decision rules and their outcome; the 20 outputs read in full |
| `install/` | the install log with times, setup's own log, sha256 of the model files, the pack and the binaries |
| `baseline.md` | the search for another runtime to compare with |
| `demo/` | the demo video's two live requests (every streamed piece with its arrival time) and their hardware samples |
| `scripts/` | every script that produced these files (protocol, clients, statistics, charts, video) |

Not in git: the 13 GB log-probability files of the KL runs and the nsys report (12 MB); ask if you need them.
