# 2026-10-15: DeepSeek V4.1 Flash on an RTX 5090 Laptop with 64 GB of RAM

The raw data behind [docs/DEEPSEEK_LAPTOP.md](../../../../docs/DEEPSEEK_LAPTOP.md). One run of `bench.sh` on
2026-10-15: the machine facts, the full test suite, a length matrix, short prompts, and requests through the API.

## The machine

MECHREVO Canglong laptop: RTX 5090 Laptop GPU (24 GB, driver 610.57.04, Dynamic Boost on), Ryzen 9 9955HX3D,
60.5 GiB of usable RAM, a 2 TB YMTC PC411 NVMe SSD, Linux 7.1 (Arch-based), CUDA 13.3. Swap: zram (priority 100),
swappiness 150. All in [machine.json](machine.json).

Model: `vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw` at `eca94a388a70841858feed8f057a9862e897aba4`, packed with
`tools/ds41/pack.py` (dense.bin 10.6 GB, experts.bin 108.9 GB). Engine source: commit `adad2779`; the API server's
`ds41_serve` also had `7b79a3e0` (a verify-window fix that no plain decode step uses).

## What ran

1. `ctest` (165 tests): 148 pass, 10 skip, 7 fail ([ctest-summary.txt](ctest-summary.txt)). The 10 skipped tests
   need the model pack (`--pack`); run by hand on the SAGE pack, all 10 and their other cases (19 runs) pass
   ([engine-tests.txt](engine-tests.txt)). The 7 failures need files this laptop does not have
   (`k10_exl3_moe_test`, `k11_cpu_moe_test`: a 3bpw test pack; `ple_parity`, `expert_parity`, `pool_test`: Qwen
   model files), a larger lock limit (`platform_memory_test`: mlock ulimit 8 MiB), or fail the same way before these
   changes (`hybrid_host_table_test`, 2 checks).
2. Length matrix: `ds41_generate --prefill --gen 256` (greedy), a fresh process per run, upstream's length-tiered
   speed prompts ([make_len_ids.py](make_len_ids.py)) at 4K, 32K, 64K, 128K (2 prompts each) and 256K (1 prompt).
   The page cache of the pack is dropped before each run.
3. Short prompts: the same, with 4 chat prompts of 51-328 tokens ([short-prompts.json](short-prompts.json),
   thinking on), 2 rounds.
4. The API: `serve/server.py` with the engine flags setup writes ([serve-config.json](serve-config.json),
   32K context), one warm-up request, then 7 streamed OpenAI chat requests of up to 512 tokens
   ([api-prompts.json](api-prompts.json)), timed by [api_bench.py](api_bench.py).

## Files

| File | What |
| --- | --- |
| [summary.tsv](summary.tsv) | one row per `ds41_generate` run: prefill, RAM tier fill, decode (all steps and without stalls), stalls, GPU wait, CPU experts, SSD reads per token |
| [steps.tsv](steps.tsv) | every decode step of every run (`--step-log` columns, with the run's name) |
| [api.jsonl](api.jsonl) | one line per API request: prompt tokens, time to first token, decode tokens/s (client and server), the answer's first characters |
| [machine.json](machine.json) | the laptop, driver, CUDA, kernel, swap, model revision |
| [engine-tests.txt](engine-tests.txt) | the pack tests run by hand (verify windows, DSpark spec, reset, failures, batch) |
| [bench.sh](bench.sh), [api_bench.py](api_bench.py), [make_len_ids.py](make_len_ids.py) | the scripts that produced this |
| [short-prompts.json](short-prompts.json), [api-prompts.json](api-prompts.json) | the prompts: text and token ids |

`decode_ms_per_token` is the mean over all 256 steps. `no_stall_ms_per_token` leaves out the steps over 100 ms.
A run's first 32 steps take about 54 ms each (the experts in GPU memory and the RAM tier are still cold); steps
129-255 take 28-30 ms. A server that is already running keeps them warm, which is why the API requests decode
faster (32-39 tokens/s) than a fresh `ds41_generate` run (about 30).

## Results

| Run | Prefill tokens/s | Decode tokens/s (all steps) |
| --- | ---: | ---: |
| 4K (2 runs) | 162 / 197 | 24.3 / 25.9 |
| 32K (2 runs) | 704 / 763 | 25.3 / 24.3 |
| 64K (2 runs) | 839 / 805 | 24.4 / 25.6 |
| 128K (2 runs) | 834 / 849 | 24.4 / 24.4 |
| 256K | 650 | 23.6 |
| agent, code, zh_chat, en_explain (2 rounds) | - | 29.2-30.8, mean 30.1 |

| API request | Prompt tokens | Time to first token | Decode tokens/s |
| --- | ---: | ---: | ---: |
| English question, round 1 (prompt cached by the warm-up) | 63 | 0.03 s | 35.6 |
| Chinese question, round 1 | 54 | 1.60 s | 39.1 |
| coding task, round 1 | 64 | 2.12 s | 34.0 |
| English question, round 2 | 63 | 2.11 s | 35.2 |
| Chinese question, round 2 | 54 | 1.79 s | 38.2 |
| coding task, round 2 | 64 | 2.09 s | 32.2 |
| 18K-token source file to summarize, round 1 | 18,062 | 32.98 s | 31.8 |
