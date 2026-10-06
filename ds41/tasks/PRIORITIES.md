# ds41 task priorities (maintained by the reviewer)

Read this file from `origin/feature/ds41` at the start of every work cycle. Only the reviewer edits it. Talk in the
coordination issue (see "How we talk" below), not here.

Last update: 2026-10-06 evening (UTC+8). Focus: DECODE speed. Prefill tasks continue after these.

## Why decode now

Measured on the RTX 4090 box (EPYC 7742, SAGE 1.59bpw pack, every expert in VRAM or RAM, nsys, one decode step):
the GPU waits 32 ms for the CPU experts, computes 17.4 ms, idles 5.4 ms. The reviewer and Codex work on the CPU
experts, CUDA graphs and wo_a (now FP8). The GPU compute is yours: the dense FP8 GEMVs are the largest part, and the
small ones (shared expert 5120 x 2304, wq_a, wkv) run at about half the bandwidth of the large ones.

## Queue status

The queue runs on the dev box: RTX 3060 12 GB (sm_86) + EPYC 7452, CUDA 12.8. "QUEUE UP" in issue #8 says when it
runs. Controls below are measured there on 2026-10-06 (merged code, origin/feature/ds41 2b6e621).

## Priorities (do them in this order)

| # | Task | Why | Control (merged, 3060) |
| --- | --- | --- | --- |
| 0 | K1c decode FP8 GEMV | about 5 ms of the 17.4 ms GPU time per token on the 4090; small N at half bandwidth (shared_w1_w3 241 GB/s, peak about 360) | K1b, score 29,170 us (m1 20,776, m8 67,194) |
| 1 | K7 hc mixes, decode | hc_partials + hc_finish about 1.2 ms per token on the 4090, 5x the bytes-read bound | merged K7, score 18.43 us (m1 15.36, m8 24.58) |
| 2 | K15 hc mixes, prefill sub-batch | 23% of a 32K prompt's GPU time on the 4090 | baseline (K7 loop), 13,294 us |
| 3 | K13 sparse attention, prefill chunk | | K13-02, 29,171 us |
| 4 | K14 indexer, prefill chunk | | K14-02, 24,070 us |
| 5 | K2 prefill GEMM | | K2-06, 22,940 us |
| - | K3, K5, K8, K10, K11, K12 | owned by the reviewer / Codex now (graph capture, mixed-K experts, CPU experts); not open | |

## Merge rules

- The reviewer merges. A variant is merged when it passes the queue, follows rule 7 of `ds41/tasks/README.md`
  (CUDA-graph capturable: no allocation, no host sync inside a call; scratch per device per kernel, allocated once),
  and beats the merged version by at least 3%.
- Base new variants on the current `origin/feature/ds41`.

## How we talk

GitHub issue #8 (ds41 coordination: reviewer <-> dots) is the only channel; its first comment has the message
format. Queue results stay in the task issues #1-#7.
