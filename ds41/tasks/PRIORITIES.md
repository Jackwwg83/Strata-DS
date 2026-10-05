# ds41 task priorities (maintained by the reviewer)

Read this file from `origin/feature/ds41` at the start of every work cycle. Only the reviewer edits it. Talk in the
coordination issue (see "How we talk" below), not here.

Last update: 2026-10-06 01:20 (UTC+8).

## Queue status

UP since 2026-10-05 23:00 (UTC+8) on an RTX 4090 + i9-14900K box. Every candidate is tested with the current
`src/ds41/tests/` (overlaid from `origin/feature/ds41`), including the rule 7 graph check. K10 results are invalid
until "K10 GOLDEN READY" is posted in issue #8.

## Priorities (do them in this order)

Controls are measured on the current box (RTX 4090 + i9-14900K) with the current tests.

| # | Task | Why | Control (merged) |
| --- | --- | --- | --- |
| 1 | K11 CPU EXL3 expert kernel (AVX2) | the CPU experts are ~90% of a decode step on AVX2 PCs; new task | vendored moe_mul1, 5,987 us (m1 3,573) |
| 1b | K10 GPU EXL3 experts | about 2 ms per token; the largest GPU expert cost | K10-03, 179.0 us (m1 95.04); codex-1 was 244.0 |
| 2 | K8 router top-k | merged; refine only if >= 3% faster and m=1 not slower | K8-07, 19.84 us (m1 16.38) |
| 3 | K3 sparse attention | merged | K3-11, 49.92 us |
| 4 | K5 indexer | merged | K5-14, 117.9 us |
| 5 | K7 hyper-connection mixes | merged | K7-06, 14.11 us (m1 12.29) |
| 6 | K1c decode GEMV | near the DRAM limit; m=1 must not regress | K1b, 8,830 us (m1 7,291) |
| 7 | K2 prefill GEMM | prefill work has not started | none merged; best K2-07 3,650 us (old box) |

## Merge rules

- The reviewer merges. A variant is merged when it passes the queue, follows rule 7 of `ds41/tasks/README.md`
  (CUDA-graph capturable: no allocation, no host sync inside a call; scratch per device per kernel, allocated once),
  and beats the merged version by at least 3%.
- Base new variants on the current `origin/feature/ds41`.

## How we talk

GitHub issue #8 (ds41 coordination: reviewer <-> dots) is the only channel; its first comment has the message
format. Queue results stay in the task issues #1-#7.
