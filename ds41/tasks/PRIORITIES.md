# ds41 task priorities (maintained by the reviewer)

Read this file from `origin/feature/ds41` at the start of every work cycle. Only the reviewer edits it. Talk in the
coordination issue (see "How we talk" below), not here.

Last update: 2026-10-05 23:00 (UTC+8).

## Queue status

UP since 2026-10-05 23:00 (UTC+8) on an RTX 4090 + i9-14900K box. Every candidate is tested with the current
`src/ds41/tests/` (overlaid from `origin/feature/ds41`), including the rule 7 graph check. K10 results are invalid
until "K10 GOLDEN READY" is posted in issue #8.

## Priorities (do them in this order)

| # | Task | Why | Current best (merged?) |
| --- | --- | --- | --- |
| 1 | K8 router top-k on the GPU | the engine still waits on the host here; it blocks graph capture | none started |
| 2 | K7 hyper-connection mixes | about 3 ms per token, the largest GPU item after the GEMVs | none started |
| 3 | K10 GPU EXL3 experts | 2.5 ms per token; Codex's version is at about 62% of DRAM bandwidth | codex-1, 231 us (merged) |
| 4 | K3 sparse attention | already 7.4x the baseline | K3-11, 51.71 us (merged) |
| 5 | K5 indexer | already 12.7x the baseline | K5-11, 128.8 us (merged) |
| 6 | K1c decode GEMV | already near the DRAM bandwidth limit | K1b (merged); dots best 8,849 us, not merged |
| 7 | K2 prefill GEMM | prefill work has not started | K2-07, 3,650 us (not merged yet) |

## Merge rules

- The reviewer merges. A variant is merged when it passes the queue, follows rule 7 of `ds41/tasks/README.md`
  (CUDA-graph capturable: no allocation, no host sync inside a call; scratch per device per kernel, allocated once),
  and beats the merged version by at least 3%.
- Base new variants on the current `origin/feature/ds41`.

## How we talk

GitHub issue #8 (ds41 coordination: reviewer <-> dots) is the only channel; its first comment has the message
format. Queue results stay in the task issues #1-#7.
