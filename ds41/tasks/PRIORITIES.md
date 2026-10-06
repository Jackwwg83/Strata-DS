# ds41 task priorities (maintained by the reviewer)

Read this file from `origin/feature/ds41` at the start of every work cycle. Only the reviewer edits it. Talk in the
coordination issue (see "How we talk" below), not here.

Last update: 2026-10-06 (UTC+8). Focus: prefill (M3): K13, K14, then K2.

## Queue status

The RTX 4090 box is stopped. The queue moves to the dev box: RTX 3060 12 GB (sm_86) + EPYC 7452, CUDA 12.8.
"QUEUE UP" in issue #8 says when it runs. Controls below marked (3060) are measured there. Decode tasks (K1c, K3, K5,
K7, K8, K10, K11) are paused: their controls are 4090 numbers, and the 3060 box has no model pack.

## Priorities (do them in this order)

| # | Task | Why | Control (merged) |
| --- | --- | --- | --- |
| 1 | K13 sparse attention, prefill chunk | new; baseline is K3 in a loop, 4.9 s per 4096-token chunk (40 layers) | K13-02, 29,171 us (3060); baseline 120,956 |
| 2 | K14 indexer, prefill chunk | new; baseline is K5 in a loop, 1.5 s per chunk (8 layers), more at long contexts | K14-02, 24,070 us (3060); K14-01 38,727; baseline 192,123 |
| 3 | K2 prefill GEMM | merged K2-06 (best of all 13 on the 3060); re-ranked on the 4090 / 5060 Ti later | K2-06, 22,940 us (3060); K2-10 24,200; K2-05 24,530 |
| - | K12 prefill experts (EXL3) | assigned to Codex (vendoring exllamav3); not open for variants yet | placeholder |
| - | K1c, K3, K5, K7, K8, K10, K11 | paused (decode; no 4090 box) | see git history of this file |

## Merge rules

- The reviewer merges. A variant is merged when it passes the queue, follows rule 7 of `ds41/tasks/README.md`
  (CUDA-graph capturable: no allocation, no host sync inside a call; scratch per device per kernel, allocated once),
  and beats the merged version by at least 3%.
- Base new variants on the current `origin/feature/ds41`.

## How we talk

GitHub issue #8 (ds41 coordination: reviewer <-> dots) is the only channel; its first comment has the message
format. Queue results stay in the task issues #1-#7.
