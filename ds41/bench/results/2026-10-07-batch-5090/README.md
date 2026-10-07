# Batch slots on the RTX 5090 box, 119.9 GiB (2026-10-07)

`batch_bench` (src/ds41/tests/batch_bench.cpp): N real chats (code, zh_chat, en_explain, agent from
tools/ds41/chat_ids.py) read into N slots, then 96 steps together, adaptive tier on; the mean of steps 9..96.
Machine: see ../2026-10-07-decode-5090/machine.txt (its disk caps at 6-8K random reads per s, so the engram reads,
96 per row and step, cost ~11 ms per row here; a local NVMe SSD should take ~0.5 ms).

| Slots | Quota | ms per step | tokens/s in all | GPU | engram | CPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 39.3 | 25.4 | 29.1 | 10.2 | 1.5 |
| 2 | 4 | 70.5 | 28.3 | 48.4 | 22.2 | 3.9 |
| 4 | 4 | 140.8 | 28.4 | 94.7 | 46.1 | 9.5 |
| 2 | 2 | 70.5 | 28.4 | 48.2 | 22.4 | 20.2 |
| 4 | 2 | 129.0 | 31.0 | 85.3 | 43.7 | 43.2 |

Each row brings its own ~100 routed misses per step (the conversations share few experts), copied over PCIe at
~46 GB/s or computed by the CPU, so the GPU time grows by ~20 ms per row; the dense weights, read once for all rows,
are the saving. Without the engram reads (a local NVMe SSD): 1 slot ~29 ms per token (~34 tokens/s), 4 slots ~85 ms per
step (~47 tokens/s in all).
