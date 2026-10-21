# DeepSeek V4.1 Flash 1017 on a DGX Spark, 2026-10-18

The measurements behind [docs/DEEPSEEK_SPARK.md](../../../../docs/DEEPSEEK_SPARK.md).

**What ran.** One DGX Spark (GB10, 128 GB unified memory, DGX OS 24.04, driver 580.173.02, CUDA 13.0), `ds41_serve`
from this branch, the pack `pack-1017-exl3h`: LibertAI's EXL3 2.9 bpw experts, attention and shared-expert linears and
head, DeepSeek's other tensors (the merge setup writes, `--model 1017-EXL3-2.9BPW`). Server settings as setup writes
them: `--max-context 262144 --threads 16 --window-prompt-max 32 --spec dspark --draft-always`, `DS41_SSD_TAIL_SKIP=1`.

## Files

| File | What |
| --- | --- |
| [agent-session.txt](agent-session.txt) | every request of a coding-agent session (DeepSeek Harness on a MacBook, through an SSH tunnel), as the server logged it: prompt tokens, reused and read, prompt time, tokens written, ms per token, DSpark windows / drafts / accepted |
| [runs.txt](runs.txt) | single `ds41_serve` runs: the context sweep (8K-250K), the time to the first token after a 10,000-token context (several `--window-prompt-max` values), the root checkpoint, a teacher-forced 4,000-token prefill |
| [install/](install) | a fresh setup from a new clone: its log, and the sha256 of the pack it wrote, compared with the measured pack (byte for byte the same) |
| [quality.txt](quality.txt) | the KL runs: 50 segments x 255 tokens teacher-forced, against the official MXFP4 pack without skips |
| [deepseek-1017-dgx-spark.png](deepseek-1017-dgx-spark.png), [x-chart-data.json](x-chart-data.json), [x-chart.py](x-chart.py) | the chart, its numbers (from runs.txt) and the script that draws it |

## Results

- Agent session (29 turns after the first, context 10,310-58,516 tokens): 70,211 tokens written at 30.8 tokens/s
  (the engine's time per token over all of them); per turn 23.6-36.4, median 28.6; DSpark accepted 84.2% of its drafts.
- Context sweep, 200 tokens each: 8K 24.2, 32K 27.4, 64K 27.4, 128K 27.3, 250K 22.9 tokens/s.
- Quality against the official MXFP4 weights: KL 0.0740 [0.0639, 0.0858], same top token 88.6% (the default); 0.0532
  and 90.2% without the SSD skip.

Single runs except the quality table. The incremental 128K -> 250K read (122,021 new tokens, 1,466 s) is three times
slower than a cold 250K read (498 s) and is not explained yet.
