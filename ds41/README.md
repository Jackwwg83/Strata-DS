# ds41: DeepSeek V4.1 Flash on Strata

This directory holds the work that ports Strata from Qwen3.8-Flash-Next to DeepSeek V4.1 Flash.
Upstream Strata files stay unchanged where possible, so upstream updates merge cleanly.

To run the model, follow a recipe: [docs/DEEPSEEK_LAPTOP.md](../docs/DEEPSEEK_LAPTOP.md) (a PC or laptop) or
[docs/DEEPSEEK_SPARK.md](../docs/DEEPSEEK_SPARK.md) (a DGX Spark). The engine is in `src/ds41/` and `include/strata/ds41/`.

| Path | Content |
| --- | --- |
| `docs/engine-plan.html` | Engine plan: targets, upstream mapping, milestones, kernel tasks |
| `proto/` | Python prototype: DeepSeek's own `model.py` (in `proto/ref/`, unmodified, MIT) run on one consumer GPU with EXL3 experts read from the SSD. It is the numerical reference for the engine. |
| `bench/scripts/` | Hardware and kernel measurement scripts used on Vast.ai |
| `bench/results/` | Measured results with machine, logs and HTML reports. Raw routing traces (`*.npz`) are not in git. |
| `tasks/` | The kernel and feature specs the engine's sources cite (K1-K15, ...) with their reports |
| `ci/` | The scripts that run the kernel tests on a remote GPU |

Early measurements (2026-10-05, before the recipes; see the reports for the machines and methods):

- RTX 4090 + Ryzen 9 7950X + 128 GB: RAM read 48 GB/s; exllamav3 CPU EXL3 expert kernel 44-46 GB/s
  (about 300 us per 3bpw expert); NVMe 6.8 GB/s for 13 MiB reads.
- The full model (EXL3 3bpw, 390 GB with Engram) runs on one RTX 4090 through the prototype: coherent greedy
  output; perplexity 2.0-3.4 on code and 4.5-6.8 on English technical text.
- Routing over 101K tokens: the hottest 10% of (layer, expert) pairs take 49% of accesses; an adaptive VRAM
  cache of 850 experts serves 53% of accesses.
