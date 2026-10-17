# exl3_smoe (vendored)

From sudoingX/dgx-spark-deepseek-v41 (Apache-2.0, Copyright 2026 sudoingX), commit cc59c3609e8e671418f39eb92aa16b0e4f770b77,
patches/exl3_moe/vllm_exl3_moe/csrc/:

- `exl3_common.cuh`: copied unchanged. Its trellis decode is adapted by sudoingX from exllamav3 v1.6.0 (MIT,
  Copyright (c) 2025 Turboderp; `LICENSE.exllamav3`).
- `exl3_smoe.cu`: not copied. `src/ds41/kernels/k19_exl3_smoe.cu` adapts its rot / gate-up / down kernels (the non-fused
  path) to K10's interface; the changes are listed at the top of that file.

`LICENSE` is the repository's Apache License 2.0.
