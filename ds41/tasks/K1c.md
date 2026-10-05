# K1c: faster FP8 block-scaled GEMV for decode

CI-TARGETS: strata_ds41 k1c_fp8_gemv_test
CI-TEST: ./k1c_fp8_gemv_test
CI-FILES: src/ds41/kernels/fp8_gemv.cu src/ds41/kernels/fp8_gemv/
CI-ISSUE: TBD

## What it computes

The decode GEMV of every FP8 dense weight (attention, shared expert, indexer, Engram projection): about 8.5 GB read
per generated token. Interface: `fp8_block_gemv_q` in `include/strata/ds41/fp8_gemv.hpp` (K1/K1b; already correct
and graph-capturable). The math is in `ds41/tasks/K1-fp8-gemv.md`: FP8 E4M3 weight with E8M0 32x32 block scales,
already-dequantized FP32 activations `x_deq [m][k]`, FP32 accumulation, BF16 output, m = 1..8.

The header (including `detail::` helpers and the dispatch policy declared there) is fixed. Put new kernels, tuning
tables and helpers in `fp8_gemv.cu` or under `src/ds41/kernels/fp8_gemv/`. `fp8_quantize_activation_f32` must stay
bit-exact (the test compares the whole linear against `ops::fp8_linear`).

## Acceptance (`src/ds41/tests/k1c_fp8_gemv_test.cu`, fixed)

- Every shape, m = 1 and 8: relative L2 against `ops::fp8_linear` at most 2e-3.
- Timing rotates through >= 256 MB of weight copies, so every call reads DRAM as in the model (no L2 reuse).
- Ranking: `score_us = token_us_m1 + token_us_m8 / 8`: the dense GEMV time of one decode token (each shape weighted
  by how often it runs per token), alone and in an 8-token verify window.
- Bandwidth limit on an RTX 4090 (measured 954 GB/s): about 9,000 us for m = 1. Current (K1b): see issue.

## Hints

Small N (512 to 4096 rows) leaves most SMs idle with one warp per row: split K across warps or blocks. Large N:
wide loads, enough bytes in flight per SM, no shared-memory staging. For m = 8, decode each weight value once and
keep 8 accumulators. Avoid integer division in the inner loop.
