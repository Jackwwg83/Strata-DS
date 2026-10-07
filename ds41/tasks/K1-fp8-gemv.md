# K1: FP8 block-scaled GEMV for DeepSeek V4.1 Flash decode

Owner: Codex (gpt-6-astra). Reviewer and GPU tester: Claude Code.
Branch: `feature/ds41-k1-fp8-gemv`, created from `feature/ds41`.

## Why

In decode, every token reads all dense FP8 weights of DeepSeek V4.1 Flash once (about 8.5 GB).
Decode is limited by memory bandwidth, not by compute. This kernel must read FP8 weights at close to the
GPU's memory bandwidth. It must not need FP8 tensor cores: the RTX 3090 (sm_86) has none.

## What to compute

Same math as `linear()` for FP8 weights in `ds41/proto/ref/model.py` (DeepSeek's reference code):

1. Quantize the activation `x` to FP8 E4M3, in blocks of 32 values along K:
   - `amax = max(|x_block|, 1e-4)`
   - `s = 2^ceil(log2(amax * (1/448)))` (power of two; see `round_pow2_scale` in `ds41/proto/torch_kernels.py`)
   - `xq = fp8_e4m3(clamp(x / s, -448, 448))`, round to nearest even
2. Weight `W` is FP8 E4M3, row-major `[N, K]`. Its scale `S_w` is E8M0 (one byte, value `2^(byte - 127)`), one
   scale per 32x32 block: shape `[ceil(N/32), K/32]`, row-major.
3. `y[m, n] = sum_k (xq[m,k] * s[m, k/32]) * (W[n,k] * S_w[n/32, k/32])`, accumulated in FP32.
4. Output `y` as BF16, round to nearest even.

The reference is `act_quant` + `fp8_gemm` in `ds41/proto/torch_kernels.py`. All products are exact in FP32;
only the summation order may differ from the reference.

## Interface

New files (do not change upstream Strata files except the one CMake include line below):

- `include/strata/ds41/fp8_gemv.hpp`
- `src/ds41/kernels/fp8_gemv.cu`
- `src/ds41/kernels/fp8_gemv_parity.cpp`
- `cmake/ds41.cmake` (targets for everything under `src/ds41/`)
- `CMakeLists.txt`: add one line `include(cmake/ds41.cmake)` inside the CUDA section

```cpp
namespace strata::ds41 {
// x: [m, k] BF16 (uint16_t bits), w: [n, k] FP8 E4M3 bytes, w_scale: [ceil(n/32), k/32] E8M0 bytes,
// y: [m, n] BF16. 1 <= m <= 8. k % 32 == 0. All pointers are device pointers. Asynchronous on `stream`.
void fp8_block_gemv(const uint16_t* x, int m, int64_t k,
                    const uint8_t* w, const uint8_t* w_scale, int64_t n,
                    uint16_t* y, void* stream);
}
```

Follow the style of `src/kernels/cuda/bf16_gemv.cu` and `src/kernels/bf16_gemv_parity.cpp`: one file comment that
says what the file is, short comments that say why, no new dependencies, C++17, CUDA 12.x.

## Shapes that must work and be fast (decode, m = 1, and verify windows, m <= 8)

| Weight | N | K |
| --- | ---: | ---: |
| attn.wq_a | 1280 | 5120 |
| attn.wq_b | 32768 | 1280 |
| attn.wkv | 512 | 5120 |
| attn.wo_b | 5120 | 8192 |
| attn.indexer.wq_b | 4096 | 1280 |
| ffn.shared_experts.w1 / w3 | 2304 | 5120 |
| ffn.shared_experts.w2 | 5120 | 2304 |
| engram.wkv | 25600 | 6144 |

## Acceptance

1. `fp8_gemv_parity --selftest` exits 0. It must:
   - build random inputs with a fixed seed for every shape above and m in {1, 2, 4, 8};
   - include blocks with large outliers, all-zero blocks and negative values;
   - compute a CPU reference with the exact quantization rules above (double accumulation);
   - pass when `||y - y_ref|| / ||y_ref|| <= 2e-3` and the activation FP8 bytes and scales match the reference
     bit for bit (export them from the kernel through a debug entry point or a separate quantization kernel);
   - print one line per case: shape, m, relative error, time in microseconds, effective GB/s
     (weight bytes + scale bytes / time).
2. Speed on an RTX 4090 (measured by the reviewer): for m = 1 and n*k >= 5e6, effective bandwidth at least 70%
   of 954 GB/s (the measured read bandwidth). For m = 8, time at most 1.5x the m = 1 time.
3. Builds for sm_86, sm_89 and sm_120 with CUDA 12.8 (`-DCMAKE_CUDA_ARCHITECTURES="86;89;120"`).
4. Register the parity test with `add_test(... COMMAND fp8_gemv_parity --selftest)` and
   `SKIP_RETURN_CODE 77` when no GPU is present, as the upstream parity tests do.

## Constraints

- No FP8 tensor-core instructions; decode FP8 bytes to float in registers.
- Read the weight with coalesced, wide loads (for example 16 bytes per thread). The weight is used once per
  token; do not stage it through shared memory unless that is faster.
- Do not change numerics to make the test pass. If the rules above are unclear, write the question in the
  commit message and in `ds41/tasks/K1-fp8-gemv.QUESTIONS.md`.
