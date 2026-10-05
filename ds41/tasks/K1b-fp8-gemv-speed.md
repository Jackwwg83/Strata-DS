# K1b: make the FP8 block-scaled GEMV fast and graph-capturable

Owner: Codex (gpt-6-astra). Reviewer and GPU tester: Claude Code.
Branch: `feature/ds41-k1b-fp8-gemv-speed`, created from `feature/ds41` (which contains the merged K1).

## Measured K1 on an RTX 4090 (2026-10-05, `fp8_gemv_parity --selftest`)

Numerics pass everywhere (activation bytes and scales bit-exact, rel error ~0). Speed:

| Weight | N | K | m=1 time (us) | m=1 GB/s | m8 / m1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| attn.wq_a | 1280 | 5120 | 19.5 | 337 | 1.84 |
| attn.wq_b | 32768 | 1280 | 70.6 | 595 | 2.13 |
| attn.wkv | 512 | 5120 | 14.3 | 183 | 1.64 |
| attn.wo_b | 5120 | 8192 | 69.6 | 603 | 2.12 |
| attn.indexer.wq_b | 4096 | 1280 | 16.4 | 320 | 1.86 |
| shared.w1 / w3 | 2304 | 5120 | 26.5 | 446 | 1.94 |
| shared.w2 | 5120 | 2304 | 25.6 | 461 | 1.88 |
| engram.wkv | 25600 | 6144 | 236.5 | 666 | 2.20 |

Measured GPU read bandwidth on this card: 954 GB/s. One decode token runs these 40 times (engram twice), so
the dense part costs about 11 ms per token now; the bandwidth limit is about 8.9 ms.

## What to change

1. **No allocation in the call.** Add a graph-capturable API; keep the old one as a thin wrapper:
   ```cpp
   // dequantized FP8 activations (exactly the values K1 computes), [m, k] floats
   void fp8_quantize_activation_f32(const uint16_t* x, int m, int64_t k, float* x_deq, void* stream);
   // GEMV on already quantized activations; no allocation, no synchronization
   void fp8_block_gemv_q(const float* x_deq, int m, int64_t k, const uint8_t* w, const uint8_t* w_scale,
                         int64_t n, uint16_t* y, void* stream);
   ```
   Several weights read the same activation (wq_a, wkv and the compressor; shared w1 and w3), so the engine
   quantizes once and calls `fp8_block_gemv_q` several times.
2. **Small and medium N.** One warp per row leaves the GPU half idle when N is small. Use split-K (several warps
   or blocks per row, then a reduction) or another layout, chosen per shape at run time.
3. **m up to 8.** Load each weight vector once and reuse it from registers for all m rows. Target m8/m1 <= 1.5.

## Acceptance (measured by the reviewer on an RTX 4090)

- Numerics: unchanged. The existing parity test must still pass bit-exact for the quantization and within
  `2e-3` for the output; add the same checks for the new entry points.
- Speed, `fp8_block_gemv_q`, m = 1: at least 75% of 954 GB/s for N*K >= 5e6; at most 10 us for wkv (512x5120)
  and indexer.wq_b (4096x1280). m8/m1 <= 1.5 for every shape.
- Report the new numbers in the same table format, with the old ones beside them.

## Constraints

Same as K1 (`ds41/tasks/K1-fp8-gemv.md`): no FP8 tensor-core instructions, sm_86/89/120, CUDA 12.8, no new
dependencies, do not change numerics, commit in small groups. The Codex sandbox cannot write `.git`; write
`ds41/tasks/K1b.COMMITS.md` with the commit messages and file groups you want, and the reviewer commits.
