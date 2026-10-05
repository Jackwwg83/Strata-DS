# K5-03: persistent scores and BF16 histogram selection

This implementation keeps the reference's SIMT dot-product order. Eight warps
share a 32-by-128 FP32 query tile and 32 FP32 weights, then visit more keys with
a capped, grid-stride launch. Each key's four BF16 elements per lane are loaded
once and reused for all heads. A zero candidate byte skips those loads and all
dot products. The three BF16 rounding points and the 16/8/4/2/1 XOR reduction
are retained; the head sum stays in increasing head order.

The score kernel also builds a private, 256-bin histogram of the upper byte of
the ordered BF16 score. One reduction picks the upper byte, and one further
histogram picks the lower byte. There is no full sort or compact-candidate
sort. Masked `-inf` scores are included so a request larger than the reachable
set still has the reference's lower-index tie behavior.

An exact cutoff is followed by three stable steps:

1. Count strictly greater and equal scores for each 1024-position tile
2. Exclusive-scan the two counts across tiles
3. Scan each tile in position order and directly scatter the selected indices

The remaining tie budget chooses the earliest equal positions. This also
produces ascending output without sorting or nondeterministic atomic offsets.
Signed zeros share one ordered key.

`candidate_blocks` computes and pins block maxima on the GPU, then uses the
same histogram/compaction machinery. This entry point accepts arbitrary FP32
scores, so it refines four bytes instead of assuming BF16. The last block is
set to positive infinity before selection, ties use lower block indices, and
negative-infinity blocks are excluded from the final mask. Partial blocks,
non-default block sizes, zero selections, and selecting all blocks are handled.

Scratch allocation and release are stream ordered and private to each call.
There are no host score/index copies, explicit stream/device synchronizations,
external libraries, or modified fixed interfaces/tests.

## Verification

Checked on 2026-10-05 against `feature/ds41` base
`1e140ee2f74df4ca7e0f64d0af0284994a70f767`:

- CUDA 12.8, C++17 compile-only checks pass for sm_86, sm_89, and sm_120
- Ptxas reports 17,536 bytes maximum static shared memory per block, with no
  stack frame or register spill loads/stores on all three targets
- The score kernel uses 39 registers on sm_86/sm_89 and 72 on sm_120
- `python3 src/ds41/kernels/k5/verify_histogram.py` passes six CPU model suites:
  exhaustive non-NaN BF16 ordering, paired warp scans, selection/masks/ties,
  more than 256 compaction tiles, arbitrary-FP32 candidate blocks, and unique
  persistent ownership of every key
- `git diff --check` passes

These are compile/resource and algorithm-model checks. They do not establish
GPU numerical parity, race freedom, queue CMake integration, or speed. The
fixed GPU acceptance test and queue timings remain the required validation.
