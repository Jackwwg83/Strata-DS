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

Scratch is retained in a bounded, thread-local cache. Each slot belongs to one
CUDA device and a lifetime-unique stream ID. A slot is reusable only after its
recorded completion event succeeds; in-flight calls get separate slots. Eight
slots, each at most 2 MiB, bound retained memory per host thread. Nearby shapes
share 64 KiB size classes. A warm call uses no allocation/free API. The cache
does not change CUDA memory-pool retention or any process/device setting.

Graph capture, oversized requests, and cache pressure use per-call stream-
ordered allocation/free. Capture therefore owns its allocation nodes instead
of embedding a reusable cache pointer. Thread teardown waits on each valid
completion event before freeing its allocation. Failed event recording prevents
reuse; an invalid/terminated context is left to CUDA's resource teardown.

There are no host score/index copies, explicit stream/device synchronizations
in ordinary calls, external libraries, or modified fixed interfaces/tests.

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

## Iteration 1 GPU result

The RTX 4090 queue passed commit `d77475a1cce001d737e4ce998a3eb2e3b781c498`
on 2026-10-05, with zero score mismatches and complete top-k overlap in every
case. It reported `us_t16k=668.7`, `us_t128k=839.7`, and `score_us=1508`.
[Exact queue result](https://github.com/Jackwwg83/Strata-DS/issues/3#issuecomment-5993612540)

## Iteration 2: isolate warm allocation overhead

Iteration 2 changes only scratch lifetime/ownership; every GPU kernel remains
byte-identical to that passing parent. It removes the repeated warm
`cudaMallocAsync`/`cudaFreeAsync` calls rather than adjusting a pool's retention
threshold. This is an allocation-overhead experiment, not a measured speedup.

Checks before submission:

- CUDA 12.8/C++17 compilation passes again for sm_86, sm_89, and sm_120
- All six histogram/selection CPU model suites still pass
- `verify_scratch.cpp` compiles with C++17 and strict warnings, and tests the
  actual scratch header against a deterministic runtime stub: warm reuse,
  leased/pending slots, stream/device identity, graph-capture fallback, bounded
  memory, exception paths, and separate host-thread teardown
- The lifecycle test also passes AddressSanitizer and UndefinedBehaviorSanitizer
  with LeakSanitizer disabled because this executor uses ptrace; the stub also
  asserts that every test releases all mock allocations
- A source comparison confirms every GPU kernel is unchanged from iteration 1
- `git diff --check` passes

The lifecycle stub is not a CUDA-driver concurrency test. Iteration 2's GPU
acceptance and timings still require the fixed queue test. Runtime API contracts:
[events](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html),
[stream IDs/capture](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html).
