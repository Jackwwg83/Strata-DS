# K8-07: warp-only selector ablation

The public header and fixed acceptance test are unchanged. Only the K8 source and
its permitted support directory are modified. This branch starts from feature
`f75130394c5b56f0a72f20d04b2a06165a543c17` and transplants K8-02 control
`87b6d90888a995bf17edc039f90e0c8da4b552b6`.

Compared with that exact control, runtime source changes are limited to the
`select_top6` body and its launch-thread constant (128 to 32). GEMV, scoring,
`math.hpp`, reduction helpers, dispatch, supplied-stream handling and the 24 KiB
scratch layout/lifetime are unchanged. The optional graph test is unchanged;
the CPU selector model and these notes are updated for the ablation.

## Algorithm and lifetime

- Decode uses one warp per expert, four experts per CTA.
- For m=2..8, one CTA owns an expert. All its threads stage that expert's BF16 row
  into 20 KiB of shared FP32 storage once. Each token warp reuses the row.
  Every BF16 weight has exactly one global-load owner, independent of m.
- Each warp produces the reference-ordered FP32 logit and its unbiased nonlinear
  score. Computing scores in this grid spreads nonlinear work across the device.
- The selector launches m independent 32-thread CTAs for every m=1..8: exactly
  one full warp per token, so separate token CTAs can be scheduled independently.
  Each lane holds 12 candidates at IDs `lane + 32*j` in registers.
- Six deterministic warp argmax rounds select the winners. Each round broadcasts
  the original ID and the owner's unbiased double score using warp shuffles.
  Lane 0 writes the ID and adds the score in selected order; output lane i retains
  score i for normalization. All 32 lanes participate in every shuffle.
- Selection uses no shared memory or CTA barriers. The control required two CTA
  barriers per round plus one final barrier, 13 executed barriers per token.
- Both launches use the supplied stream. There are no host copies or device/stream
  synchronizations in the implementation.
- Scratch is a K8-only, 24 KiB score array per device, allocated on its first eager
  call for all eight tokens and retained for process lifetime. A mutex protects
  device registration. The engine's guarantee that K8 calls on a device do not
  overlap makes reuse safe; other tasks never share this storage.

## Numerical reasoning

`ops.cu::bf16_gemv_k` assigns dimension `lane + 32*j` to each lane, accumulates
160 terms in ascending j using FP32 FMA, then reduces offsets 16,8,4,2,1. Both
router paths retain that sequence exactly. BF16-to-FP32 shared staging is exact;
it does not quantize the values or split/reassociate the sum.

K8.md and the fixed interface specify FP32 logits followed by sqrt(softplus),
selection on score+bias, and normalization from unbiased selected scores. They
do not require an additional FP32 rounding after the nonlinear step. The fixed
acceptance test explicitly uses double for nonlinear scores, comparisons and
normalization. `math.hpp` follows that expression and the `z > 20` branch exactly.
The original baseline uses floats there; retaining its intermediate rounding
would create artificial ties for some near-tie inputs and differ from the fixed
oracle. No epsilon-based tie rule is added.

For ordered numeric scores, `better` selects higher score and then lower ID, a
strict total order on distinct expert IDs. Each local and warp reduction
therefore returns the same maximum regardless of grouping. The winner is removed
by both setting its score to negative infinity and its ID to 384, so it cannot be
selected again, even when valid scores are negative infinity. Exactly one thread
owns its unbiased score, including zero. Repeating this six times is equivalent
to the fixed oracle's partial sort. The denominator is summed in selected order.

## Validation completed without a GPU

- CUDA 12.8 C++17 compile: sm_86, sm_89 and sm_120 passed
- All three architectures: selection uses 80 registers/thread, zero shared
  memory, zero barriers and no stack/spill loads/stores; the full implementation
  uses at most 20,480 shared bytes, below the 99 KiB task limit
- sm_89 scoring resource counts remain unchanged: tile kernels 30 registers,
  decode 27. The control selector used 37 registers/thread and 104 shared bytes
  with 128 threads; the new selector uses 32 threads and 80 registers/thread
- `host_semantics.cpp`: 2,503 routing comparisons against an independent partial
  sort; 13,824 bit-exact FP32 logits across all m=1..8; maximum relative weight
  difference 0 in the CPU model
- Explicit cases include equal scores, a near-tie erased by FP32 score+bias,
  softplus threshold neighbors, large positive values, strongly negative values,
  zero scores after underflow, negative bias and repeated ties, every owner
  lane/register, six successive winners owned by one lane, negative-infinity
  comparison ties, maximal finite biases and 1,024 permutation cases
- CPU weights match the independent oracle bit-for-bit; no relaxed tolerance
  masks a normalization-order difference
- Exact control audit verifies that code outside the selector and its thread
  constant is byte-identical, including all GEMV/scoring and scratch logic

Run the portable semantic model from the repository root:

    g++ -std=c++17 -O3 -march=native src/ds41/kernels/k8/host_semantics.cpp -o /tmp/k8_semantics
    /tmp/k8_semantics

`graph_validation.cu` is an additional GPU regression test, separate from the
fixed acceptance test. Its build command is at the top of that file. It checks
all m values on a non-default stream, warms up at m=1 before larger captures,
checks the graph has two nodes, changes inputs over 24 replays, and checks exact
ties and near-ties. It was compile-checked only.

## Integer order keys (2026-10-09)

`select_top6` now compares `order_key(score + bias)` instead of the doubles: the IEEE bits as a signed integer,
with the negative values' low 63 bits flipped, -0 folded to +0, and NaN mapped to the lowest key (a NaN never wins,
as with the double compare). The order equals `better` on doubles, so the IDs, the selected order and the weight
bits are unchanged. FP64 compares are slow on GPUs with few FP64 units: on the RTX 5090 Laptop, select_top6 took
10.8 us per decode layer and now takes 4.3 us (nsys, 40 layers). `graph_validation.cu` is now the CTest target
`k8_graph_validation`. It keeps the previous scorer and selector as a reference and requires equal IDs and equal
weight bits for every case; it adds negative, mixed-sign, extreme and NaN biases.

The selector trades more candidates/registers per lane for removal of cross-warp
reductions and barriers. The m=1 path now has one selector warp instead of four,
so reduced inter-warp latency hiding and longer serial candidate scans could
outweigh the synchronization savings. Only GPU measurement can resolve this.
No block-size specialization or narrowed math is used for any m.

CPU checks establish the algorithm and rounding-order model, not actual device
execution. GPU correctness, CPU/CUDA libm parity at extremely close scores,
graph capture/replay, sanitizer results, and latency remain unverified until a
GPU is available. No speedup claim is made from compile-only validation.
