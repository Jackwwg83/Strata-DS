# GPU router: weight reuse and deterministic top six

The public header and fixed acceptance test are unchanged. Only the K8 source and
its permitted support directory are modified.

## Algorithm and lifetime

- Decode uses one warp per expert, four experts per CTA.
- For m=2..8, one CTA owns an expert. All its threads stage that expert's BF16 row
  into 20 KiB of shared FP32 storage once. Each token warp reuses the row.
  Every BF16 weight has exactly one global-load owner, independent of m.
- Each warp produces the reference-ordered FP32 logit and its unbiased nonlinear
  score. Computing scores in this grid spreads nonlinear work across the device.
- A second kernel per token repeatedly reduces `(biased score, expert ID)` pairs,
  emits six distinct experts, and normalizes their unbiased scores.
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
strict total order on distinct expert IDs. Each local, warp and CTA reduction
therefore returns the same maximum regardless of grouping. The winner is removed
by both setting its score to negative infinity and its ID to 384, so it cannot be
selected again, even when valid scores are negative infinity. Exactly one thread
owns its unbiased score, including zero. Repeating this six times is equivalent
to the fixed oracle's partial sort. The denominator is summed in selected order.

## Validation completed without a GPU

- CUDA 12.8 C++17 compile: sm_86, sm_89 and sm_120 passed
- sm_89 resource report: no spill loads/stores; maximum shared memory 20,480 bytes;
  tile kernels 30 registers, decode 27, selection 37
- `host_semantics.cpp`: 1,056 routing comparisons against an independent partial
  sort; 13,824 bit-exact FP32 logits across all m=1..8; maximum relative weight
  difference 0 in the CPU model
- Explicit cases include equal scores, a near-tie erased by FP32 score+bias,
  softplus threshold neighbors, large positive values, strongly negative values,
  zero scores after underflow, negative bias and repeated ties

Run the portable semantic model from the repository root:

    g++ -std=c++17 -O3 -march=native src/ds41/kernels/k8/host_semantics.cpp -o /tmp/k8_semantics
    /tmp/k8_semantics

`graph_validation.cu` is an additional GPU regression test, separate from the
fixed acceptance test. Its build command is at the top of that file. It checks
all m values on a non-default stream, warms up at m=1 before larger captures,
checks the graph has two nodes, changes inputs over 24 replays, and checks exact
ties and near-ties. It was compile-checked only.

CPU checks establish the algorithm and rounding-order model, not actual device
execution. GPU correctness, CPU/CUDA libm parity at extremely close scores,
graph capture/replay, sanitizer results, and latency remain unverified until a
GPU is available. No speedup claim is made from compile-only validation.
