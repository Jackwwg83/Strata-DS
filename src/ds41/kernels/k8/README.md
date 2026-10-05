# K8-08: full activation cache shared by four experts

Only K8 CI-FILES change. The fixed header, acceptance test, and reference ops
remain unchanged. The implementation covers every interface-legal m=1..8.

## Layout and algorithm

A CTA owns four experts and all m tokens. One warp owns each expert/token pair,
so the block has 128*m threads. Before computing any logits, its threads copy
all m activation rows into a persistent BF16 shared cache. Each activation has
exactly one global-load owner per CTA and is reused by all four experts.

Weights stream through one BF16 shared window of four rows by 2,048 columns.
The three phases cover columns 0..2047, 2048..4095, and 4096..5119. A producer
loads each real weight once, independently of m. The tail never loads or reads
the unused shared padding. A publication barrier precedes each phase's dot
products; a retirement barrier precedes overwriting the window. The activation
cache stays intact until the CTA completes. Scalar BF16 loads support naturally
aligned offset views without stronger vector-alignment assumptions.

Maximum dynamic shared memory is 98,304 bytes (96 KiB): an 80 KiB activation
cache and 16 KiB weight window. There are no other shared arrays in the score
kernel. All larger-shape shared-memory opt-ins are initialized at the first
call on each device, including when that first call uses m=1. Later captures
of m=4..8 do not change function attributes or allocate storage.

This deliberately differs from K8-02's one-expert full FP32 weight-row cache,
which rereads x for each expert, and K8-06's two-expert asynchronous ping-pong
input/weight tiles. K8-08 uses a full-lifetime activation cache, four experts,
and one synchronous weight window. It does not use cp.async or double buffers.
At source level, every call reads 3.75 MiB of weights. At m=8 the activation
loads total 7.5 MiB, compared with 30 MiB for K8-02's one-expert layout and
15 MiB for K8-06's two-expert layout. These are logical load counts, not measured
DRAM traffic: caches, occupancy, and barrier costs decide actual performance.
The higher shared footprint and 1,024-thread m=8 block are explicit tradeoffs.

A second kernel performs deterministic GPU top-six selection for each token.
Both launches use the supplied stream. No host copy, host synchronization,
allocation, free, or function-attribute change occurs on the warmed path.

## Numerical and ownership invariants

Every dot-product lane processes dimension lane+32*j for ascending j, with one
FP32 FMA accumulator carried through all three weight phases. Reduction offsets
are exactly 16,8,4,2,1, matching ops.cu::bf16_gemv_k. BF16 shared copies are exact;
there are no split-K partial sums, extra quantization, or reassociated sums.

The fixed oracle evaluates sqrt(softplus), score+bias comparisons, and selected
score normalization in double after its FP32 logit. math.hpp keeps that same
expression and the z>20 branch. No approximate-score filter can misorder
underflowed float scores, and no epsilon tie rule is introduced. Selection uses
higher score then lower expert ID. Both the winning score and its ID are marked
removed, so real negative-infinity keys still beat the sentinel. Unbiased
scores, including zero, have exactly one producer. Six selected scores are
summed in selection order before converting output weights to float.

Scratch is one K8-only 24 KiB score buffer per device, allocated at that device's
first eager call for maximum m=8 and retained for process lifetime. The runtime
guarantees that calls of the same task on one device never overlap. Different
tasks may run concurrently and have independent scratch. A mutex protects the
per-device registration; it does not synchronize with the GPU.

## Validation completed without a GPU

- CUDA 12.8, C++17, sm_86/sm_89/sm_120 compile checks pass
- Resource reports have no spills; sm_89 score kernels use 36..37 registers,
  the selector uses 37, and the maximum dynamic shared allocation is 96 KiB
- CPU model: 1,070 routing cases; 18,432 bit-exact FP32 logits; maximum relative
  routing-weight difference 0 against an independent partial-sort oracle
- All m=1..8; unique activation/weight/score ownership; publication and retirement
  phases; short weight tail; exact lane visit order; cross-phase cancellation
- Exact and float-collapsed near ties, negative-infinity bias, float-underflow
  ordering, softplus threshold neighbors, large positive and negative logits
- Fixed acceptance test plus reference ops build/link for sm_89, unchanged
- Optional graph regression harness builds/links for sm_89; it specifies m=1
  warmup followed by all larger captures on a non-default stream, two graph
  nodes, 24 changing-input replays, ties and near ties

Run the CPU model from the repository root:

    g++ -std=c++17 -O3 -march=native src/ds41/kernels/k8/host_semantics.cpp -o /tmp/k8_semantics
    /tmp/k8_semantics

The graph harness build command is at the top of graph_validation.cu. It is
separate from the fixed acceptance test. GPU acceptance, CUDA libm parity at
extremely close scores, actual capture/replay, sanitizer checks, and timings
have not run. No GPU pass or speedup is claimed; the GPU queue is offline.
