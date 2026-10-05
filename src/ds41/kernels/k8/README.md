# K8-06: asynchronous BF16 staging and GPU top six

This candidate changes only K8's allowed source and support directory. The fixed
interface, task specification, and acceptance test are unchanged.

## Pipeline and resource choices

For every legal m=1..8, a CTA owns two experts and launches 2*m full warps, one
warp per expert/token pair. The 384-expert grid therefore has 192 CTAs. Each warp
holds one FP32 accumulator per lane. No token is handled on the host.

The 5120-element dot product uses five 1024-element stages. Each stage contains
two BF16 weight rows and m BF16 input rows. Threads cooperatively copy aligned
16-byte chunks with cp.async. A weight chunk has one global-load owner and is
reused by all m token warps; an input chunk is reused by both expert warps.
Sources with only natural 2-byte BF16 alignment use scalar BF16 copies instead.
Row and tile strides preserve the original source alignment. Every legal base
alignment is handled without reading outside the arrays.

The two shared-memory buffers occupy 12,288 through 40,960 bytes as m increases
from one to eight. The maximum is below both the 99 KiB task limit and the 48 KiB
static-shared-memory limit, so no device attribute change is needed. Blocks use
64 through 512 threads. These are dimensional/resource choices, not tuning
results; no GPU is available to measure the occupancy/bandwidth tradeoff. Input
copies are repeated across CTAs and rely on ordinary device caching.

After an initial copy, per-thread wait, and CTA barrier, each iteration issues
copies for the next tile into the other buffer before computing the current
one. A per-thread wait followed by a CTA barrier publishes every producer's
next-tile writes and retires every current-tile reader. Only then may the next
iteration reuse the old buffer. Even threads with no transaction commit/wait;
all threads reach the CTA barrier. Scalar alignment fallbacks obey the same
barriers. Emitted PTX confirms cp.async/commit, ordered FMA work, then wait/barrier.

This differs from K8-02's synchronous full-expert FP32 row staging: both inputs
and weights remain BF16 in a rolling, double-buffered pipeline, with two experts
per CTA and asynchronous overlap on the aligned path.

## Numerics and selection

Each lane visits dimensions lane+32*j in ascending j across tile boundaries.
The FP32 accumulator is carried directly between stages, with explicit
round-to-nearest FP32 FMA. Reduction uses the reference offsets 16,8,4,2,1 and
FP32 addition. BF16 copies preserve the input bits. There are no K-split partial
sums, extra rounding steps, or reassociated dot products.

As required by the fixed oracle, sqrt(softplus(logit)), score+bias comparisons,
and selected-score normalization use double precision after the FP32 logit.
The softplus branch is exactly z>20; no approximate exponential or epsilon tie
rule is added. Nonlinear work is distributed across the score-producing grid.

A second kernel performs six deterministic maximum reductions per token,
comparing higher biased score and then lower expert ID. Each winner is removed
by setting both value=-infinity and ID=384. This also handles valid negative
infinity biased values without repeating a winner. One owner writes each
selected unbiased score, including zero. The denominator is summed in selection
order before producing FP32 weights.

Both kernels launch on the supplied stream. Production code contains no host
copies or device/stream synchronization. Scratch is one K8-only 24 KiB score
allocation per device, allocated for all eight tokens on the first eager call
and retained for process lifetime. Later calls allocate nothing, including when
m increases. Device registration is mutex-protected. This uses the engine's
same-task/same-device non-overlap guarantee; different tasks have independent
storage. Warmup must precede graph capture.

## Checks completed without a GPU

- CUDA 12.8, C++17 compilation passes for sm_86, sm_89, and sm_120
- All three ptxas resource reports show no spill loads/stores; maximum static
  shared memory is 40,960 bytes
- sm_89 pipeline kernels use 32 registers for m=1 and 33 for m=2..8; selection
  uses 37 registers and 104 shared bytes
- CPU model: 13,824 bit-exact reference FP32 logits across m=1..8 and 1,057
  routing comparisons with zero relative weight difference
- The model checks unique copy ownership, once-per-call weight traffic, input
  reuse, ping-pong publication/retirement, and every natural BF16 base alignment
- Routing cases cover exact and near ties, negative-infinity bias, the softplus
  threshold and neighbors, large finite positive logits, underflow, and zeros
- Optional graph/offset regression test compiles for sm_89

Run the portable model from the repository root:

    g++ -std=c++17 -O3 -march=native src/ds41/kernels/k8/host_semantics.cpp -o /tmp/k8_semantics
    /tmp/k8_semantics

`graph_validation.cu` has its GPU build command at the top. It warms up at m=1,
captures all m=1..8 on a non-default stream, checks two graph nodes, verifies
24 changing-input replays, and exercises 24 offset-view combinations plus ties.
It is separate from, and does not modify, the fixed acceptance test.

CPU models and compile/resource checks do not establish GPU execution, CUDA
libm parity at extremely close scores, graph replay, racecheck/sanitizer safety,
queue acceptance, or performance. These remain unverified while the GPU queue
is offline. No GPU timing or speedup claim is made.
