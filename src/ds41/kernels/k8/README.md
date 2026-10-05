# K8-11: exact-order register-prefetch ablation

Base: feature `94274899eb0b46ccd0f67276f2c823ee073d918c`, after the K8-07
merge. The base's runtime K8 source is byte-identical to the measured K8-07
control `b45ddd889c37a5bb27811396b64883eb67b2935f`.

That control scored 19.84 us (m1 16.38, m8 27.65) on the RTX 4090 + i9-14900K
queue; its numerical cases and explicit m8 graph check passed:
https://github.com/Jackwwg83/Strata-DS/issues/5#issuecomment-5997241853
Those are control results, not measurements of this candidate. The reviewer
requires at least 3% lower score with no m1 regression.

## The isolated change

`register_prefetch.hpp` loads one input/weight pair before starting a lane's
FP32 chain. Each iteration loads the next pair before consuming the current
pair. A separate final drain consumes step 159 without fetching step 160.
The helper is shared by the actual CUDA scorer and its portable CPU tests.

- Decode loads BF16 input and BF16 weight, converting both exactly to FP32
- Multi-token scoring keeps the original one-time shared FP32 weight staging;
  each token warp loads BF16 input plus the corresponding shared FP32 weight
- Every lane still consumes dimensions `lane + 32*j`, j=0..159, in that order
- One `__fmaf_rn` accumulator and the original 16,8,4,2,1 shuffle/add tree remain
- The unroll factor remains eight; there are no separate partial sums

No other runtime behavior changes. The warp-only top-six selector, double
nonlinear scoring/comparisons/normalization, cooperative shared staging, block
sizes, dispatch, two supplied-stream launches, and scratch lifetime are exactly
the K8-07 control. The public header and fixed acceptance tests are untouched.

The source audit reverses only the two loop replacements and their new support
definitions/include, then requires byte equality with the exact control.
`math.hpp`, `host_semantics.cpp` and `graph_validation.cu` are also unchanged.

## Evidence from generated code

CUDA 12.8.93, C++17, O3, precise division/square root and subnormals enabled:

- Inspected the control PTX before editing: both scoring loops repeated
  load/convert/FMA for the current step, with no explicit next-input load-ahead
- Candidate decode PTX loads future input and weight values before current
  FMAs. Candidate tile PTX retains future BF16-input loads, but the compiler
  sinks shared-weight reads near their consuming FMAs. Therefore this is not a
  claim that both members of every shared-path pair remain prefetched in PTX
- All eight scorer specializations have distinct normalized PTX and distinct
  raw cubin `.text` bytes versus control on sm86, sm89 and sm120
- The selector has identical normalized PTX and byte-identical cubin `.text`
  on all three architectures
- sm86/sm89 registers: decode 27 -> 29; every tile specialization 30 -> 28;
  selector stays 80
- sm120 registers: decode 28 -> 30; every tile specialization stays 30;
  selector stays 80
- Zero stack/spill bytes in every kernel on all three architectures
- Shared memory remains 20,480 bytes for tile scoring and zero for decode and
  selection. Tile scoring retains its one staging barrier; selector has none
- sm89 scorer text sizes: decode 6,400 -> 7,168 bytes; tile 6,528 -> 7,040 bytes

Machine text differs, so this is a distinct compiled ablation. Register counts
and PTX load order are evidence of generated-code changes, not latency proof.
Larger scorer code and the peeled/drain loop control may offset load-ahead.

## Numerical and lifetime invariants

The reference `ops.cu::bf16_gemv_k` uses 160 ascending lane-stride FP32 FMA terms
then the same warp tree. Early loads do not change any arithmetic dependency.
BF16-to-FP32 conversion and shared FP32 storage are exact. The final legal
indices are lane+32*159, at most 5119; there is no speculative out-of-row read.

Scoring follows the fixed oracle's double sqrt(softplus), score+bias comparison,
lower-ID ties, and left-to-right selected-score normalization. No narrowed math,
score approximation, seed-dependent behavior, or reduction reassociation is added.

Rule 7 is unchanged: K8 owns a 24 KiB maximum-m8 score array per device,
allocated on its first eager call and retained for process lifetime. Warm calls
have no device allocation/free, host transfer or device/stream synchronization.
The engine guarantees nonoverlapping calls/replays of K8 on the same device;
other task kernels use independent scratch. Both launches use the supplied
stream. Candidate graph execution still needs its own queue result.

## Completed checks

- CUDA 12.8 compilation for sm86, sm89 and sm120
- Fixed acceptance and existing optional graph test compile for all three
  architectures and link for sm89; neither executable was run here
- Existing CPU model: 2,503 routing cases, 13,824 bit-exact FP32 logits, zero
  relative weight error, covering every m, ties, near-ties, underflow,
  thresholds, extreme biases and permutations
- New actual-helper model: 27,648 bit-exact direct/shared FP32 logit checks for
  m=1..8, plus 36 routing comparisons
- Schedule trace for every lane: exactly 160 ordered loads and 160 ordered
  consumes, next-pair load before current consume, exactly one final drain
- 864 cancellation/drain fixtures, including all lanes and unroll boundaries
- 1,024 randomized exact-BF16 exponent/sign stress dot products
- Exact-control source audit and all-three-architecture PTX/cubin comparisons

Run the portable checks from the repository root:

    g++ -std=c++17 -O3 -march=native src/ds41/kernels/k8/host_semantics.cpp -o /tmp/k8_semantics
    /tmp/k8_semantics
    g++ -std=c++17 -O3 -march=native src/ds41/kernels/k8/check_register_prefetch.cpp -o /tmp/k8_prefetch
    /tmp/k8_prefetch

`graph_validation.cu` remains the control's optional non-default-stream test:
all m values, m1 warmup before larger captures, exactly two graph nodes, 24
changing-input replays, exact ties and near-ties. It was compile/link checked,
not executed. The fixed queue overlays the current acceptance sources and
includes the m8 graph check.

No GPU is available in this workspace. Candidate GPU numerical parity, graph
capture/replay, sanitizer results and speed remain pending. Keep the first
published head unchanged until its exact-SHA queue result arrives.
