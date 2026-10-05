# K8-10: two-token microtiles with a shared activation cache

Only K8 CI-FILES change. The branch starts at feature/ds41
`90dca49d2e90736c3c6fcc1e156e4bed34dd15b3`. The design is an ablation of K8-08
`d283c5b85b35905e1e120f2a84229ddac4013f3b`; its double math, selection kernel,
FP32 lane FMA order and warp reduction are retained. The fixed header,
acceptance test and reference ops remain unchanged. All legal m=1..8 work.

## One design: fixed two-token groups

A CTA owns four experts and two token slots, with one warp per expert/token
pair: 256 threads. The score grid is (96, ceil(m/2)). All CTAs use the same
kernel and 36 KiB dynamic shared allocation: 20 KiB for two complete BF16
activation rows, plus the original 16 KiB four-expert by 2,048-column BF16
weight window. No dynamic-shared opt-in or shape-dependent kernel setup is
needed because the allocation stays below 48 KiB.

The cache producer copies only the one or two real activation rows in its
CTA. The second token in an odd-m tail has no activation load, cache read,
FMA or score store. Its warps still reach every publication/retirement barrier
and the full-mask shuffle reduction. No warp exits before a CTA barrier.
Each active token/expert score has one lane-zero writer; every input element
has one cache producer per corresponding CTA. Naturally aligned BF16 offset
views are supported without vector-alignment assumptions.

The streamed weight phases are unchanged: columns 0..2047, 2048..4095 and
4096..5119. Each phase has one publication and one retirement barrier. The
short phase never loads or reads its 1,024 columns of unused padding. Every
lane carries its FP32 FMA accumulator through all phases in the order
lane+32*j. The final reduction offsets are 16,8,4,2,1, exactly as in
ops.cu::bf16_gemv_k. There is no split-K reassociation or new quantization.

The original double sqrt(softplus), biased comparisons, lower-ID tie rule,
ID/sentinel removal and selection-order normalization remain unchanged.
In particular, zero and double-only nonzero scores are retained; no FP32
underflow filter or epsilon tie rule is introduced.

## Static traffic and resource tradeoff

These are source-level BF16 load counts, not measured DRAM traffic.

| m | Token groups | K8-10 weight reads | Activation reads, both designs |
|---|---|---|---|
| 1 | 1 | 3.75 MiB | 0.9375 MiB |
| 2 | 1 | 3.75 MiB | 1.875 MiB |
| 3 | 2 | 7.5 MiB | 2.8125 MiB |
| 4 | 2 | 7.5 MiB | 3.75 MiB |
| 5 | 3 | 11.25 MiB | 4.6875 MiB |
| 6 | 3 | 11.25 MiB | 5.625 MiB |
| 7 | 4 | 15 MiB | 6.5625 MiB |
| 8 | 4 | 15 MiB | 7.5 MiB |

K8-08 reads 3.75 MiB of weights for every m. Microtiling preserves total
activation traffic (96 reads per activation element) but rereads weights
once per token group. At m=8 the total logical input traffic doubles from
11.25 to 22.5 MiB. The score launch grows from 96 to 384 CTAs, while each CTA
falls from 1,024 to 256 threads and from 96 to 36 KiB shared memory. Thus the
shared-memory footprint allows more than one CTA to fit in a 99 KiB budget;
actual residency and speed must be measured. At m=1, the fixed tile instead
uses 256 rather than 128 threads and 36 rather than 26 KiB shared memory,
with half its compute warps idle. There is no separate m=1 fallback in this
ablation.

CUDA 12.8 ptxas reports 36 registers per score thread on sm_86/sm_89 and 40 on
sm_120, with no stack or spill traffic. The selector uses 37/37/38 registers
and 112 bytes static shared memory. For sm_89 the score CTA's unrounded
register count is 9,216 (36*256), versus 37,888 for the old m=8
1,024-thread layout (37 registers/thread, rebuilt at its exact SHA); hardware allocation rounding
and achieved occupancy are not measured. The maximum allocation is 36 KiB,
well under the 99 KiB task limit.

## Rule 7 and validation

The two GPU launches use only the supplied stream. Scratch is one fixed
K8-only 24 KiB score buffer per device, allocated on its first eager call for
maximum m=8 and retained for process lifetime. Warm calls do not allocate,
free, change function attributes, copy to/from the host or synchronize the
GPU with the host. A mutex protects the device registry. The engine guarantees
same-task calls and graph replays on one device do not overlap; other tasks
use independent scratch and may run on other streams.

Completed without a GPU:

- CUDA 12.8 C++17 compile checks for sm_86, sm_89 and sm_120
- CPU model: 1,094 routing cases and 27,648 bit-exact FP32 logits, with maximum
  relative routing-weight difference 0 against an independent partial-sort oracle
- Every m=1..8, all odd token tails, input extents, activation and weight
  producers, score/ID/selected-score/output-weight owners, six CTA barriers,
  full-warp shuffle participation and the short streamed-weight tail
- Random and cross-phase cancellation inputs with different token-row scales;
  exact/near ties, negative-infinity bias, threshold neighbors, double-only
  underflow scores and large finite logits
- Fixed acceptance test and reference ops compile/link for sm_89, unchanged
- Optional graph harness compiles/links for sm_89: eager m1 warmup, all m=1..8,
  exact-extent offset inputs, guarded outputs, global non-default-stream
  capture, two graph nodes and 48 changing-input replays compared bitwise to
  eager results, plus exact and float-collapsed near ties

Run the CPU model from the repository root:

    g++ -std=c++17 -O3 -march=native src/ds41/kernels/k8/host_semantics.cpp -o /tmp/k8_semantics
    /tmp/k8_semantics

The optional graph harness build command is at the top of graph_validation.cu.
It is separate from the fixed acceptance test. GPU execution, CUDA libm parity
at extremely close scores, actual capture/replay, sanitizer checks, occupancy
and timing remain untested because the queue is down. Graph safety is only
source-reviewed. No GPU pass, merge recommendation or speedup is claimed.
