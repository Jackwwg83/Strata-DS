# K7-07: move collapse into the producer launch

Scheduling-only ablation of repaired K7-02 at
`5dc13d8d983c3abad1f9853b2dfa6731adc77dde`, applied on feature parent
`90dca49d2e90736c3c6fcc1e156e4bed34dd15b3`.

## Schedule

- First launch: 212 one-warp CTAs. CTAs 0..19 collapse 256 features each
  for all tokens; CTAs 20..211 retain the original 24 rows x 8 warp producers
- Second launch: one warp CTA per token, coefficients and Sinkhorn only
- Normally two launches. Collapse uses the old `pre_in`, so it does not depend
  on the new coefficients. The scheduling change permits collapse and weight
  work to overlap, without an in-kernel global barrier or completion counter
- The dot producers retain all 80 terms of every original stride-256 FMA chain;
  norm producers retain all 20 terms of each original stride-1024 FMA chain.
  Warp shuffle trees, ascending warp-total sums, coefficient normalization,
  scale/bias, and all 20 Sinkhorn iterations are unchanged from repaired K7-02
- Each collapse output retains the original j=0,1,2,3 FP32 chain, then exactly
  one BF16 rounding. Every output has one producer owner

This is distinct from K7-05's last-CTA completion protocol and K7-06's register
prefetch. No speedup is claimed before a GPU result. In particular the larger
producer argument set and collapse branch increase the m=8 register allocation
from K7-02's 40 to 54 on sm89, which may offset reduced final-stage scheduling.

## Aliasing and scratch

The fixed header does not forbid y/input aliasing. Concurrent producer writes
therefore cannot blindly go to y. The launcher checks the numeric byte ranges
of y against all five inputs: x, fn, scale, base, and pre_in. This is host
pointer arithmetic only, with no dereference or device-to-host transfer.

If y is disjoint from those inputs, producers write it directly. Otherwise they
write an 81,920-byte BF16 array in the private workspace. The coefficient launch
then consumes the remaining scale/base inputs, and a third stream-ordered copy
kernel publishes y. This handles exact and shifted x/y overlap, including
cross-token overlap, without overwriting data that another CTA still reads.
It also handles y overlapping fn, scale, base, or pre_in. As in the reference,
the logically separate output tensors must be able to hold their results;
o claim is made for mutually conflicting output/output aliases. General
coefficient-output/input aliasing is outside this scheduling change.

K7 owns one fixed 89,088-byte allocation per CUDA device, initialized on the
first eager call and retained for process lifetime. It covers every m=1..8
and both dispatch paths, even when warmup uses only the disjoint m=1 path.
Same-device K7 calls/replays must not overlap, as guaranteed by the engine.
Other tasks use unrelated allocations. There are no warm-call allocations,
frees, host copies, host synchronization, counters, or memset resets. Every
launch uses the supplied stream. Graph safety is source-reviewed, not GPU-tested.

## Checks

CPU checks (no CUDA kernel execution):

    g++ -std=c++17 -O3 -ffp-contract=off src/ds41/kernels/k7/check_numerics.cpp -o /tmp/k7-numerics
    /tmp/k7-numerics
    python3 src/ds41/kernels/k7/check_schedule.py

- 360 token cases / 8,640 raw dots from ten input families, all m=1..8;
  dot, norm, pre, post, comb are bitwise equal in the CPU arithmetic model
- Known cancellation regression gives reference=1, defective tiled=0, repaired=1
- 737,280 bitwise collapse values, unique owners, j-order cancellation, and
  96 modeled exact/shifted alias copies
- Exhaustive ordered coordinate checks for 6,144 dot lanes and 1,024 norm lanes
- 4,194,516 tests of the actual host overlap functions, including all five
  input ranges, adjacency, contained overlap, and high addresses
- Producer dot/norm code, coefficient/Sinkhorn helpers, and final reductions
  match the repaired K7-02 source byte for byte

CUDA 12.8 C++17 compiles for sm86, sm89, and sm120. On sm89: at most 54 registers,
100 bytes static shared memory, zero stack/spill bytes. Fixed acceptance and
optional regression compile and link. They have not run; no correctness,
performance, graph replay, or compute-sanitizer pass is claimed.

Optional device regression:

    nvcc -std=c++17 -O3 -arch=sm_89 -Iinclude -Isrc \
      src/ds41/kernels/k7/check_graph_alias.cu src/ds41/kernels/k7_hc.cu \
      src/ds41/ops.cu -o /tmp/k7-graph-alias
    /tmp/k7-graph-alias

It starts with m=1, then covers every m=1..8, direct y plus seven input-alias
arrangements, random and cancellation inputs, non-default-stream global graph
capture, two replays with restored aliased inputs, bitwise eager/replay equality,
and reference tolerances. Expected graph kernel counts are two and three.
It iterates visible devices. Run under compute-sanitizer when a GPU is available.
The fixed acceptance header and test remain unchanged.
