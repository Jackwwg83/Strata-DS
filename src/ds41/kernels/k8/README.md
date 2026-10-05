# K8-04: exact-tree cross-CTA split-K

This candidate uses two kernels on the supplied stream:

1. Split each expert's 5120 dimensions by parity between two CTAs. Each CTA
   handles two experts and holds all `m` token accumulators in registers, so
   every BF16 weight element is loaded once and reused across all tokens.
2. Add the two partials, apply the fixed oracle's double-precision nonlinear
   scoring, select six experts with lower-ID tie breaks, and normalize on GPU.

The 384 producer CTAs provide more decode blocks than the reference's 48 or
K8-02's 96. The second launch is the stream-ordered inter-CTA dependency;
there are no atomics, device fences, host copies or host waits in the router.
Scratch is exactly 24 KiB per device, owned by K8, allocated on the first eager
call at the complete interface maximum `m=8`, and retained for process life.
Later eager calls and captures only reuse it. This relies on the engine's
stated same-task, same-device nonoverlap guarantee. Other tasks have separate
scratch and can overlap on other streams.

## Why this split preserves the reference

Ordinary contiguous split-K restarts the FP32 accumulation chains and can
change near-tie expert choices. Here, reference lane `l` still accumulates
`d=l,l+32,...` with explicit round-to-nearest FMA. The even CTA owns original
lanes `0,2,...,30`; the odd CTA owns `1,3,...,31`. Each reduces with logical
half-warp offsets `8,4,2,1`, precisely the reference's original lane offsets
`16,8,4,2`. Their final sum is the original offset-1 addition. Thus both the
FMA chains and every contributing reduction node preserve the reference
order. This argument applies to all legal token counts and all defined
finite-reference logits; it is not dependent on the benchmark's values.

Biased comparisons and normalization use double, matching the fixed
acceptance oracle. In particular, small bias differences must not be rounded
away by forming the biased score in float.

## Comparison with K8-02

Compared against published K8-02 commit
`87b6d90888a995bf17edc039f90e0c8da4b552b6`:

- K8-02 uses 96 four-warp decode CTAs, or 384 CTAs staging a 20 KiB weight row
  for multiple token warps. It computes nonlinear scores in those CTAs.
- K8-04 uses 384 one-warp CTAs for every `m`, two expert subgroups per warp,
  register reuse across tokens, and two partials per token/expert. It has no
  producer shared memory and only 200 bytes in the final kernel.
- Both preserve the reference dot order and have two stream-ordered launches.
- Tradeoffs requiring GPU measurement: K8-04's parity loads use only half the
  elements in a fetched sector, and its nonlinear work is concentrated in the
  final token CTAs. More decode blocks do not prove lower latency.

## Checks

`host_split_model.cpp` is a standalone CPU model of the actual geometry,
subgroup reductions, scratch indexing, comparison reductions and output math.
It checks every `m=1..8`, cancellation across K halves and both parities,
exact ties, below-FP32 near-ties including the sixth/seventh boundary,
softplus threshold neighbors, underflow and large finite scores. A deliberately
unsafe contiguous-split control must change expert IDs on the cancellation
fixture, so random-data agreement alone cannot pass this test.

Build and run from the repository root:

```
g++ -std=c++17 -O3 -march=native -ffp-contract=off \
  src/ds41/kernels/k8/host_split_model.cpp -o /tmp/k8_split_model
/tmp/k8_split_model
```

`graph_split_test.cu` is a separate optional GPU regression. It warms up `m=1`,
then captures each legal `m` on a nondefault stream, verifies exactly two
kernel nodes, and checks three replays with changed inputs. It also checks
exact ties, near-ties and cancellation, and repeats per available device.
The fixed acceptance test and header are unchanged.

```
nvcc -std=c++17 -O3 -arch=sm_89 -Iinclude -Isrc \
  src/ds41/kernels/k8/graph_split_test.cu \
  src/ds41/kernels/k8_router.cu -o k8_split_graph
./k8_split_graph
```

Local evidence: CUDA 12.8 compile passes for sm_86, sm_89 and sm_120; CPU model
passes 20,352 bit-exact logits and 2,075 routing cases with zero modeled
relative weight error. The sm_89 producer uses 32–40 registers; the final
kernel uses 38 registers and 200 bytes shared memory, with no spills.
GPU correctness, CUDA/host libm parity, actual capture/replay and timings have
not been tested here. The GPU queue is offline; no speedup is claimed.
