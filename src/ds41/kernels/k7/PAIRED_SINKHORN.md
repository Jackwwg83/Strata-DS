# K7-08: two independent token subgroups per Sinkhorn warp

A scheduling ablation of repaired K7-02
`5dc13d8d983c3abad1f9853b2dfa6731adc77dde`, on feature parent
`90dca49d2e90736c3c6fcc1e156e4bed34dd15b3`.

## What changes

The old finish warp calculated the same sixteen comb values in both halfwarps.
This version assigns one token to lanes 0..15 and its neighbor to lanes 16..31.
Each lane owns one comb value. Lanes 0..3 within each subgroup also own that
token's pre/post outputs. Indexed shuffles use width 16 and the disjoint masks
`0x0000ffff` and `0xffff0000`. Every named lane executes the same shuffle with
its subgroup's mask. No inactive odd-tail lane is named by the active subgroup.

A finish CTA now owns one 256-feature tile of up to two tokens. All threads
collapse token zero, then token one when present. The grid is
`(5120/256, (m+1)/2)`. The token guard precedes every tail access. Only tile zero
reduces scratch: fifty threads own two sets of twenty-four mixes plus RMS.
The block barrier makes both valid tokens' shared values visible before the
first warp reads them. No thread reads the absent token's shared values.

The producer, scratch allocator and producer launch are unchanged from the
repaired control. Each dot keeps its original eighty-term stride-256 FMA
chain and warp shuffle tree; RMS keeps its stride-1024 FMA chains. The finish
stage still sums eight dot warp totals and thirty-two RMS warp totals in
reference order, starting from positive zero.

Normalization retains a separate rounded multiply by reciprocal RMS before
scale/bias. Row maxima and all four-entry sums remain in scalar reference
order. There is one initial row softmax with additive epsilon, one initial
column normalization, and nineteen row/column pairs with denominator epsilon.
All twenty steps, `1e-6` epsilons, FP32 operations, ordinary `expf`, divisions,
and BF16 collapse rounding are retained. There is no fast-math flag, approximate
sigmoid, explicit approximate exponential, or approximate division.

## Work and resources, not a measured speedup

- Comb lane work falls from 32 duplicated values per token to 16 unique values.
  Pre/post and all dot/RMS arithmetic still have the same useful work.
- Finish CTAs fall from `20*m` to `20*ceil(m/2)`: m1 stays 20; m3 goes 60 to 40;
  m8 goes 160 to 80. Each paired CTA does up to twice the collapse work. The
  useful collapse arithmetic and number of output elements do not change.
- Producer CTAs stay 192. Both versions make exactly two kernel launches.
- Fixed K7 scratch stays 7,168 bytes per device for all legal m=1..8. First
  eager use allocates the maximum size, and subsequent calls only look it up.
  There is no warm allocation/free, host copy/wait, scratch clear, atomic,
  cross-CTA spin, or default-stream kernel launch. Other tasks use separate
  storage. This relies on the stated same-device K7 nonoverlap guarantee.
- CUDA 12.8 ptxas reports 38 finish registers and 200 shared bytes for sm86,
  sm89 and sm120, with no stack or spills. The repaired control uses 38
  registers and 100 shared bytes on sm89. Producer registers are 18..40,
  with no shared memory or spills. All blocks are below the 99 KiB limit.

These are source/compiler facts. Pairing changes the CTA scheduling and may
help or hurt latency. No GPU, acceptance, graph, sanitizer or timing result
has been measured here; the replacement queue remains the decision point.

## Checks performed

- CUDA 12.8 C++17 kernel compilation for sm86, sm89 and sm120; all eight
  producer specializations and the finish kernel have no spills.
- Fixed acceptance source and optional graph regression built and linked for
  sm89, but not executed. Optional graph source also compiles for sm86/120.
- CPU arithmetic model: 96 batches and 432 tokens, all m=1..8, with bitwise
  dot, RMS, pre/post/comb and BF16 collapse equality. Cases include zero,
  signs, sparse inputs, wide dynamic ranges and subnormals.
- Original cancellation regression: weights at offsets 0, 1024 and 1280
  are `2^25`, `-2^25`, and `1`. Reference and this candidate give dot=1;
  rejected contiguous tiles give zero. Additional cases move cancellation
  across every row and original lane/step positions.
- 10,000 extra paired coefficient cases independently compare the scalar
  reference with subgroup operations, including distinct neighboring tokens,
  ties, saturated and underflowing logits. All outputs match bitwise in the
  CPU model. Inactive upper halves are poisoned and remain unread/unwritten.
- Static source and ownership checks cover every legal m, subgroup masks,
  pre/post/comb exclusive writers, every collapse element, every scratch
  reader and writer, the shared barrier and odd-tail bounds. Producer and
  allocator text exactly match the repaired control.
- PTX audit confirms disjoint masks, width-16 indexed shuffles, separate
  rounded RMS multiplies and rounded division. Standard `expf` lowering can
  contain `ex2.approx` internally; the source uses the unchanged full `expf`
  implementation, without `__expf` or fast math.

CPU libm equality is an arithmetic-order check, not CUDA libm or device proof.
The optional `check_graph.cu` covers eager m1 warmup, non-default Global
capture for all m, changed inputs, cancellation, two bitwise-equal replays per
call, odd-tail/output canaries and separate-stream traffic. Its traffic
kernel is not an engine-level K7/K8 overlap test. Multi-device checks run when
hardware is available. None of these GPU checks has run in this workspace.

## Reproduce

From the repository root:

```sh
g++ -std=c++17 -O2 -ffp-contract=off src/ds41/kernels/k7/check_numerics.cpp -o /tmp/k7-paired-model
/tmp/k7-paired-model
python3 src/ds41/kernels/k7/check_structure.py
# Compile the kernel with nvcc -std=c++17 -O3 for sm_86, sm_89 and sm_120.
# Generate sm89 PTX with -ptx, plus resource logs with -Xptxas=-v, then:
python3 src/ds41/kernels/k7/check_ptx.py KERNEL.ptx SM86.log SM89.log SM120.log
# On an authorized GPU, compile/link check_graph.cu, k7_hc.cu and ops.cu;
# run the resulting optional harness, then compute-sanitizer memcheck,
# racecheck and synccheck separately. Also run the fixed acceptance test.
```

The structural comparison needs the repaired control commit available locally.
Only K7 CI-FILES are changed; the fixed header and acceptance test are untouched.
