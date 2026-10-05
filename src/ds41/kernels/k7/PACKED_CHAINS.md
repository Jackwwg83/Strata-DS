# K7-10: four adjacent reference lanes per physical lane

This is a load-issue ablation of repaired K7-02 at
`5dc13d8d983c3abad1f9853b2dfa6731adc77dde`. It has not run on a GPU. The queue is
down; there is no speed, numerical CUDA pass, or graph-execution claim.

## Exact arithmetic

Each physical lane owns four adjacent original dot lanes. Eight physical lanes
represent one original 32-lane warp; a 32-thread producer contains four such
groups. There are 2 by 24 producer CTAs. Every component retains its original
80-term, stride-256 FP32 FMA chain. Each loaded weight component is reused by all
m tokens. There is no contiguous-K partial sum or reassociation of FMA chains.

For each component, width-8 shuffle offsets 4, 2, 1 reproduce original offsets
16, 8, 4. Only afterward are components combined as `(x + z) + (y + w)`, which
reproduces original offsets 2, 1. Combining adjacent components first, or using
`((x + y) + z) + w`, is not equivalent. The final stage still starts from +0 and
adds eight original warp totals in order.

The first four weight rows keep the repaired reference norm ownership:
step modulo 4 selects one original stride-1024 lane chain per component. Its
original 32-lane tree and ordered 32 warp totals are preserved. The maximum
7,168-byte per-device, task-private workspace, scalar fallback, RMS scaling,
Sinkhorn, collapse, and second-stage kernel are unchanged from repaired K7-02.
Only producer mapping/load width and host alignment dispatch differ.

The aligned path requires fn aligned to 16 bytes and x aligned to 8 bytes.
All row, token, and step strides preserve these alignments. Naturally aligned
FP32/BF16 offset views use repaired K7-02's original scalar producer. This
fallback adds no staging, warm allocation, copy, or host synchronization.

## Expected tradeoff, not a measured speedup

CUDA 12.8 PTX for all m and sm86/89/120 contains one `ld.global.nc.v4.f32` weight
load and m `ld.global.nc.v2.u32` activation loads per loop step. Analytical
producer warp-load issues are 192*80*(1+m) for scalar and 48*80*(1+m) for packed:
30,720 to 7,680 for m1 and 138,240 to 34,560 for m8. This is 4x fewer PTX warp-load
issues, with unchanged useful input bytes. These are not measured SASS issue
counts or DRAM traffic. The scalar PTX has duplicated paths on sm86/89; only one
path executes for each row/step.

The cost is 192 to 48 one-warp CTAs, 6,144 to 1,536 physical threads, and four
dot plus four norm accumulator components per token per thread. On a 128-SM
RTX 4090, a 48-CTA producer cannot occupy all SMs at once. Wider transactions and
lower issue overhead therefore compete with substantially less parallelism;
this candidate may lose. It is distinct from prefetch, row grouping, finish
fusion, and constant-cache ablations. Keep or discard only after exact-SHA GPU
results, never from these analytical counts alone.

ptxas registers per producer thread, m1 through m8:

| Architecture | Original scalar | Packed |
|---|---|---|
| sm86 | 18,22,24,27,31,37,36,40 | 30,38,47,56,64,72,80,94 |
| sm89 | 18,22,24,27,31,37,36,40 | 30,38,47,56,64,72,80,94 |
| sm120 | 17,20,23,29,32,35,38,39 | 30,37,40,48,63,72,80,94 |

All kernels have zero stack and spills. Producers use zero shared bytes;
the unchanged finish uses 100 shared bytes and 38 registers. No new dependency
or architecture newer than sm86 is required.

## Validation

- `python src/ds41/kernels/k7/check_mapping.py`: exact symbolic 32-lane tree,
  ordered eight/32 warp-total trees, every lane chain, unique scratch owners,
  vector bounds, every legal alignment residue, and inherited-source parity
- Compile/run `check_numerics.cpp` with C++17, `-O2 -ffp-contract=off`: 432 token
  cases and 10,368 raw dots, m1 through m8, 12 input families; raw dot, norm,
  and coefficients all bitwise equal to independent reference models
- Original cancellation case remains reference=1, old tiled=0, packed=1
- Two added cancellation families reject adjacent-component-first and
  shuffle-stage reassociation; both give the reference zero
- CUDA12.8 C++17 compilation for sm86/89/120; ptxas stack/spill/shared checks
- `check_codegen.py <evidence-directory>` checks generated PTX and ptxas logs
- Unmodified fixed acceptance and optional `check_packed.cu` are compiled and
  linked, not executed. The optional harness includes the candidate translation
  unit directly: 32 aligned raw-dot/norm comparisons and 512 all-m/alignment/
  output/non-default Global-capture cases, with two bitwise replays per case.
  It compares actual packed producers with the actual scalar fallback. Harness
  copies, allocations, and host waits are test-only operations.

Actual CUDA parity, timings, graph capture/replay, sanitizer checks, and
cross-task overlap remain pending. The reviewer owns the fixed graph test and
merge-time K7/K8 overlap check. Fixed headers/tests are unchanged in this patch.
