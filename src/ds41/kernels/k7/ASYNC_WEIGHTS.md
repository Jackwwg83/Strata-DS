# K7-11: asynchronous weight tiles, unchanged FP32 chains

Branch: `task/K7/dots-K7-11`  
Base: `be8c969a1f1b7bf88d8a64ef1b3e935dcc2f376a`  
Target: CUDA 12.8, C++17, sm86/sm89/sm120

## Grounding and scope

The current 3060 control is score 18.43 us (m1 15.36 us, m8 24.58 us),
according to the reviewer-maintained `ds41/tasks/PRIORITIES.md` at the base.
The reviewer's 4090 profile identifies hc_partials + hc_finish as roughly five
times the bytes-read bound. These are control/profile observations, not K7-11
measurements. K7-11 has no GPU result yet.

The merged K7-06 producer already reads every FP32 weight once and reuses it
for all tokens. This candidate does **not** claim to reduce those bytes.
It changes the issue granularity and lookahead of the weight reads:

- Keep all 192 one-warp producer CTAs for every m=1..8
- Cooperatively load 16 reference steps of FP32 weights into each 2-KiB tile
- Double-buffer two tiles, using 16-byte `cp.async.cg.shared.global` copies
- Issue 20 warp-level asynchronous copy instructions across 80 steps, versus
  80 scalar warp-level global weight-load instructions in the control
- Preserve the existing two-step scalar activation prefetch while reading
  staged FP32 weights from shared memory
- Preserve the two stream-ordered launches, finish kernel, coefficients,
  collapse, and 7,168-byte per-device K7-private workspace
- Select the unchanged scalar producer when fn is not 16-byte aligned

The hypothesis is more weight requests in flight and less global-load issue
pressure at low m, without reducing the producer grid. The CG copies also
bypass L1; any effect on activation caching or throughput is unmeasured.
The tradeoffs are 4 KiB shared per producer, extra warp barriers/shared reads,
and additional registers. Only the fixed GPU queue can establish a win.

All nine prior remote heads (K7-02 through K7-10) were inspected before choosing
the unused K7-11 branch. They cover stream-ordered scalar partials, one-CTA
fusion, synchronous two-row vector tiles, last-CTA completion, two-step scalar
prefetch, early collapse, paired-token Sinkhorn, four-token microtiles, and
packed adjacent lanes. K7-11 differs from K7-04's synchronous row-sharing and
K7-10's 48-CTA packed-lane path. It does not port K15-01's one-CTA-per-token
prefill strategy into a decode grid with only 1..8 CTAs. K15-01/02 reinforce the
need to retain the exact FP32 chains; no prefill code or other task is changed.

## Arithmetic and ownership

Every original lane still visits column `warp*32 + lane + 256*step`, for
step 0..79, in order. Accumulators survive all five tiles. Every tile begins at
a multiple of four, so the norm's step-mod-four partition is unchanged:
1024 original norm lanes each retain the same 20-term stride-1024 FMA chain.
The shuffle tree and sequential original-warp totals are unchanged.

For copy number 0..3 and physical lane 0..31, shared offset is
`4*lane + 128*copy`. The corresponding global column is
`32*warp + 256*(16*tile + offset/32) + offset%32`.
Each copy covers four adjacent FP32 values in one 32-column reference stripe.
All 512 tile values have one writer; every one of the 24*20480 weights is read
exactly once across the producer grid. All source and destination copy offsets
are multiples of 16 bytes, and the final copy ends inside step 79.
The host's fn alignment check is the only added dispatch condition; x still
uses scalar BF16 loads and supports ordinary BF16 alignment.

The scalar fallback and the workspace-through-producer region are checked
byte-for-byte against the base. The row/column sums, separate mix-times-RMS
rounding, sigmoid, 20 Sinkhorn iterations and epsilons, and incoming-pre BF16
collapse are also checked byte-for-byte. No TF32/BF16 conversion of fn,
reassociated dot sum, approximate math flag, tolerance change, or test removal.

## Asynchronous lifetime proof

1. Every lane issues its part of tiles 0 and 1 into different buffers, committing
   one group per tile
2. Before consuming a tile, every lane waits until at most one newer group
   remains. A full-warp barrier then publishes all 32 lanes' completed copies
   to the consumers, including copies produced by a different lane
3. The second full-warp barrier is after consumption. No lane can overwrite
   the current buffer for tile+2 until all 32 lanes have finished reading it
4. The last tile uses wait_group 0, because there is no newer group to leave
   pending. There is no tile 5, speculative copy, zero-fill, or overread
5. Shared tiles are CTA-local. Global partials retain exclusive writers and
   the second launch consumes them only after the first launch completes

The shared host/CUDA schedule is `async_weights.hpp`; the FP32 consumption
body remains `register_prefetch.hpp`. This is source/order evidence, not a
substitute for GPU racecheck or execution.

PTX semantics used here are documented by NVIDIA in
[cp.async and async-group completion](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async)
and the
[warp synchronization programming guidance](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/).

## Compiled resources

PTXAS reports zero stack, spill stores and spill loads for every specialization.
The aligned producer uses 4,096 bytes shared, the scalar fallback uses none,
and finish uses 100 bytes, all below 99 KiB per block. Producer register counts
for m=1..8 are:

- sm86/sm89 async: 33, 38, 39, 51, 54, 60, 64, 77
- sm120 async: 40, 40, 40, 40, 48, 60, 72, 64
- sm86/sm89 scalar fallback: 25, 29, 35, 39, 46, 52, 56, 68
- sm120 scalar fallback: 27, 40, 40, 40, 48, 56, 63, 64
- Finish: 38 registers on all three architectures

These are compiler resource reports, not measured occupancy or latency.

## Graph and scratch lifetime

The existing K7-private workspace remains maximum-size m=8, allocated once
per device during the required eager first call and retained for process
lifetime. Its lookup is unchanged. K7 invocations/replays on the same device
must not overlap, as the engine guarantees; other tasks have separate scratch.
The hot path has no allocation/free, copy to/from host, host synchronization,
memset, device-result read, atomic, completion counter, or global spin barrier.
Both runtime kernel launches use the supplied stream. No first-use CUDA
function attribute, additional persistent state or architecture query is needed.

## Validation and limits

The machine-readable `validation.json` records the final source and fixed-file
hashes, compiler resources and binary/artifact hashes. `SHA256SUMS` identifies
all deliverable source/report files except itself.

- Host C++17 model with UBSan: 112 batches, 504 token cases, every m=1..8
- Bitwise raw dot/norm and host coefficient parity with the reference model,
  plus bitwise equality between the async model and scalar fallback
- Signed/zero/sparse/tiny/large/wide-exponent/subnormal inputs; original
  cancellation regression; pair-boundary, tile-boundary and final-drain cases
- Bounds/ownership for each weight, shared tile, original norm input, scratch
  writer and collapse output; only-float-aligned fn offsets select fallback
- Five actual schedule mutations rejected: missing wait, missing publication
  barrier, early reuse, wrong final drain and a tile overread
- All three architectures compile and link the unchanged fixed acceptance test
  and supplemental graph harness, with native matching cubins
- PTX checks cover every m specialization: hardware async-copy instructions,
  both wait depths, full-warp publication/reuse barriers, FP32 shared loads,
  no scalar global weight reads in the new path, and no local-memory accesses
- The existing toolchain has no nvdisasm. cuobjdump emits native-cubin metadata
  but cannot disassemble SASS; no SASS instruction/overlap claim is made
- Both GPU executables return skip 77 on this cloud machine, which has no CUDA
  device. GPU parity, actual capture/replay, sanitizers and latency are untested

The optional graph harness prepares 640 mixed-m eager/replay cases per device:
aligned inputs, fn+1/+2/+3 float offsets, x+1 BF16 offsets with both aligned and
unaligned fn, poisoned outputs, unrelated-task traffic, cancellation, extreme
amplitudes, and sequential device switching if multiple GPUs exist. It retains
the 1e-5 coefficient / 1e-3 y limits. It is not the fixed acceptance test.
An additional raw device dot/norm comparison would be useful GPU-only evidence;
raw equality is currently established by the host arithmetic model only.

To reproduce on the dot cloud toolchain, from the worktree root:

```sh
source ../toolchain/env.sh
bash src/ds41/kernels/k7/build_checks.sh ../build/K7-11
```

That build helper intentionally expects GPU skip 77 on this no-GPU validation
host. On an authorized GPU, run the generated fixed test and supplemental
`check_graph` executable directly, then run the latter under compute-sanitizer
memcheck, racecheck and synccheck. Neither a host-model pass nor a compile pass
is GPU acceptance or a speed measurement.

Only task CI-FILES changed. Header, fixed test, reference ops, CMake, engine and
other kernels remain unchanged. Publication, queue measurement and merging
belong to the parent/reviewer; this candidate is locally committed only.
