# K7-06: exact-order scalar register prefetch

Control: repaired K7-02, `5dc13d8d983c3abad1f9853b2dfa6731adc77dde`.
Branch base: feature/ds41, `90dca49d2e90736c3c6fcc1e156e4bed34dd15b3`.

## Strategy

Keep the control's 192 warp-only producer CTAs and separate finish/collapse
launch. Within each original reference lane, prime two scalar input/weight
stages. Each iteration loads the next pair, consumes the current pair in order,
and advances the registers. A separate drain consumes steps 78 and 79 without
an out-of-bounds speculative load. Every FP32 weight is read once and reused
for all tokens. The shared header contains the actual pipeline body used by
the CUDA kernel and CPU arithmetic test.

No contiguous-K partial sum, two-row vectorized sharing, last-block completion,
shared-memory staging, or new architecture-specific instruction is introduced.
The experiment trades additional registers for a two-step load lookahead and
half as many producer loop trips. Performance is unmeasured.

## Arithmetic and lifetime

- Each lane retains all 80 stride-256 FMAs, in their original order
- Each original warp uses the same shuffle tree; finish adds the eight warp
  totals starting from +0 in the reference order
- The first four rows retain the original stride-1024 RMS chains and 32 warp
  totals, using exactly the same partition and final sum order as the control
- Finish, explicit FP32 normalization rounding, sigmoid/Sinkhorn, and BF16
  collapse are unchanged from the repaired control
- The 7,168-byte K7-private workspace is allocated once per device on the first
  eager call, at the interface maximum m=8, and retained for process lifetime
- Calls/replays of K7 on the same device must not overlap, as guaranteed by the
  engine. Different tasks do not share this workspace
- Both kernels launch on the supplied stream. The steady-state path has no
  allocation, free, host/device copy, memset, host wait, or device-result read

## Checked on the cloud CPU, CUDA 12.8, 2026-10-05

- C++17 CUDA compilation: sm86, sm89, sm120
- ptxas: zero stack, spill stores, and spill loads for every specialization
- Maximum shared memory: 100 bytes (finish); producer uses zero
- Producer register counts for m=1..8:
  - sm86/sm89: 25, 29, 35, 39, 46, 52, 56, 68
  - sm120: 27, 40, 40, 40, 48, 56, 63, 64
- sm89 control producer counts: 18, 22, 24, 27, 31, 37, 36, 40
- sm89 PTX shows two future FP32 weight loads and 2*m future BF16 loads before
  the current-pair FMAs in every producer loop; no local-memory accesses
- 96 batched CPU cases, 432 token cases, every m=1..8: bitwise raw dot/norm and
  coefficient agreement against reference arithmetic
- Includes zero, sparse, signed, tiny, large, wide mixed-exponent, subnormal,
  and signed-zero inputs. The original cancellation case gives raw dot 1,
  whereas rejected contiguous tiling gives 0. Moved cancellation across all
  rows and multiple pair boundaries also gives the exact reference result
- Ownership/bounds model covers every original FMA/norm chain, single weight
  ownership, all scratch slots and collapse outputs, and the pipeline drain
- Fixed acceptance test and optional graph harness compile/link on sm89;
  optional graph harness additionally compiles for sm86 and sm120

None of these checks executes on a GPU. GPU acceptance, raw device parity,
actual capture/replay, compute-sanitizer, and latency are still pending while
the queue is down. PTX inspection is not proof of final SASS overlap or speed.

## Reproduce supplemental checks

From repository root, with a host C++17 compiler:

```
g++ -O2 -std=c++17 -ffp-contract=off -fno-fast-math \
  src/ds41/kernels/k7/check_numerics.cpp -o /tmp/k7-model
/tmp/k7-model
python3 src/ds41/kernels/k7/check_structure.py
nvcc -std=c++17 -O3 -arch=sm_89 -Iinclude -Isrc -ptx \
  src/ds41/kernels/k7_hc.cu -o /tmp/k7.ptx
python3 src/ds41/kernels/k7/check_ptx.py /tmp/k7.ptx
```

`check_graph.cu` is an optional standalone GPU test linked with `ops.cu` and
`k7_hc.cu`, not a replacement for the fixed acceptance test. It checks the
expected two graph nodes, all m=1..8, repeated eager/replay calls on a
non-default stream, large/tiny/zero and cancellation inputs, unrelated task
traffic, and sequential device switching when multiple GPUs are available.
Run it under memcheck, racecheck and synccheck when GPU access returns. Test
harness copies/waits are deliberately outside the implementation.
