# K7-05: one compute launch with a last-completing CTA

This candidate uses one ordinary 48-CTA grid. Each 128-thread CTA owns half
of a weight row's original 256-lane accumulation sequences, sharing FP32 weights
across all `m` tokens. Contiguous-K tiling was rejected after a legal cancellation
case proved that it changes the result substantially. After publishing
partials, every CTA takes one integer completion ticket. The last ticket holder
reduces the partials, evaluates the exact 20-step Sinkhorn sequence, collapses
all outputs, and resets the counter. Nonwinners return immediately.

This is an offline candidate. Compilation and CPU models do not establish GPU
correctness, race freedom on hardware, graph replay, or performance.

## Workspace ownership and bounds

- The fixed header permits exactly `m=1..8`, `x[m][4][5120]`, and 24 weight rows.
- The K7 translation unit owns a separate device allocation for every CUDA
  device ordinal. It contains `dots[8][24][8]` plus `squares[8][32]`
  (7,168 bytes total) and one aligned unsigned counter (4 bytes). No pointer or arena is shared with another task.
- The first eager call allocates the full 7,172 bytes and zeros only the counter
  with `cudaMemsetAsync` on the supplied stream. It rejects first-call capture.
  Every later eager call/capture only looks up that allocation. No growth, free,
  memcpy, host result read, or synchronization is used by the implementation.
- Each `(token,row,original_warp)` dot slot has one writer. Rows 0 through 3
  also own separate norm warps, covering all 32 original norm warps. Every slot consumed by a call is overwritten by that
  call, including when `m` changes. Unused token slots need no initialization.
- A host mutex protects the per-device allocation registry. This mutex is not a
  substitute for device serialization. The engine's explicit guarantee is that
  K7 calls on the same device, including decode graph replays, do not overlap.
  Different task kernels can overlap because their workspace allocations differ.

## CUDA memory-order proof

Use `cuda::atomic_ref<unsigned, cuda::thread_scope_device>` with the required
alignment. Do not replace it with an unsuffixed legacy atomic or block-scope
atomic. The counter and partials are in ordinary device global memory.

1. For CTA B, its writers store the partials and then all threads execute a CTA
   barrier. Those writes therefore happen-before B's leader executes its
   completion `fetch_add(1, memory_order_acq_rel)`.
2. Integer RMWs on this device-scope atomic form one modification order. Each
   RMW reads the immediately previous value. Its acquire synchronizes with the
   preceding release RMW; its release carries that acquired history plus its
   own CTA's preceding writes. By induction, ticket `i` acquires the partials
   published by all `i+1` CTAs through that point.
3. Exactly one leader receives ticket 47, so that leader has acquired all 48
   publishers. A second CTA barrier propagates the acquired history and the
   shared winner flag to every thread in the winning CTA before any partial
   reads. The global partial accesses are non-atomic but happens-before ordered;
   no volatile-cache workaround or bare-fence assumption is needed.
4. Nonwinners have no scratch accesses after their RMW and return after the
   shared flag barrier. They never poll, spin, or wait on a different CTA. Even
   a GPU scheduling only one CTA at a time makes progress. No cooperative launch,
   all-resident grid, or fixed last block index is assumed.
5. The winner completes all reads and output stores, executes a final CTA
   barrier, then its leader performs a release store of zero. Every other CTA
   has already issued its only counter operation. The engine orders the next
   K7 execution after this execution, so its first RMW sees zero and cannot
   overwrite scratch still in use. Outputs and scratch are not shared between
   overlapping same-task executions because those executions are prohibited.

CUDA's [memory model](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html)
specifies device-scope release/acquire message passing and C++ atomic semantics.
The CUDA 12.8 [programming guide](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#memory-fence-functions)
also demonstrates the last-block pattern; its adjacent synchronization section
specifies CTA-barrier visibility. This candidate uses ordered atomics instead
of that example's legacy atomic/fence/volatile sequence.

## Lifecycle, capture, and errors

A successful eager execution ends with counter zero. Capture records one
kernel node and does not execute it, so it does not change the counter. A
successful replay resets it exactly like an eager execution; multiple K7
nodes on the task's stream and serialized replays use the same allocation.
Capturing or discarding an unexecuted graph leaves the counter unchanged.
A launch rejected before any CTA executes also leaves the counter unchanged;
reported initialization or launch errors abort through `check_cuda`.

An interrupted/partially executed grid is different: if some publishers ran
without the winner's tail reset, the counter is poisoned. This candidate does
**not** claim recovery by launching again, and does not silently reset on the
host. The negative CPU model demonstrates why reuse would elect a winner too
early. Illegal addresses, kernel exceptions, assertion failures, and execution
timeouts have a documented process-restart requirement in the CUDA 12.8
[runtime error definitions](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__TYPES.html).
The implementation does not detect asynchronous faults immediately because it
must not wait for the device. Existing engine/runtime error handling must treat
such faults as terminal, as CUDA requires; these checks are not a recovery API.

The allocation has the process/context lifetime required by task rule 7. Do not
use this candidate after `cudaDeviceReset`, context destruction/recreation, a
custom mechanism that cancels only part of a grid while retaining a usable
context, or an execution fault. A device ordinal alone cannot identify a new
context. Supporting those scenarios would need an explicit lifecycle hook or
per-launch reset and is not provided by the fixed interface. If the engine
requires recoverable partial runs or same-device K7 overlap, reject K7-05 and
use an appropriately isolated, externally reset implementation instead.

## Numerics and checks

FP32 input weights and BF16 inputs are retained. Every dot lane retains its original 80-step stride-256 FMA sequence; every norm
lane retains its original 20-step stride-1024 sequence. Warp reductions and the
final eight/32-warp sums retain the reference parenthesization. This fixes the
contiguous-K tiling defect: x=1, fn row0 columns 0/1024/1280 = 2^25/-2^25/1
produces reference dot 1, while the rejected tiling produced 0. Norm multiplication uses `__fmul_rn` before
scale/bias. Sinkhorn performs ordered four-value sums, initial row softmax plus
epsilon, initial column normalization, and exactly 19 row/column pairs with
epsilon in every denominator. Collapse uses `pre_in` (not new coefficients),
ordered `j=0..3` accumulation, and one BF16 conversion.

- `python3 src/ds41/kernels/k7/check_last_cta.py`: source invariants, exact
  ownership/bounds, all small-grid ticket permutations, all 48 possible winners,
  512 mixed-m replay models across three device-private states, and a negative
  partial-run recovery model. This is a logical model, not CUDA racecheck.
- Compile `check_numerics.cpp` with C++17, `-O2 -ffp-contract=off`, then run:
  CPU arithmetic comparisons for all m, random/zero/alternating/sparse/tiny
  inputs and the exact cancellation regression; bitwise reference-versus-candidate
  raw dot/norm comparisons and scalar-versus-warp Sinkhorn. CPU libm is not CUDA libm.
- `check_graph.cu` is an optional hardware harness; compile/link with `k7_hc.cu`
  and unmodified `ops.cu`. It checks one-node captures after m=1 warmup, every
  legal m, changing inputs, the cancellation regression, 128 mixed eager/replay calls per
  device, unrelated
  stream traffic, and per-device registry reuse. Run separately under Compute
  Sanitizer memcheck, racecheck and synccheck. These hardware runs are pending.
- The fixed acceptance test is untouched. CUDA 12.8 builds target sm_86, sm_89,
  and sm_120. No benchmark-seed branches or GPU timing claims are present.

### Offline validation of the submitted source

CUDA 12.8 compile-only checks pass for sm_86, sm_89, and sm_120. Each architecture
reports at most 40 registers/thread, 801 bytes shared memory, and zero spills.
All eight sm_89 PTX specializations contain device-scope acquire-release integer
RMWs, a release reset, and four CTA barriers. The fixed acceptance test and the
optional graph harness compile and link for sm_89; the harness also compiles
for sm_86 and sm_120. Neither executable has been run here. The CPU numerical
model passes 216 token cases with bitwise raw dot/norm and ordered-transform
equality, including the exact cancellation regression. The logical ownership,
ordering, and lifecycle model passes. No full CMake build or GPU measurement
has been performed in this environment.
