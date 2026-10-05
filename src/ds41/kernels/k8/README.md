# K8-12: one-launch decode, unchanged two-stage m2..8

This bounded experiment starts at feature commit
`3ee92cd12e59289735849f39df7357ee02d5e9fd`
and preserves the merged K8-07 numerical implementation at
`b45ddd889c37a5bb27811396b64883eb67b2935f`. Only K8 CI-FILES change; the
interface, fixed acceptance test and task specification stay unchanged.

The new path is only m=1. Ninety-six 128-thread CTAs each produce four expert
scores, exactly as K8-07 does. Each CTA publishes one completion ticket. The last
CTA's first warp performs the original deterministic double-precision top-six
selection. No CTA spins or waits for another CTA to become resident. m=2..8 keep
K8-07's existing score kernel, selector kernel, launch geometry and stream usage.

## Revision 2: explicit score-writer and winner fences

Following [reviewer directive 5998972308](https://github.com/Jackwwg83/Strata-DS/issues/8#issuecomment-5998972308),
this revision adds `__threadfence()` in each score-writing lane (thread IDs
0, 32, 64, 96) after its store and before the first CTA barrier. Every thread in
the winning CTA also executes `__threadfence()` after the winner-flag barrier and
uniform nonwinner return, before any selector score load. All three existing CTA
barriers, the device-scope acq_rel RMW and release reset remain in place.

The [CUDA 12.8 memory-fence documentation](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#memory-fence-functions)
defines the device-scope fence and demonstrates producer-store-before-ticket
ordering for last-block reduction. Fences order memory accesses; they do not
replace the retained acquire/release visibility proof or CTA barriers here.

The prior unfenced head `2c835e1e2e0e1e57bb0dc474a9fac16ac517dbe5` scored
18.92 us (m1 15.36) in the queue. Those measurements do not belong to this revision.
The added-fence head needs its own queue result and must beat 19.84 us by at least
3%, with m1 no slower than 16.38 us. It is not approved for merge on old timings.

## Workspace and ordering contract

One K8-private `Workspace` allocation per CUDA device contains all eight tokens'
384 double scores (24,576 bytes), an aligned unsigned completion counter and
padding: 24,584 bytes total. It is allocated once on the first legal eager call,
even if that call is m>1. An asynchronous memset on the supplied stream initializes
only the counter; score elements are fully overwritten before use. Every shape
then retains and reuses the same pointer for the process lifetime. There is no
shared arena with any other task.

The engine guarantees nonoverlapping, ordered K8 calls and graph replays on each
device, including across streams. The first eager call completes before a replay
using this arena. Different task kernels may overlap because their state is
separate. Host locking protects the per-device registry, not GPU execution.

After the first eager call, lookup does not allocate, free, copy to/from the host,
initialize memory, query capture status, or synchronize with the host. First-call
capture is rejected clearly because the task contract requires eager warmup.
All kernels and initialization use the supplied stream. m=2..8 never access the
counter. Capturing or discarding a graph does not execute the recorded work and
therefore does not change the counter.

## Publication and reuse proof

The protocol follows K7-05's reviewed device-scope acquire/release design, not its
collapse computation. `Completion` is
`cuda::atomic_ref<unsigned, cuda::thread_scope_device>` with explicitly required
alignment. Explicit producer and reader fences supplement this protocol; they
do not replace it with legacy unsuffixed atomics.

1. Each CTA's four warp leaders write their disjoint scores and each writer
   executes a device fence. Then all 128 threads join the first CTA barrier. All four writes happen-before the CTA leader's
   completion operation.
2. The leader performs one `fetch_add(1, memory_order_acq_rel)`. Integer RMWs form
   a modification order. Each RMW acquires the immediately preceding published
   history and releases it together with its own CTA's four scores. Inductively,
   the final ticket (95) acquires every expert score, independent of CTA order.
3. The second CTA barrier distributes the acquired history and shared winner
   flag. Nonwinning CTAs return uniformly and never access the arena again. The
   winner's 128 threads each execute a device fence. Its entire warp 0 then loads
   the 384 scores and executes the unchanged
   deterministic selector. All 32 lanes participate in each shuffle.
4. All winner threads join the third CTA barrier after selection. Every score
   read and every output write happens-before the leader's release store of zero.
   All other CTAs already issued their sole counter operation. Ordered next calls
   or graph replays begin with zero; there is no per-call memset or reset launch.

This is regular device-scope C++ acquire/release message passing, with CTA
barriers connecting the participating non-atomic writers/readers to the atomic
leader. Ordinary global score loads are ordered; volatile is unnecessary. The
CUDA 12.8 PTX emitted here uses `atom.add.acq_rel.gpu.u32` and
`st.release.gpu.b32`, plus two static `membar.gl` fence sites and three
executed CTA barriers on the winning path. The first fence site runs in the four
score-writing lanes of every CTA; the second runs in all 128 winner threads.

A rejected launch that executes no work leaves the counter unchanged. Recovery
from a partially executed kernel/device fault is not supported: failure before
the final reset can poison the counter. The engine must not continue reusing this
arena after such a device execution failure. Normal sequential calls and complete
graph replays are the supported lifecycle; overlap is expressly unsupported.

## Exact arithmetic preservation

Every expert keeps the original lane ownership (`lane + 32*j`), 160 ascending
FP32 `__fmaf_rn` operations, and FP32 additions at shuffle offsets 16,8,4,2,1.
There is no split-K, reassociation, input quantization change or FP32 shortcut.
`math.hpp` is unchanged: convert the FP32 logit to double, use the original
`z > 20` softplus branch, then double sqrt. Selection adds double-converted bias,
compares doubles, resolves exact ties toward the lower original expert ID, and
removes winners using both negative infinity and the sentinel ID.

The fused warp helper is an exact copy of the K8-07 selector body with token fixed
to zero. The separate global selector remains byte-identical for m2..8. Original
unbiased scores are broadcast from their owner lanes. Lane zero adds them in
selected order. Weight normalization retains the original double expression and
final float conversion. Completion order never enters the ranking or sum.

## Checks and limits

Portable checks from the repository root:

    python3 src/ds41/kernels/k8/check_last_cta.py
    g++ -std=c++17 -O3 -march=native src/ds41/kernels/k8/host_semantics.cpp -o /tmp/k8_semantics
    /tmp/k8_semantics

The first checks exact source equivalence to K8-07; device-scope ordered protocol
and workspace invariants; all 5,913 completion permutations for grids of size
1..7; each of 96 possible final CTAs; 1,024 randomized mixed-shape calls across
three independent device arenas; and negative examples for missing publication,
missing producer barriers and partial-failure reuse. It also checks every writer
and selector-reader fence, rejecting 36 omitted-lane coverage cases. The runtime
source audit proves the revision adds exactly the requested fences to the prior
head and retains its arithmetic, workspace, dispatch and synchronization. It models happens-before
relationships; it does not execute the device weak-memory implementation.

The inherited numerical model passes 2,503 routing cases, including 13,824
bit-exact FP32 logits for every m=1..8, threshold neighbors, exact/near ties,
underflow, extreme bias, all winner owner lanes/registers and 1,024 permutations.
The CPU-model weight difference is zero. This proves the modeled ordering and
arithmetic, not GPU libm behavior or a latency improvement.

CUDA 12.8 C++17 compile checks pass on sm_86, sm_89 and sm_120, including the
fixed acceptance and optional graph test translation units. Both tests link on
sm_89. The optional executable exits 77 (no CUDA device), so it has not passed a
GPU run. All m2..8 kernel bodies have identical normalized PTX and byte-identical
cubin text sections to K8-07 on all three targets. The fused kernel
uses 93 registers per thread on each architecture, one shared byte and no
stack/spill loads or stores. All task kernels use at most 20,480 shared bytes,
below 99 KiB. The old decode producer used 27/28 registers and its separate
selector used 80; fusion can reduce occupancy across all four producer warps.
Ninety-six serialized integer atomics, explicit device fences, three winner-path
CTA barriers, increased
register footprint and the single-warp tail can cost more than one saved launch.
Only an exact-head GPU measurement can decide this experiment.

`graph_validation.cu` is an optional test, not the fixed acceptance test. Its
build command is in the file. In fresh processes, run `k8_graph_validation 1`
through `k8_graph_validation 8` to cover every first-eager shape. It checks:

- One captured kernel node for m1, two for every m2..8
- Discarding a captured m1 graph without executing it
- All legal shapes with 24 changing-input replays
- 4,096 additional m1 replays over two event-ordered non-default streams, with
  poisoned outputs and a checked output snapshot from every replay
- Changing inputs every 128 replays and m2..8 eager calls between batches
- Captured exact ties and FP32-collapsed near-ties at both m1 and m8

The fixed queue's K8 graph test covers m8 only. Its pass must never be described
as validation of the new fused m1 capture path. The optional test must actually
run to establish m1 graph-replay evidence. Revised-head GPU correctness, m1 graph runtime,
compute-sanitizer and latency are pending; no speedup is claimed from these CPU
or compile checks.
