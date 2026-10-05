# exllamav3 GPU GEMV subset

Source: https://github.com/turboderp-org/exllamav3, commit
`16a49792a3c93d8432d72e6c4bce800841566577` (v1.5.4). MIT license;
see LICENSE (copyright Turboderp).

The paths in UPSTREAM.sha256 are copied from `exllamav3/exllamav3_ext/`
except LICENSE, which comes from the repository root. They were copied and
byte-verified before adaptation. The original history contains the unmodified
vendor import followed by the raw-pointer adaptation; the historical staging
plan is retained in ds41/tasks/K10.COMMITS.md.

This subset uses the small-row GEMV path, so no GEMM compilation units are
needed. `quant/exl3_gemv.cu` is included by the K10 unity translation unit.
The adaptation replaces its PyTorch host dispatch with a raw device-job API,
and moves the GEMV kernel's input/output Hadamards to surrounding launches.
The trellis decoder, codebook, MMA loop, FP16-to-FP32 folds, and reduction
remain upstream code. K10 calls the unchanged Hadamard helpers directly.
Only 3-bit mul1, single-row, narrow GEMV is instantiated.

`strata.patch` records all changes to upstream files. Apply it in reverse to
recover the unmodified vendor tree; verify against UPSTREAM.sha256. This makes
the first commit reproducible without relying on an untracked temporary copy.
The patch itself belongs to the subsequent adaptation commit.

K10-01 also applies the upstream narrow two-block launch bound to the integer
3-bit instance. This changes compiler register allocation, not the decoding,
MMA, FP16 fold cadence, reduction, or surrounding Hadamards/activation. The
updated reversible patch includes this change. See ds41/tasks/K10.REPORT.md
for compile-only resource evidence and the still-pending GPU measurements.

K10-03 keeps the exact K10-01 two-block bound and four-slot prefetch ring.
Only the narrow integer packed-weight loads change from `__ldcs`
(`ld.global.cs.u32`) to `__ldcg` (`ld.global.cg.u32`). The latter caches in L2
without filling L1. The scalar width, original 4-byte alignment requirement,
lane guards and addresses are unchanged; wide and half-integer templates keep
the upstream `__ldcs` fallback. No wider vector load is introduced. Hadamard,
input and codebook loads, decode arithmetic, FP16 folds and reductions remain
unchanged. The host check normalizes exactly this explicit policy block before
its whole-loop comparison; the reversible patch and all pristine hashes are
rechecked. Timing and GPU acceptance remain pending.

## K10-09 cooperative asynchronous packed-word staging

K10-09 starts from current feature and transplants the exact K10-04 async
schedule at `69cd9335938b531adca1acc5470ccd6c5ec4e57e`. It retains the narrow
3-bit mul1 kernel, original slice order and four-slice FP16 folds. Two
warp-private four-slice buffers overlap packed-word movement with unchanged
upstream decoding and MMA. Other template instances retain their register path.

The proven address mapping is `B32 + ks * ntiles * 24 + group * 48 +
load * 24 + lane`. Four adjacent words are contiguous, all non-lane byte
offsets are multiples of 16, and 24 is divisible by four. When the source base
is 16-byte aligned, lanes 0, 4, 8, 12, 16 and 20 each issue one 16-byte
`cp.async.cg` for four lanes. The destination has explicit 16-byte alignment.
Four-byte-aligned offset views use the original per-lane four-byte
`cp.async.ca`. Merely uint16-aligned views use two natural uint16 reads; the
public descriptor has no stronger source alignment contract.

All lanes commit. Leaders wait for their own copies, then a full warp barrier
publishes copied words to their peers. That same barrier retires the previous
buffer's consumers before any eager overwrite. A nonleader's own wait is not
used as a substitute for publication. The following decode shuffles, MMA,
FP16 folds, FP32 reduction, activation and all Hadamards are unchanged.

`CXX=g++ python3 src/ds41/kernels/k10/check_host.py` checks exact schedule
normalization to K10-04 and then the pristine arithmetic, the extracted
cooperative copy lambda with independent byte/ownership oracles, a temporal
publication/retirement model, tail/alignment/canary cases, mutations and the
reversible patch. These checks and three-architecture compilation do not
establish GPU correctness or speed. Exact-head GPU results remain pending.
