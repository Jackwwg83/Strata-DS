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

K10-08 combines the exact K10-03 narrow integer cache policy with K10-02's
register schedule: two prefetch slots, retaining the original four-slice
narrow arithmetic unroll and FP16 fold boundaries. The two-CTA bound stays.
Wide/half-integer load policies, guards, widths and addresses remain unchanged.
The reversible patch contains both changes; the host audit normalizes only
those explicit substitutions before comparing upstream arithmetic. An actual
source-extracted C++ schedule model checks ring tails and fixed upstream fold
boundaries, including mutation rejection. K10-08's own GPU acceptance and
performance are pending; individual controls' gains are not assumed additive.
