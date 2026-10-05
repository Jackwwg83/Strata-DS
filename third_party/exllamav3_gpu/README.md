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
remain upstream code. K10-05 uses a token-identical preloaded specialization
of the unchanged FP16 Hadamard helper for paired gate/up input transforms;
the remaining stages call unchanged Hadamard helpers directly. Only 3-bit
mul1, single-row, narrow GEMV is instantiated.

`strata.patch` records all changes to upstream files. Apply it in reverse to
recover the unmodified vendor tree; verify against UPSTREAM.sha256. This makes
the first commit reproducible without relying on an untracked temporary copy.
The patch itself belongs to the subsequent adaptation commit.

K10-01 also applies the upstream narrow two-block launch bound to the integer
3-bit instance. This changes compiler register allocation, not the decoding,
MMA, FP16 fold cadence, reduction, or surrounding Hadamards/activation. The
updated reversible patch includes this change. See ds41/tasks/K10.REPORT.md
for compile-only resource evidence and the still-pending GPU measurements.
