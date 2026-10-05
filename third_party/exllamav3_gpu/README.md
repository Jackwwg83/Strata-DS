# exllamav3 GPU GEMV subset

Source: https://github.com/turboderp-org/exllamav3, commit
`16a49792a3c93d8432d72e6c4bce800841566577` (v1.5.4). MIT license;
see LICENSE (copyright Turboderp).

The paths in UPSTREAM.sha256 are copied from `exllamav3/exllamav3_ext/`
except LICENSE, which comes from the repository root. They were copied and
byte-verified before adaptation. The reviewer must make the unmodified vendor
commit first, then the patches, as described in ds41/tasks/K10.COMMITS.md.
No commits are made by the implementation agent.

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

## K10-04 asynchronous packed-word adaptation

The K10 raw-pointer adapter enables a guarded, warp-private double buffer of
four raw K slices using four-byte `cp.async` copies. Each lane reads its own
shared words before the original shuffle/decoder/MMA path. The four-slice FP16
fold cadence, reduction and all activation/Hadamard paths are unchanged.
The new flag defaults off for other configurations. The complete adaptation
is recorded in `strata.patch`; reversing it still restores every upstream hash.

`src/ds41/kernels/k10/check_host.py` checks provenance, exact allowed scheduling
edits, mutation-sensitive arithmetic preservation and the actual C++ schedule
under eager and delayed symbolic copy completion. The manifest and scripts are
CPU validation tools, not kernel runtime dependencies. Compilation and source
checks do not establish GPU numerical parity, race safety or speed.
