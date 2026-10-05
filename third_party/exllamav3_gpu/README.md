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

## K10-02 prefetch experiment (2026-10-05)

The narrow kernel now uses a two-slot register prefetch ring, while retaining
its four-slice unrolled arithmetic group, four-slice FP16-to-FP32 fold,
16 K-split warps, two N tiles and ordered cross-warp reduction. The narrow
launch bound permits two 512-thread blocks. No decode/MMA/fold arithmetic,
Hadamard, workspace, job dispatch or routing logic changes.

`strata.patch` includes this scheduling change and still reverses to all 13
pristine hashes. `check_host.py` compares the arithmetic core after explicitly
normalizing only the reviewed scheduling edits, and checks ring contents and
fold boundaries across zero-work and partial-tail cases. CUDA 12.8 compilation
passes for sm_86, sm_89 and sm_120; GPU correctness and speed are unmeasured.
