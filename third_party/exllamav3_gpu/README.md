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


## K10-07: eight-warp narrow gate/up tuning

The vendor CFG 1 is not used: it also changes the FP16 fold cadence to two.
This candidate adds a `KSPLIT8` override to CFG 0 for gate/up only: eight
warps, two adjacent n-tiles, four prefetch entries and four-slice FP16 folds.
This is a new combination of vendor geometry parameters, not an existing
unchanged upstream configuration. The loop, trellis decoding, MMA operations
and reduction statements remain the original vendor source.

Gate/up K=5120 has 320 slices: the original 16 splits each process 20 slices,
and the new eight splits each process 40. Both produce the same 80 contiguous
four-slice FP16 fold groups. Their subsequent FP32 association differs.
Down K=2304 stays at 16 warps because changing its nine-slice chunks to 18
would move FP16 fold boundaries. The original K10-01 two-block register bound
is retained on the down path. Gate/up uses a three-block minimum bound to
avoid spills in CUDA 12.8 on sm_86/89/120.

`src/ds41/kernels/k10/check_split_geometry.py` extracts the actual ring and
partition code and models the changed FP32 additions from identical arbitrary
finite FP16 fold outputs. It does not emulate tensor-core products or establish
golden parity. Fixed GPU acceptance at relative L2 <=5e-3, graph replay and
actual timings remain required after K10 GOLDEN READY. No gains are claimed.
