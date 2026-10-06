# K1c-09: packed conversion for small single-token GEMVs

Base: `be8c969a1f1b7bf88d8a64ef1b3e935dcc2f376a` (`feature/ds41`).
Branch: `task/K1c/dots-K1c-09`.

This is a local, unmeasured candidate for the fixed K1c queue, not a GPU acceptance result. The current
3060 control is 29,170 us (m1 20,776 us; m8 67,194 us). Acceptance still requires the fixed numerical/graph
tests, at least a 3% aggregate improvement, and no regression in real m1 decode. No PR, merge, or push was made.

## Scope and implementation

Only aligned m1 calls with the unchanged header's SPLIT > 1 policy and at most 16 MiB of weights select the
new kernel. In the fixed acceptance test this means wq_a, wkv, idx_wq_b, shared_w1_w3, and shared_w2.
All m2..8 calls, natural-alignment fallback calls, and the large wo_b/wq_b/engram calls use the original kernel.
The original quantizer, validation, stream handling, public interface, fixed tests, and CMake are unchanged.

The new path retains the original 128-thread CTA, ROWS/SPLIT policy, 16-byte lane packet ownership, adjacent-row
activation/scale reuse, sequential FP32 FMAs, warp reduction, and ordered cross-warp sum. It changes:

- Paired FP8 conversion. On sm86, packed bit operations form FP16 representations of each finite E4M3 value
  divided by 256, including signed zero and E4M3 subnormals. Conversion to FP32 is exact. Multiplying by the
  E8M0 scale times 256 gives the same FP32 product. Scale codes 0..246 use this path; 247..255 use the original
  scalar decode/multiply. The rare-scale guard is outside the pair conversions. NaNs are explicitly retained
- On sm89/sm120, CUDA's native E4M3x2-to-FP16x2 conversion supplies the exact intermediate, then conversion and
  scaling are FP32. These are conversion instructions, not FP8 tensor-core operations
- The n <= 8192 and <=16 MiB launch conditions prove that the grid is not capped and all offsets fit signed int.
  A CTA therefore visits its row group once, uses 32-bit indexing, and needs only one shared-partial barrier
- A 128-thread/12-CTA launch bound constrains register use without spills. The register reduction and shorter
  conversion dependency path are reasons to test this variant, not measured bandwidth claims

No new workspace, allocation, free, host synchronization, or host/device data transfer is introduced. The
pre-existing compatibility wrapper's allocation behavior is unchanged; the tested `fp8_block_gemv_q` path
still launches one kernel on the supplied stream.

## Validation on the dot cloud computer

Genuine CUDA 12.8.93, GCC 12, C++17, O3, FTZ disabled, precise division/square root. No GPU is available.
All checks below used the final runtime sources recorded in `K1c-09-validation.json`.

- sm86, sm89, and sm120: kernel, unchanged reference ops, and unchanged fixed acceptance test compile and link
- Every linked executable contains the corresponding native cubin. Each fixed test exits 77, explicitly skipped
- On each architecture, all 97 original GEMV/quantizer PTX entries are identical to the unchanged control after
  normalization of anonymous-namespace symbols and compiler basic-block numbering
- All four added entries use wide weight and activation loads and one barrier. sm86 PTX contains the packed
  byte permutation and FP16-to-FP32 conversion; sm89/sm120 contain the native E4M3x2 conversion
- New-kernel registers: sm86 40; sm89 38 for one row and 40 for two; sm120 37/39 for one row and 40 for two.
  Shared memory is 16 or 32 bytes. Stack and spill traffic are zero. The sm86 control uses 43/48 registers
- The production software decoder passes all 16,777,216 pair/scale combinations (33,554,432 values) against the
  original scalar helper and CUDA's host model of the native converter. Finite results, including signed zero,
  are compared bitwise; NaNs are compared by classification, not payload/sign preservation
- 280 layout geometries check unique weight/output ownership, row-scale reuse, 16-byte tails, and fallback
  capped-grid coverage. 2,496 sampled output rows compare every lane accumulator and final BF16 output bitwise
- The unchanged host parity test passes: 288 geometries; all finite BF16 inputs and FP8/E8M0 codes; rounding
  boundaries; 56 lane-model cases, worst relative L2 1.84642982e-09
- Three deliberately broken decoder variants (scale bias, sign preservation, byte order) are rejected by the
  exhaustive check. The working test also includes swapped-byte, signed-value, subnormal-scale, extreme-scale,
  and NaN negative controls
- Protected header/test/reference/build files match the base byte-for-byte; changed files are within CI-FILES

GPU numerical parity, real native-conversion execution, graph replay, sanitizer checks, bandwidth, and the
weighted score remain untested. The existing toolchain has no nvdisasm, so SASS instruction validation is
explicitly unavailable; no claim is based on the short cuobjdump fatbin metadata output.

## Prior candidates reviewed

The public K1c-01..08 kernel sources were read before implementation. They explored smaller m8 packets (01),
prefetch/cache hints and scale shuffles (02), four-row reuse (03), shared activation tiles (04), a scalar
exponent-bias decoder (05), asynchronous weight staging (06), m8 packets plus L1 bypass (07), and m8 occupancy
bounds (08). This candidate uses a separate small-m1 packed converter, preserves the old cache policy and
m8 implementation, and adds neither cross-CTA split-K nor an extra launch. It is not a replay of 07 or 08.

## Reproduction and evidence

From this checkout, source the existing campaign `toolchain/env.sh`, then run:

```sh
src/ds41/kernels/fp8_gemv/build_small_pair.sh /path/to/build/K1c-09
python3 src/ds41/kernels/fp8_gemv/check_pair_mutants.py /path/to/build/K1c-09/mutants
python3 src/ds41/kernels/fp8_gemv/validate_small_pair.py \
    /path/to/build/K1c-09 /path/to/unchanged-control-build
```

Canonical local final evidence is `strata-prefill/build/K1c-09-frozen/`; unchanged control evidence is
`strata-prefill/build/K1c-10-control/`, both based on be8c969. Earlier draft/final/verified directories are
intermediate work and are not the delivered evidence. Build metadata reports the base commit because the
local source was committed after validation; the manifest binds the compiled source content by SHA256.
