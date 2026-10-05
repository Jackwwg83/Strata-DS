# K1 FP8 block-scaled GEMV report

## Status

Work is on `feature/ds41-k1-fp8-gemv`, starting at `e7e62fb`.
The machine is an arm64 Mac with Apple clang 17.0.0. No network operations or pushes were attempted.

| Part | Code written | Compiled here | Tested on GPU |
| --- | --- | --- | --- |
| Public interface and scalar E4M3/E8M0/scale helpers | Yes | Yes, host C++17 | No |
| CUDA quantizer and GEMV | Yes | **not compiled: no CUDA toolkit on this machine** | No NVIDIA GPU |
| CPU reference and host lane/reduction model | Yes | Yes, clang++, including UBSan | Not applicable; these are CPU checks |
| CUDA runtime portions of parity and timing executable | Yes | **not compiled: no CUDA toolkit on this machine** | No NVIDIA GPU |
| CMake library and CTest registration | Yes | Not configured: CMake and CUDA toolkit are absent | No |
| sm_86 / sm_89 / sm_120 build acceptance | Sources written | Not compiled | No |
| RTX 4090 bandwidth and m=8/m=1 latency acceptance | Timing code written | Not compiled | **Not measured; acceptance remains unverified** |

The independent double-accumulation reference and the CPU model of the kernel passed 64 shape/batch
cases, including every specified shape with m in {1,2,4,8}. Maximum relative L2 error was
`0.00010515002`, below the unchanged `2e-3` tolerance. This is CPU-model evidence, not CUDA parity.
Direct comparison with `ds41/proto/torch_kernels.py` was unavailable because PyTorch is not installed.

## Implementation and review

- Quantize BF16 activation blocks of 32 with the FP32 `amax * (1/448)` operation and exact
  power-of-two ceiling. Software E4M3 conversion handles subnormals, signed zero, clamping and
  ties to even. The debug entry point uses the production quantizer and exports bytes and FP32 scales.
- Allocate dequantized activations with `cudaMallocAsync`, enqueue quantization and GEMV on the supplied
  stream, then enqueue `cudaFreeAsync`. There is no mutable global scratch and no explicit synchronization,
  including for the default stream. CUDA's stream-ordered memory-pool support is required.
- Four warps per block; each warp handles an output row and reads 16 weight bytes per lane. Decode
  the weight once for all tokens. Activation loads are also vectorized. A scalar-load fallback supports
  byte tensors whose base pointer is not aligned to 16 bytes. No FP8 tensor-core instructions are used.
- Reviewed K tails, partial row blocks, scale addressing, full-warp shuffle participation, integer bounds,
  and scratch lifetime. GPU tests cover unaligned weights, n=33, k=32/96/544, every m from 1 through 8,
  default/nondefault streams, and two simultaneous streams. These tests are written, not executed here.
- The parity CPU reference decodes FP8 arithmetically, finds nearest values independently, computes scales
  with `frexp`/`ldexp`, accumulates in double, and rounds directly from double to BF16. The required error
  is L2 against BF16 output; error against the unrounded double sum is an additional diagnostic.
- GPU timings encompass the full public call, including quantization and stream-ordered allocation/free.
  They use the median of 11 CUDA-event samples, with a buffer at least twice the reported L2 capacity
  overwritten before each sample outside its timing interval. The benchmark temporarily retains scratch
  in the CUDA allocation pool. Effective bandwidth counts exactly weight bytes plus scale bytes.
  The executable reports the 667.8 GB/s and 1.5x limits for reviewer evaluation; its success status is
  numerical parity, not a claim that the speed limits pass.
- Only the four specified new source/build files, the one top-level include line, and the three explicitly
  requested task documents were added or changed. Assumptions are in `K1-fp8-gemv.QUESTIONS.md`.

## Checks actually run

### Tool availability

```text
Apple clang version 17.0.0 (clang-1700.4.4.1)
Target: arm64-apple-darwin25.5.0
Thread model: posix
InstalledDir: /Users/jackwu/Applications/Xcode-26.1.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
nvcc: None
cmake: None
```

`python3 -c 'import sys; print(sys.version); import torch; print(torch.__version__)'` exited 1:

```text
3.14.3 (main, Feb  3 2026, 15:32:20) [Clang 17.0.0 (clang-1700.6.3.2)]
ModuleNotFoundError: No module named 'torch'
```

### Host C++17 build and exhaustive conversion / small geometry checks

```sh
clang++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror \
  -DSTRATA_DS41_HOST_ONLY -Iinclude src/ds41/kernels/fp8_gemv_parity.cpp \
  -o /tmp/ds41-fp8-gemv-host
/tmp/ds41-fp8-gemv-host --host-selftest
```

The compiler emitted no diagnostics and exited 0. The test exited 0 with:

```text
host conversions: 65280 finite BF16 inputs, 256 E4M3 codes, 256 E8M0 codes, 756 FP8 tie probes OK
host BF16: 32639 positive boundaries and their negatives, ties and adjacent doubles OK
host lane model: 32 cases, max relative L2=0, tolerance=0.002 OK
```

### All required shapes on the CPU

```sh
/tmp/ds41-fp8-gemv-host --host-shapes > /tmp/ds41-fp8-gemv-host-shapes.log 2>&1
```

Exit 0. Actual output:

```text
host conversions: 65280 finite BF16 inputs, 256 E4M3 codes, 256 E8M0 codes, 756 FP8 tie probes OK
host BF16: 32639 positive boundaries and their negatives, ties and adjacent doubles OK
host model zero               N=1 K=32 m=1 rel=0
host model zero               N=1 K=32 m=2 rel=0
host model zero               N=1 K=32 m=3 rel=0
host model zero               N=1 K=32 m=4 rel=0
host model zero               N=1 K=32 m=5 rel=0
host model zero               N=1 K=32 m=6 rel=0
host model zero               N=1 K=32 m=7 rel=0
host model zero               N=1 K=32 m=8 rel=0
host model tail               N=33 K=96 m=1 rel=0
host model tail               N=33 K=96 m=2 rel=0
host model tail               N=33 K=96 m=3 rel=0
host model tail               N=33 K=96 m=4 rel=0
host model tail               N=33 K=96 m=5 rel=0
host model tail               N=33 K=96 m=6 rel=0
host model tail               N=33 K=96 m=7 rel=0
host model tail               N=33 K=96 m=8 rel=0
host model tail_k             N=65 K=544 m=1 rel=0
host model tail_k             N=65 K=544 m=2 rel=0
host model tail_k             N=65 K=544 m=3 rel=0
host model tail_k             N=65 K=544 m=4 rel=0
host model tail_k             N=65 K=544 m=5 rel=0
host model tail_k             N=65 K=544 m=6 rel=0
host model tail_k             N=65 K=544 m=7 rel=0
host model tail_k             N=65 K=544 m=8 rel=0
host model blocks             N=17 K=1280 m=1 rel=0
host model blocks             N=17 K=1280 m=2 rel=0
host model blocks             N=17 K=1280 m=3 rel=0
host model blocks             N=17 K=1280 m=4 rel=0
host model blocks             N=17 K=1280 m=5 rel=0
host model blocks             N=17 K=1280 m=6 rel=0
host model blocks             N=17 K=1280 m=7 rel=0
host model blocks             N=17 K=1280 m=8 rel=0
host model attn.wq_a          N=1280 K=5120 m=1 rel=0
host model attn.wq_a          N=1280 K=5120 m=2 rel=0
host model attn.wq_a          N=1280 K=5120 m=4 rel=0
host model attn.wq_a          N=1280 K=5120 m=8 rel=4.61225592e-08
host model attn.wq_b          N=32768 K=1280 m=1 rel=1.85498526e-07
host model attn.wq_b          N=32768 K=1280 m=2 rel=1.31221696e-07
host model attn.wq_b          N=32768 K=1280 m=4 rel=1.05558871e-07
host model attn.wq_b          N=32768 K=1280 m=8 rel=2.078893e-06
host model attn.wkv           N=512 K=5120 m=1 rel=0
host model attn.wkv           N=512 K=5120 m=2 rel=0
host model attn.wkv           N=512 K=5120 m=4 rel=0
host model attn.wkv           N=512 K=5120 m=8 rel=0
host model attn.wo_b          N=5120 K=8192 m=1 rel=0
host model attn.wo_b          N=5120 K=8192 m=2 rel=0
host model attn.wo_b          N=5120 K=8192 m=4 rel=6.38590067e-06
host model attn.wo_b          N=5120 K=8192 m=8 rel=4.52581338e-06
host model attn.indexer.wq_b  N=4096 K=1280 m=1 rel=0
host model attn.indexer.wq_b  N=4096 K=1280 m=2 rel=2.23225693e-08
host model attn.indexer.wq_b  N=4096 K=1280 m=4 rel=5.05304873e-07
host model attn.indexer.wq_b  N=4096 K=1280 m=8 rel=3.57299891e-07
host model shared.w1/w3       N=2304 K=5120 m=1 rel=3.52165115e-07
host model shared.w1/w3       N=2304 K=5120 m=2 rel=2.57274659e-07
host model shared.w1/w3       N=2304 K=5120 m=4 rel=1.81912252e-07
host model shared.w1/w3       N=2304 K=5120 m=8 rel=1.42959323e-07
host model shared.w2          N=5120 K=2304 m=1 rel=0
host model shared.w2          N=5120 K=2304 m=2 rel=2.7740581e-10
host model shared.w2          N=5120 K=2304 m=4 rel=1.96160401e-10
host model shared.w2          N=5120 K=2304 m=8 rel=2.27704902e-07
host model engram.wkv         N=25600 K=6144 m=1 rel=0.00010515002
host model engram.wkv         N=25600 K=6144 m=2 rel=7.43145659e-05
host model engram.wkv         N=25600 K=6144 m=4 rel=5.41661674e-05
host model engram.wkv         N=25600 K=6144 m=8 rel=3.83913255e-05
host lane model: 64 cases, max relative L2=0.00010515002, tolerance=0.002 OK
```

### UndefinedBehaviorSanitizer

```sh
clang++ -std=c++17 -O1 -g -Wall -Wextra -Wpedantic -Werror \
  -fsanitize=undefined -fno-omit-frame-pointer -DSTRATA_DS41_HOST_ONLY \
  -Iinclude src/ds41/kernels/fp8_gemv_parity.cpp -o /tmp/ds41-fp8-gemv-host-ubsan
/tmp/ds41-fp8-gemv-host-ubsan --host-selftest
```

Compiler and test both exited 0, with no sanitizer diagnostics. Actual test output:

```text
host conversions: 65280 finite BF16 inputs, 256 E4M3 codes, 256 E8M0 codes, 756 FP8 tie probes OK
host BF16: 32639 positive boundaries and their negatives, ties and adjacent doubles OK
host lane model: 32 cases, max relative L2=0, tolerance=0.002 OK
```

### AddressSanitizer: incomplete

```sh
clang++ -std=c++17 -O1 -g -Wall -Wextra -Wpedantic -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer -DSTRATA_DS41_HOST_ONLY \
  -Iinclude src/ds41/kernels/fp8_gemv_parity.cpp -o /tmp/ds41-fp8-gemv-host-asan
/tmp/ds41-fp8-gemv-host-asan --host-selftest
```

The executable was built, but the invocation produced no output and did not complete. It was interrupted
(exit 130). A Python `subprocess.run` probe of `--unknown`, with `ASAN_OPTIONS=detect_leaks=0` and a
15-second timeout, also did not complete. The probe printed:

```text
ASan CLI probe timed out after 15 seconds before completing argument parsing
```

The cause is unverified. ASan is **not** recorded as passing; the separate UBSan run above passed.

### CLI and host-only skip behavior

The host-only executable was invoked by Python `subprocess.run`, checking the exact return codes:

```text
SKIP: host-only build; CUDA was not compiled
exit: 77
usage: fp8_gemv_parity [--selftest | --host-selftest | --host-shapes]
exit: 2
```

The first invocation was `--selftest`, the second `--unknown`. This verifies the host-only skip branch,
not the CUDA executable's runtime no-device branch. CTest's `SKIP_RETURN_CODE 77` is written but unrun.

### Source and geometry checks

A Python check compared `CMakeLists.txt` byte-for-byte with `git show e7e62fb:CMakeLists.txt` after
removing the added include. It enumerated the lane/vector indexing for K=32,96,544,1280,2304,5120,6144,8192,
checking exact coverage, 16-byte alignment, and no 32-value scale-block crossings. It also checked the
CUDA source for stream-ordered allocations and absence of explicit synchronization/tensor-core instructions.
Actual output:

```text
CMakeLists.txt: exactly one include line; all original bytes preserved
GEMV index model: all 8 K geometries covered exactly once, aligned, no scale-block crossings
CUDA source: stream-ordered allocation/free; no explicit synchronization or tensor-core instructions
```

`git diff --check` exited 0 with no output. Source inspection is not CUDA compilation or generated-code inspection.

The final scope/branch/whitespace check exited 0 with:

```text
Final scope: exactly 8 permitted paths; no other repository files changed
Whitespace: git diff --check and all new-file trailing-whitespace checks passed
Branch: feature/ds41-k1-fp8-gemv
Commit: cf8bec0 (4 files); remaining changes uncommitted because Git metadata writes are protected
```

## Commits and remaining work

The first local commit succeeded:

```text
cf8bec0 ds41: implement block-scaled FP8 decode GEMV
```

It contains four files: the public header, CUDA source, QUESTIONS.md and initial PROGRESS.md.
The attempted second commit (four files: CMakeLists.txt, cmake/ds41.cmake, parity source and progress update)
was blocked at `git add` by the workspace's protected Git metadata:

```text
fatal: Unable to create '/Users/jackwu/projects/Strata-DS/.git/index.lock': Operation not permitted
```

The remaining implementation/build/test files and final report/progress changes are left in the working tree.
No attempt was made to bypass that protection. Completing the remaining small commits needs Git metadata
write access; it is not an unanswered implementation question. No push was attempted.

On a CUDA machine, the reviewer still needs to compile and run, for example:

```sh
cmake -S . -B /tmp/strata-k1-cuda -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES='86;89;120' -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/strata-k1-cuda --target fp8_gemv_parity -j
ctest --test-dir /tmp/strata-k1-cuda -R '^fp8_gemv_parity$' --output-on-failure -V
compute-sanitizer --tool memcheck /tmp/strata-k1-cuda/fp8_gemv_parity --selftest
```

Those commands were **not run here**. Check the printed RTX 4090 timings against both specified limits;
no performance conclusion can be drawn from the Mac's CPU model.
