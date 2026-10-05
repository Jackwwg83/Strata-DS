# K1b FP8 GEMV implementation and validation report

Work is on `feature/ds41-k1b-fp8-gemv-speed`, starting at `cb99f4a`, on an arm64 Mac.
No commit, index write or push was attempted. Changes are limited to the three permitted source files
and four `ds41/tasks/K1b.*` documents.

## Status by part

| Part | Code written | Compiled | Tested on GPU |
| --- | --- | --- | --- |
| New public declarations, scalar helpers and dispatch policy | Yes | Host C++17, clang 17 | No |
| Allocation-free FP32 quantizer and GEMV, compatibility wrapper | Yes | **not compiled: no CUDA toolkit on this machine** | No NVIDIA GPU |
| Split-K, paired output rows and weight/activation register reuse | Yes | **not compiled: no CUDA toolkit on this machine** | No NVIDIA GPU |
| CPU reference, numerical model and layout checks | Yes | Host C++17; optimized and sanitizer builds | Not applicable; CPU checks only |
| CUDA parity, graph capture/replay, concurrent streams and timings | Yes | **not compiled: no CUDA toolkit on this machine** | No NVIDIA GPU |
| sm_86 / sm_89 / sm_120 acceptance with CUDA 12.8 | Source written | Not compiled | Not tested |
| RTX 4090 speed acceptance | Benchmark written | Not compiled | **Not measured; all speed targets remain unverified** |

The final host test passes 88 numerical cases, including every specified shape with m in {1,2,4,8},
and 288 layout geometries. Maximum relative L2 against the independent BF16-rounded double reference
is `0.00010515002`, below the unchanged `2e-3` tolerance. Activation bytes, scales and dequantized FP32
values are checked bitwise in the host numerical cases. This does not establish CUDA correctness.

UBSan passes the 56 small/boundary numerical cases plus the conversion/layout checks. The ASan+UBSan
build succeeds, but execution times out after 30 seconds without test output; ASan is **not passing**.
Direct execution of `ds41/proto/torch_kernels.py` is unavailable: this Python has no `torch` module.

## Implementation and source review

- `fp8_quantize_activation_f32` writes caller-owned `[m,k]` float storage. It invokes the same quantizer
  as K1. The quantizer and scalar numerical conversion helpers are unchanged byte for byte.
- `fp8_block_gemv_q` consumes that storage and launches one GEMV kernel. Neither new entry point allocates,
  frees, queries the device, or synchronizes. Launch error checks remain asynchronous. The wrapper retains
  stream-ordered allocation/free and delegates to the two new functions.
- The launch uses 128 threads. For N <= 2048, four warps split K for each output group; for
  2048 < N <= 8192, two warps split K; larger N uses one warp per group. For N >= 1024, a group computes
  two neighboring output rows, including m=1. Smaller N computes one row per group to retain more blocks.
  These thresholds are an unmeasured starting policy, not GPU-tuned results.
- Each lane reads a coalesced 16-byte weight vector once and reuses its decoded values across all M
  activation rows. Paired output rows share each float4 activation load and the weight scale. Their
  independent accumulators also expose more instruction parallelism. Scales can be shared because each
  pair starts on an even row and cannot cross a 32-row scale block.
- The 2/4-warp paths reduce FP32 partials in block-local shared memory, then round once to BF16.
  The one-warp path writes directly. There are no atomics, global partial sums or caller reduction workspace.
  Both block barriers are reached by all threads, including inactive output rows; the second barrier
  protects partials on grid-stride iterations. The grid is capped at 65535 blocks.
- K is divisible by 32: every 16-byte load is in bounds and stays within one scale block. Vector loads
  require both weight and activation bases aligned to 16 bytes. Naturally aligned slices fall back to
  scalar loads. Row tails are masked without returning before a barrier. Integer geometry checks precede
  all launches and wrapper allocation. This is source inspection, not generated-code inspection.
- GPU checks are written for bitwise bytes/scales/dequantized values, all required shapes and m values,
  wrapper/q equality, CPU-quantized inputs, misaligned weight and activation slices, default/nondefault
  streams, two concurrent streams, paired-row tails, all split policies and the capped grid-stride path.
- Graph checks capture quantize + two different weights, and separately two GEMVs consuming a previously
  quantized activation. They require exactly three/two kernel nodes (no allocation nodes), replay twice
  with changed inputs, compare both outputs at `2e-3`, and check that GEMV preserves the quantized input.
  These 36 graph replay cases are written but not run. CUDA capture is tested on a nonblocking stream;
  ordinary calls remain asynchronous on the default stream.
- The original independent CPU reference, error definition and all numerical tolerances are unchanged.
  Build registration and Python reference files are unchanged. No new dependencies or tensor-core
  instructions were introduced.

## RTX 4090 measurements

Old measurements below are copied from `K1b-fp8-gemv-speed.md` (2026-10-05). They include the old wrapper.
New columns are for `fp8_block_gemv_q` alone and are pending actual GPU measurement, not estimates.

| Weight | N | K | old m1 us | new m1 us | old GB/s | new GB/s | old m8/m1 | new m8/m1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| attn.wq_a | 1280 | 5120 | 19.5 | pending | 337 | pending | 1.84 | pending |
| attn.wq_b | 32768 | 1280 | 70.6 | pending | 595 | pending | 2.13 | pending |
| attn.wkv | 512 | 5120 | 14.3 | pending | 183 | pending | 1.64 | pending |
| attn.wo_b | 5120 | 8192 | 69.6 | pending | 603 | pending | 2.12 | pending |
| attn.indexer.wq_b | 4096 | 1280 | 16.4 | pending | 320 | pending | 1.86 | pending |
| shared.w1/w3 | 2304 | 5120 | 26.5 | pending | 446 | pending | 1.94 | pending |
| shared.w2 | 5120 | 2304 | 25.6 | pending | 461 | pending | 1.88 | pending |
| engram.wkv | 25600 | 6144 | 236.5 | pending | 666 | pending | 2.20 | pending |

The GPU executable prints this comparison table with the new measured columns. Timing takes the median
of 11 CUDA-event samples after warmup. A buffer at least twice the device's L2 size is overwritten before
start events. Only q-GEMV is timed: activation quantization, allocation and L2 eviction are excluded.
Effective bandwidth counts weight bytes plus weight-scale bytes. The temporary allocation-pool setting
is retained for compatibility-wrapper tests and restored at the end; the q-only call does not use it.

The executable labels each shape's target PASS/MISS against 715.5 GB/s (75% of 954) for N*K >= 5e6,
10 us for wkv/indexer.wq_b, and m8/m1 <= 1.5. Its exit status still represents numerical correctness,
so a numerical pass does not imply speed acceptance. The reviewer must inspect speed results on a 4090.

## Checks actually run on the final sources

The commands below were invoked from the repository root. Outputs are copied from captured logs.
The sanitizer timeout is the check runner terminating the process, not a test pass.

### Tool availability

```sh
python3 -c 'import platform,shutil,sys; print('"'"'platform:'"'"',platform.platform()); print('"'"'python:'"'"',sys.version.split()[0]); print('"'"'nvcc:'"'"',shutil.which('"'"'nvcc'"'"')); print('"'"'cmake:'"'"',shutil.which('"'"'cmake'"'"'))'
```

Result: exit 0.

```text
platform: macOS-26.5-arm64-arm-64bit-Mach-O
python: 3.14.3
nvcc: None
cmake: None
```

### Compiler version

```sh
clang++ --version
```

Result: exit 0.

```text
Apple clang version 17.0.0 (clang-1700.4.4.1)
Target: arm64-apple-darwin25.5.0
Thread model: posix
InstalledDir: /Users/jackwu/Applications/Xcode-26.1.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
```

### PyTorch availability

```sh
python3 -c 'import torch; print(torch.__version__)'
```

Result: exit 1.

```text
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import torch; print(torch.__version__)
    ^^^^^^^^^^^^
ModuleNotFoundError: No module named 'torch'
```

### Optimized host-only build

```sh
clang++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -DSTRATA_DS41_HOST_ONLY -Iinclude src/ds41/kernels/fp8_gemv_parity.cpp -O2 -o /tmp/k1b-host
```

Result: exit 0.

No compiler diagnostics.

### Host conversion, layout and boundary checks

```sh
/tmp/k1b-host --host-selftest
```

Result: exit 0.

```text
host layout: 288 geometries, each output/weight owned once, tails and grid-stride cap OK
host conversions: 65280 finite BF16 inputs, 256 E4M3 codes, 256 E8M0 codes, 756 FP8 tie probes OK
host BF16: 32639 positive boundaries and their negatives, ties and adjacent doubles OK
host lane model: 56 cases, max relative L2=9.32403817e-08, tolerance=0.002 OK
```

### All required shapes on the CPU

```sh
/tmp/k1b-host --host-shapes
```

Result: exit 0.

```text
host layout: 288 geometries, each output/weight owned once, tails and grid-stride cap OK
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
host model paired_tail        N=1025 K=96 m=1 rel=0
host model paired_tail        N=1025 K=96 m=2 rel=0
host model paired_tail        N=1025 K=96 m=3 rel=0
host model paired_tail        N=1025 K=96 m=4 rel=0
host model paired_tail        N=1025 K=96 m=5 rel=0
host model paired_tail        N=1025 K=96 m=6 rel=0
host model paired_tail        N=1025 K=96 m=7 rel=0
host model paired_tail        N=1025 K=96 m=8 rel=0
host model split2_tail        N=2049 K=544 m=1 rel=0
host model split2_tail        N=2049 K=544 m=2 rel=0
host model split2_tail        N=2049 K=544 m=3 rel=0
host model split2_tail        N=2049 K=544 m=4 rel=0
host model split2_tail        N=2049 K=544 m=5 rel=0
host model split2_tail        N=2049 K=544 m=6 rel=0
host model split2_tail        N=2049 K=544 m=7 rel=0
host model split2_tail        N=2049 K=544 m=8 rel=0
host model split1_tail        N=8193 K=96 m=1 rel=0
host model split1_tail        N=8193 K=96 m=2 rel=0
host model split1_tail        N=8193 K=96 m=3 rel=9.32403817e-08
host model split1_tail        N=8193 K=96 m=4 rel=8.07490328e-08
host model split1_tail        N=8193 K=96 m=5 rel=7.2223931e-08
host model split1_tail        N=8193 K=96 m=6 rel=6.59312958e-08
host model split1_tail        N=8193 K=96 m=7 rel=6.10404383e-08
host model split1_tail        N=8193 K=96 m=8 rel=5.70983777e-08
host model attn.wq_a          N=1280 K=5120 m=1 rel=0
host model attn.wq_a          N=1280 K=5120 m=2 rel=2.30604307e-08
host model attn.wq_a          N=1280 K=5120 m=4 rel=1.63077272e-08
host model attn.wq_a          N=1280 K=5120 m=8 rel=1.15306398e-08
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
host model attn.wo_b          N=5120 K=8192 m=4 rel=0
host model attn.wo_b          N=5120 K=8192 m=8 rel=1.36889801e-07
host model attn.indexer.wq_b  N=4096 K=1280 m=1 rel=0
host model attn.indexer.wq_b  N=4096 K=1280 m=2 rel=0
host model attn.indexer.wq_b  N=4096 K=1280 m=4 rel=0
host model attn.indexer.wq_b  N=4096 K=1280 m=8 rel=0
host model shared.w1/w3       N=2304 K=5120 m=1 rel=1.09997901e-08
host model shared.w1/w3       N=2304 K=5120 m=2 rel=6.28549883e-08
host model shared.w1/w3       N=2304 K=5120 m=4 rel=4.44431352e-08
host model shared.w1/w3       N=2304 K=5120 m=8 rel=1.59665884e-05
host model shared.w2          N=5120 K=2304 m=1 rel=0
host model shared.w2          N=5120 K=2304 m=2 rel=1.10962324e-10
host model shared.w2          N=5120 K=2304 m=4 rel=2.05689089e-05
host model shared.w2          N=5120 K=2304 m=8 rel=1.45442959e-05
host model engram.wkv         N=25600 K=6144 m=1 rel=0.00010515002
host model engram.wkv         N=25600 K=6144 m=2 rel=7.43145659e-05
host model engram.wkv         N=25600 K=6144 m=4 rel=5.41661674e-05
host model engram.wkv         N=25600 K=6144 m=8 rel=3.83913255e-05
host lane model: 88 cases, max relative L2=0.00010515002, tolerance=0.002 OK
```

### UBSan host-only build

```sh
clang++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -DSTRATA_DS41_HOST_ONLY -Iinclude src/ds41/kernels/fp8_gemv_parity.cpp -O1 -g -fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer -o /tmp/k1b-host-ubsan
```

Result: exit 0.

No compiler diagnostics.

### UBSan execution

```sh
/tmp/k1b-host-ubsan --host-selftest
```

Result: exit 0.

```text
host layout: 288 geometries, each output/weight owned once, tails and grid-stride cap OK
host conversions: 65280 finite BF16 inputs, 256 E4M3 codes, 256 E8M0 codes, 756 FP8 tie probes OK
host BF16: 32639 positive boundaries and their negatives, ties and adjacent doubles OK
host lane model: 56 cases, max relative L2=9.32403817e-08, tolerance=0.002 OK
```

### ASan + UBSan host-only build

```sh
clang++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -DSTRATA_DS41_HOST_ONLY -Iinclude src/ds41/kernels/fp8_gemv_parity.cpp -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer -o /tmp/k1b-host-asan
```

Result: exit 0.

No compiler diagnostics.

### ASan + UBSan execution (incomplete)

```sh
/tmp/k1b-host-asan --host-selftest
```

Result: timed out after 30 seconds.

```text

TIMEOUT after 30 seconds; process terminated by check runner
```

### Host-only skip behavior

```sh
/tmp/k1b-host --selftest
```

Result: exit 77.

```text
SKIP: host-only build; CUDA was not compiled
```

### Invalid CLI argument

```sh
/tmp/k1b-host --unknown
```

Result: exit 2.

```text
usage: fp8_gemv_parity [--selftest | --host-selftest | --host-shapes]
```

### Source checks

An ad hoc Python script compared the scalar conversion helpers, production quantizer and independent
reference functions with `git show HEAD:<path>`, checked that allocation/free remains confined to the
wrapper, checked source for host synchronization/atomics/MMA, and compared the unchanged build and Python
reference files with HEAD. It exited 0 with this real output (source checks do not compile CUDA):

```text
Scalar conversion/scale helpers and production quantizer: byte-for-byte unchanged from HEAD
Independent CPU reference and error definition: byte-for-byte unchanged; tolerance remains 2e-3
Source audit: allocation/free only in compatibility wrapper; no host synchronization, atomics or MMA instructions
GPU tests written for kernel-only graph capture, replay with changed inputs, and two-weight activation reuse
Build registration and Python references: unchanged
```

## Reviewer work remaining

The following commands are provided for the reviewer and were **not run here**:

```sh
cmake -S . -B /tmp/strata-k1b-cuda -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES='86;89;120' -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/strata-k1b-cuda --target fp8_gemv_parity -j
ctest --test-dir /tmp/strata-k1b-cuda -R '^fp8_gemv_parity$' --output-on-failure -V
compute-sanitizer --tool memcheck /tmp/strata-k1b-cuda/fp8_gemv_parity --selftest
compute-sanitizer --tool racecheck /tmp/strata-k1b-cuda/fp8_gemv_parity --selftest
compute-sanitizer --tool synccheck /tmp/strata-k1b-cuda/fp8_gemv_parity --selftest
```

Use CUDA 12.8 and verify all three architecture builds. On the 4090, paste the executable's measured
table into the pending columns, inspect register counts/spills and the shape dispatch if a speed target
misses, and check all numerical and graph tests. No speed-up is claimed from the CPU tests.
Assumptions are in `K1b.QUESTIONS.md`; the two reviewer commit groups are in `K1b.COMMITS.md`.

## Final scope check

The branch/scope/whitespace script exited 0 with:

```text
Branch: feature/ds41-k1b-fp8-gemv-speed
Scope: exactly 7 permitted files changed (3 sources, 4 K1b documents)
Whitespace: git diff --check and all changed/new-file trailing-whitespace checks pass
```

Every progress item was reviewed. The only unfinished validation items are explicitly identified in
`K1b.PROGRESS.md`: ASan runtime, direct PyTorch comparison, and CUDA/GPU acceptance.
