# DS41 kernel review fixes

Base: `e2d8499`, PR #10. Branch: `fix/ds41-revfix-kernels`.
Local host: Darwin arm64. Compiler: Apple clang 17.0.0.

All five findings in `rev_kernels_out.md` and finding 8 in `rev_tiers_out.md`
are real. Each was checked against the source before changes.
The implementation and host regressions are complete. Target validation is not complete.
No changes were made to `src/ds41/engine.cu` or `src/ds41/ds41_generate.cpp`.
The two review input files were already untracked. They were left unchanged.

## State by finding

| Finding | Code written | Compiled locally | Tested on RTX 4090 / Linux | Not done |
|---|---|---|---|---|
| Kernels 1: staging | Yes | Actual registry/staging source, extracted from the CPU file | No | Full Linux CPU translation unit and target run |
| Kernels 2: shapes | Yes | Actual raw registration source | No | Full Linux CPU run; optional Torch entry point build |
| Kernels 3: non-finite errors | Yes | Shared acceptance helper and negative tests | No | Four CUDA test translation units and GPU runs |
| Kernels 4: top-k IDs | Yes | Shared acceptance helper and negative tests | No | K5/K14 CUDA test translation units and GPU runs |
| Kernels 5: failed registration | Yes | Actual raw registration source with allocation failure injection | No | Full Linux CPU run; optional Torch entry point build |
| Tiers 8: E8M0 | Yes | Actual `deq` body with host scalar type models | No | CUDA compilation, bitwise GPU parity, sanitizer, and timing |

Host fixtures are synthetic. No real model inference was run.
The extracted CPU harness does not compile the SIMD compute code or production pool.
The scalar E8M0 harness models CUDA FP8/BF16 conversions. It does not execute CUDA.
The 10 portable CTest passes below are additional repository checks. They do not replace target acceptance.

## Fixes and regressions

1. **Mixed-K staging.** `stage_phase` used expert 0's byte counts while copying
   each selected expert's actual byte count. The new staging plan stores one
   matrix pointer and one prefix offset per selected projection before starting
   workers. It retains expert-major gate/up/down order. It checks IDs and size
   overflow before copying. The regression checks exact bytes and both guards,
   K1 versus K6 in both directions, per-projection rates, reordered and repeated
   selections, gated/gateless layers, native/swizzled descriptors, and 1/4 threads.
   Baseline: failed. Fixed source: passed, including UBSan.

2. **Layer shapes.** Registration checked only the first up/down pair.
   Both registration entry points now check every gate/up as H-to-I and every
   down as I-to-H. Dimensions must be positive. Raw tile dimensions are checked
   before multiplication to avoid signed overflow. Rates remain independent.
   The regression changes both dimensions of every projection of every expert.
   It covers zero, negative, mismatched, and overflowing tile counts.
   Valid mixed-rate gated/gateless layers still register.
   Baseline: 66 failures. Fixed raw source: passed, including UBSan.

3. **Non-finite errors.** K2, K1c, K15, and prefill routing/NLL used `std::max`
   to collect errors. It could discard NaN. They now use `max_error`, which
   maps any non-finite input or accumulator to positive infinity. A later finite
   error cannot clear the failure. Routing also checks non-finite weight errors
   on rows with different expert IDs. Finite differences on those rows remain
   excluded, as before. The host regression injects NaN and both infinities in
   both positions. Finite maxima remain unchanged.
   The old acceptance logic was first moved into the helper without a behavior
   change. The new negative test failed before the helper was fixed.

4. **Top-k IDs.** K5/K14 accepted equal adjacent IDs and counted repeated hits.
   Valid prefixes must now be strictly increasing and within `[offset, offset+t)`.
   Overlap counts the intersection of two sets. K14 still requires -1 padding.
   The negative test rejects duplicates, descending IDs, and both range bounds.
   Four copies of one correct ID count as one hit. Empty rows and valid prefixes
   remain accepted. The old helper behavior failed the new negative checks.
   All existing numerical and overlap tolerances remain unchanged.

5. **Registration ownership.** Both registration entry points now hold a
   `unique_ptr<MoeCpuLayer>` until `g_layers.push_back` succeeds. Only then does
   `release()` transfer ownership to the registry. The regression counts live
   allocations across repeated invalid-rate, null-pointer, and shape failures.
   Separate processes inject `bad_alloc` at each allocation, including registry
   growth. Injection points 0..10 fail registration cleanly; point 11 reaches
   success. Baseline: leaks at points 1..10. Fixed raw source: passed with UBSan.

6. **E8M0 boundaries.** Code 0 now uses float bits `0x00400000` (2^-127).
   Code 255 uses a quiet NaN. Codes 1..254 retain exponent-bit decoding.
   Codes 0..2 also need BF16 subnormal rounding before decode accumulation.
   Without it, a valid scale fix alone still differs from the stored BF16 path.
   The scalar regression checks all 254 finite E4M3 encodings at scale codes
   0, 1, 2, 112, 127, 254, and 255. Baseline: 544 mismatches. Fixed: zero.
   `wo_a_fp8_test.cu` retains its original random case, timing, and bitwise checks.
   It adds codes 0, 1, 2, 254, and 255. Dequant tests signed weights, zero,
   subnormals, and overflow. Decode uses positive weights to avoid cancellation;
   its code-254 case uses small inputs to keep the output finite. Valid scale
   cases require bit-for-bit BF16 parity. Code 255 requires NaN, not a particular
   NaN payload. GPU parity and performance are still unmeasured.

The CPU vendor directory has no `strata.patch`. Its README now records these
changes and test commands. The GPU vendor tree is unchanged. Its existing patch
was reversed in a temporary copy with `git apply --unidiff-zero --reverse`.
All 15 upstream hashes matched. The existing K12 host audit also passed.
BSD `patch` was not suitable for this zero-context patch; its initial attempt
was discarded. No vendor source was changed by that check.

## Local commands and real output

All commands ran from `/Users/jackwu/Projects/Strata-DS-revfix-kernels`.

### CPU regressions: before and after

```sh
python3 third_party/exllamav3_moe/tests/check_revfix_host.py --source-ref e2d8499
python3 third_party/exllamav3_moe/tests/check_revfix_host.py --sanitize undefined
```

Baseline output below is filtered to compilation, result, and process-exit lines.
The baseline runner exits 1. Each case runs in a separate process.

```text
COMPILED extracted registry and staging
RESULT fail stage (48 failures)
EXIT stage: 1
RESULT fail shape (66 failures)
EXIT shape: 1
RESULT fail leak (35 failures)
EXIT leak: 1
RESULT pass alloc (0 failures)
EXIT alloc 0: 0
RESULT fail alloc (1 failures)
EXIT alloc 1: 1
RESULT fail alloc (1 failures)
EXIT alloc 2: 1
RESULT fail alloc (1 failures)
EXIT alloc 3: 1
RESULT fail alloc (1 failures)
EXIT alloc 4: 1
RESULT fail alloc (1 failures)
EXIT alloc 5: 1
RESULT fail alloc (1 failures)
EXIT alloc 6: 1
RESULT fail alloc (1 failures)
EXIT alloc 7: 1
RESULT fail alloc (1 failures)
EXIT alloc 8: 1
RESULT fail alloc (1 failures)
EXIT alloc 9: 1
RESULT fail alloc (1 failures)
EXIT alloc 10: 1
registration reached success
RESULT pass alloc (0 failures)
EXIT alloc 11: 0
```

Fixed source (runner exit 0):

```text
COMPILED extracted registry and staging
RESULT pass stage (0 failures)
EXIT stage: 0
RESULT pass shape (0 failures)
EXIT shape: 0
RESULT pass leak (0 failures)
EXIT leak: 0
RESULT pass alloc (0 failures)
EXIT alloc 0: 0
RESULT pass alloc (0 failures)
EXIT alloc 1: 0
RESULT pass alloc (0 failures)
EXIT alloc 2: 0
RESULT pass alloc (0 failures)
EXIT alloc 3: 0
RESULT pass alloc (0 failures)
EXIT alloc 4: 0
RESULT pass alloc (0 failures)
EXIT alloc 5: 0
RESULT pass alloc (0 failures)
EXIT alloc 6: 0
RESULT pass alloc (0 failures)
EXIT alloc 7: 0
RESULT pass alloc (0 failures)
EXIT alloc 8: 0
RESULT pass alloc (0 failures)
EXIT alloc 9: 0
RESULT pass alloc (0 failures)
EXIT alloc 10: 0
registration reached success
RESULT pass alloc (0 failures)
EXIT alloc 11: 0
```

### Acceptance helper regressions

Before the fix, the helper retained the original `std::max`, `std::is_sorted`,
and per-element membership count. The test was compiled and run first:

```sh
clang++ -std=c++17 -Wall -Wextra -Werror src/ds41/tests/test_validation_test.cpp -o /tmp/revfix-validation
/tmp/revfix-validation
```

```text
PASS finite errors unchanged
FAIL non-finite error fails tolerance
FAIL later finite error cannot hide failure
PASS non-finite accumulator fails tolerance
PASS non-finite error fails tolerance
PASS later finite error cannot hide failure
PASS non-finite accumulator fails tolerance
FAIL non-finite error fails tolerance
FAIL later finite error cannot hide failure
FAIL non-finite accumulator fails tolerance
PASS strict valid IDs accepted
PASS complete set intersection
FAIL duplicate IDs rejected
FAIL duplicates count once in set intersection
PASS descending IDs rejected
FAIL out-of-range IDs rejected
PASS valid prefix and empty row accepted
PASS padding cannot enter valid prefix
RESULT fail (8 failures)
```

After the fix:

```sh
clang++ -std=c++17 -Wall -Wextra -Werror -fsanitize=undefined src/ds41/tests/test_validation_test.cpp -o /tmp/revfix-validation-ubsan
/tmp/revfix-validation-ubsan
```

```text
PASS finite errors unchanged
PASS non-finite error fails tolerance
PASS later finite error cannot hide failure
PASS non-finite accumulator fails tolerance
PASS non-finite error fails tolerance
PASS later finite error cannot hide failure
PASS non-finite accumulator fails tolerance
PASS non-finite error fails tolerance
PASS later finite error cannot hide failure
PASS non-finite accumulator fails tolerance
PASS strict valid IDs accepted
PASS complete set intersection
PASS duplicate IDs rejected
PASS duplicates count once in set intersection
PASS descending IDs rejected
PASS out-of-range IDs rejected
PASS valid prefix and empty row accepted
PASS padding cannot enter valid prefix
RESULT pass (0 failures)
```

### E8M0 scalar regression

```sh
python3 src/ds41/tests/check_wo_a_fp8_host.py --source-ref e2d8499
python3 src/ds41/tests/check_wo_a_fp8_host.py
```

Before (exit 1):

```text
FAIL E8M0 code 0: 244 mismatches among 254 finite FP8 values
FAIL E8M0 code 1: 32 mismatches among 254 finite FP8 values
FAIL E8M0 code 2: 16 mismatches among 254 finite FP8 values
PASS E8M0 code 112: 0 mismatches among 254 finite FP8 values
PASS E8M0 code 127: 0 mismatches among 254 finite FP8 values
PASS E8M0 code 254: 0 mismatches among 254 finite FP8 values
FAIL E8M0 code 255: 252 mismatches among 254 finite FP8 values
RESULT fail scalar dequant (544 failures)
```

After (exit 0):

```text
PASS E8M0 code 0: 0 mismatches among 254 finite FP8 values
PASS E8M0 code 1: 0 mismatches among 254 finite FP8 values
PASS E8M0 code 2: 0 mismatches among 254 finite FP8 values
PASS E8M0 code 112: 0 mismatches among 254 finite FP8 values
PASS E8M0 code 127: 0 mismatches among 254 finite FP8 values
PASS E8M0 code 254: 0 mismatches among 254 finite FP8 values
PASS E8M0 code 255: 0 mismatches among 254 finite FP8 values
RESULT pass scalar dequant (0 failures)
```

### Existing host checks

```sh
python3 third_party/exllamav3_moe/tests/check_mixedk_host.py --registry-only
python3 third_party/exllamav3_moe/tests/check_mixedk_host.py
python3 src/ds41/kernels/k12/check_host.py
```

```text
PASS extracted raw CPU registry: 6 mixed experts, per-projection K1..K6, relocation, atomic rate rejection, native/swizzled descriptors
```

```text
PASS extracted raw CPU registry: 6 mixed experts, per-projection K1..K6, relocation, atomic rate rejection, native/swizzled descriptors
PASS x86_64 band harness compilation (extracted source; no production pool)
SKIP: AVX2/FMA unavailable
NOT RUN: x86 band numerical checks require AVX2/FMA
```

```text
PASS vendor: 15 pristine hashes, reverse/forward patch, unchanged reconstruct device code
PASS source audit: K10 activation/casts and scale rounding preserved; 15 local includes, no Torch
PASS native C++17 layout: every base alignment, row-tile tails, disjoint buffers, INT_MAX bounded size
PASS host routing model: empty groups, nonzero offsets, 512-row tails, duplicate tokens and untouched rows
HOST AUDITS PASSED; CUDA compilation, GPU numerical acceptance, graph capture and timing NOT RUN
```

Independent reverse-patch hash check:

```text
PASS GPU vendor reverse patch: 15 upstream SHA-256 hashes match
```

### Repository build and portable CTest

```sh
cmake -S . -B /tmp/revfix-cmake -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=ON
cmake --build /tmp/revfix-cmake -j 4
```

```text
-- The CXX compiler identification is AppleClang 17.0.0.17000604
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
-- Found Threads: TRUE
-- Configuring done (0.5s)
-- Generating done (0.0s)
-- Build files have been written to: /tmp/revfix-cmake
```

The full build failed. Relevant real output:

```text
clang++: error: unsupported option '-mavx512f' for target 'arm64-apple-darwin25.5.0'
clang++: error: unsupported option '-mavx512bw' for target 'arm64-apple-darwin25.5.0'
clang++: error: unsupported option '-mavx512vl' for target 'arm64-apple-darwin25.5.0'
clang++: error: unsupported option '-mavx512dq' for target 'arm64-apple-darwin25.5.0'
clang++: error: unsupported option '-mavx512vnni' for target 'arm64-apple-darwin25.5.0'
clang++: error: unsupported option '-mavx512vbmi' for target 'arm64-apple-darwin25.5.0'
clang++: error: unsupported option '-mf16c' for target 'arm64-apple-darwin25.5.0'
clang++: error: unsupported option '-mfma' for target 'arm64-apple-darwin25.5.0'
make[2]: *** [CMakeFiles/strata_kernels_cpu.dir/src/kernels/cpu/expert.cpp.o] Error 1
make[1]: *** [CMakeFiles/strata_kernels_cpu.dir/all] Error 2
make: *** [all] Error 2
```

The available portable tests were then built explicitly:

```sh
cmake --build /tmp/revfix-cmake -j 4 --target ds41_suffix_drafter_test ds41_generate_suffix_mock_test gguf_reader_test gguf_split_test suffix_drafter_test controller_test draft_policy_test conv_cache_test coupled_draft_test bf16_bits_test
ctest --test-dir /tmp/revfix-cmake --output-on-failure -R '^(ds41_suffix_drafter_test|ds41_generate_suffix_mock_test|gguf_reader_test|gguf_split_test|suffix_drafter_test|controller_test|draft_policy_test|conv_cache_test|coupled_draft_test|bf16_bits_test)$'
```

```text
[ 50%] Built target strata_ds41_suffix
[ 75%] Building CXX object CMakeFiles/ds41_suffix_drafter_test.dir/src/ds41/tests/suffix_drafter_test.cpp.o
[100%] Linking CXX executable ds41_suffix_drafter_test
ld: warning: search path '/opt/homebrew/opt/tcl-tk/lib' not found
[100%] Built target ds41_suffix_drafter_test
[ 50%] Built target strata_ds41_suffix
[ 75%] Building CXX object CMakeFiles/ds41_generate_suffix_mock_test.dir/src/ds41/tests/generate_suffix_mock_test.cpp.o
[100%] Linking CXX executable ds41_generate_suffix_mock_test
ld: warning: search path '/opt/homebrew/opt/tcl-tk/lib' not found
[100%] Built target ds41_generate_suffix_mock_test
[100%] Built target gguf_reader_test
[100%] Built target gguf_split_test
[ 66%] Built target strata_spec
[ 83%] Building CXX object CMakeFiles/suffix_drafter_test.dir/src/spec/suffix_drafter_test.cpp.o
[100%] Linking CXX executable suffix_drafter_test
ld: warning: search path '/opt/homebrew/opt/tcl-tk/lib' not found
[100%] Built target suffix_drafter_test
[ 66%] Built target strata_spec
[ 83%] Building CXX object CMakeFiles/controller_test.dir/src/spec/controller_test.cpp.o
[100%] Linking CXX executable controller_test
ld: warning: search path '/opt/homebrew/opt/tcl-tk/lib' not found
[100%] Built target controller_test
[ 66%] Built target strata_spec
[ 83%] Building CXX object CMakeFiles/draft_policy_test.dir/src/spec/draft_policy_test.cpp.o
[100%] Linking CXX executable draft_policy_test
ld: warning: search path '/opt/homebrew/opt/tcl-tk/lib' not found
[100%] Built target draft_policy_test
[100%] Built target conv_cache_test
[100%] Built target coupled_draft_test
[100%] Built target bf16_bits_test
```

```text
Test project /tmp/revfix-cmake
      Start  1: ds41_suffix_drafter_test
 1/10 Test  #1: ds41_suffix_drafter_test .........   Passed    3.15 sec
      Start  2: ds41_generate_suffix_mock_test
 2/10 Test  #2: ds41_generate_suffix_mock_test ...   Passed    0.13 sec
      Start  3: gguf_reader_test
 3/10 Test  #3: gguf_reader_test .................   Passed    0.12 sec
      Start  4: gguf_split_test
 4/10 Test  #4: gguf_split_test ..................   Passed    0.13 sec
      Start  5: suffix_drafter_test
 5/10 Test  #5: suffix_drafter_test ..............   Passed    0.12 sec
      Start  6: controller_test
 6/10 Test  #6: controller_test ..................   Passed    0.12 sec
      Start  7: draft_policy_test
 7/10 Test  #7: draft_policy_test ................   Passed    0.12 sec
      Start  8: conv_cache_test
 8/10 Test  #8: conv_cache_test ..................   Passed    0.12 sec
      Start  9: coupled_draft_test
 9/10 Test  #9: coupled_draft_test ...............   Passed    0.12 sec
      Start 10: bf16_bits_test
10/10 Test #10: bf16_bits_test ...................   Passed    5.23 sec

100% tests passed, 0 tests failed out of 10

Total Test time (real) =   9.37 sec
```

## Limitations

- I could not run CUDA compilation, GPU acceptance, Compute Sanitizer, or GPU
  timing, because this Mac has no `nvcc` or NVIDIA device. No target session was
  available. The CUDA changes are written, but are not reported as compiled.
- I could not run the complete production CPU layer here, because it targets
  x86 Linux/Windows. The full local build rejects the x86 compiler flags on arm64.
- I could not run the existing x86 band numerical checks, because the translated
  process reported no AVX2/FMA. Its extracted x86_64 harness compiled, then skipped.
- I could not complete ASan execution, because both ASan binaries stalled before
  test output on this host. They compiled. I interrupted them with Ctrl-C.
  The runtime cause is unknown. UBSan and unsanitized executions passed.
  At the time of the attempt the CPU runner used a boolean `--sanitize` flag;
  the final equivalent is `--sanitize address,undefined`.
- I could not run the K10 host audit, because the available Python lacks NumPy.
  The K12 host audit and independent vendor hash check passed.
- I could not create commits, because the sandbox rejected the Git index write.
  `REVFIX.COMMITS.md` contains five commit units, each with at most five files
  and the required co-author trailer. No protected engine files were changed.

K10 audit output:

```text
Traceback (most recent call last):
  File "/Users/jackwu/Projects/Strata-DS-revfix-kernels/src/ds41/kernels/k10/check_host.py", line 15, in <module>
    import numpy as np
ModuleNotFoundError: No module named 'numpy'
```

ASan CPU attempt (interrupted, no test pass):

```text
COMPILED extracted registry and staging
Traceback (most recent call last):
  File "/Users/jackwu/Projects/Strata-DS-revfix-kernels/third_party/exllamav3_moe/tests/check_revfix_host.py", line 52, in <module>
    r = subprocess.run([str(tmp / "test"), *case])
  File "/opt/homebrew/Cellar/python@3.14/3.14.7/Frameworks/Python.framework/Versions/3.14/lib/python3.14/subprocess.py", line 557, in run
    stdout, stderr = process.communicate(input, timeout=timeout)
                     ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Cellar/python@3.14/3.14.7/Frameworks/Python.framework/Versions/3.14/lib/python3.14/subprocess.py", line 1213, in communicate
    self.wait()
    ~~~~~~~~~^^
  File "/opt/homebrew/Cellar/python@3.14/3.14.7/Frameworks/Python.framework/Versions/3.14/lib/python3.14/subprocess.py", line 1279, in wait
    return self._wait(timeout=timeout)
           ~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/opt/homebrew/Cellar/python@3.14/3.14.7/Frameworks/Python.framework/Versions/3.14/lib/python3.14/subprocess.py", line 2084, in _wait
    (pid, sts) = self._try_wait(0)
                 ~~~~~~~~~~~~~~^^^
  File "/opt/homebrew/Cellar/python@3.14/3.14.7/Frameworks/Python.framework/Versions/3.14/lib/python3.14/subprocess.py", line 2042, in _try_wait
    (pid, sts) = os.waitpid(self.pid, wait_flags)
                 ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt
```

## Exact Linux RTX 4090 commands

These commands are pending. Use the `revfix-kernels` checkout below. If the box
uses another tree suffix, change only the `cd` path. CMake consumes
`CMAKE_CUDA_ARCHITECTURES`; the shell variable `CUDA_ARCH` is set to 89.

```sh
cd /workspace/Strata-DS-revfix-kernels
export CUDA_ARCH=89
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_NATIVE_EXPERTS=OFF -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
cmake --build build -j 8 --target \
  ds41_test_validation moe_revfix_test moe_set_expert_test \
  k2_fp8_gemm_test k1c_fp8_gemv_test k15_hc_prefill_test prefill_ops_test \
  k5_indexer_test k14_indexer_prefill_test wo_a_fp8_test \
  k11_cpu_moe_test ds41_generate
```

Findings 1, 2, and 5: run the full native CPU source, then the sanitizer build.
The CTest cases cover staging, shapes, leaks, and all allocation failure points.

```sh
ctest --test-dir build --output-on-failure -R '^(moe_revfix_.*|moe_set_expert_test)$'
CXX=g++ python3 third_party/exllamav3_moe/tests/check_revfix_host.py --native --sanitize address,undefined
```

To reproduce the native CPU baseline failures without changing the checkout:

```sh
cpu_baseline=$(mktemp -d /tmp/revfix-cpu-before.XXXXXX)
git show e2d8499:third_party/exllamav3_moe/moe_mul1.cpp > "$cpu_baseline/moe_mul1.cpp"
g++ -std=c++17 -O1 -pthread -I third_party/exllamav3_moe   "$cpu_baseline/moe_mul1.cpp" third_party/exllamav3_moe/tests/revfix_test.cpp   -o "$cpu_baseline/test"
# These three baseline cases must return 1.
for case in stage shape leak; do
  "$cpu_baseline/test" "$case"
  result=$?
  test "$result" -eq 1 || exit 1
done
for n in $(seq 1 10); do
  "$cpu_baseline/test" alloc "$n"
  result=$?
  test "$result" -eq 1 || exit 1
done
```

Finding 3: run the negative host assertions and all four unchanged-tolerance GPU suites.

```sh
ctest --test-dir build --output-on-failure -R '^(ds41_test_validation|k2_fp8_gemm_test|k1c_fp8_gemv_test|k15_hc_prefill_test|prefill_ops_test)$'
```

Finding 4: the host test contains duplicate-ID and range negative checks.

```sh
ctest --test-dir build --output-on-failure -R '^(ds41_test_validation|k5_indexer_test|k14_indexer_prefill_test)$'
```

Tiers finding 8: scalar boundary checks, bitwise GPU parity, and device memory checks.

```sh
CXX=g++ python3 src/ds41/tests/check_wo_a_fp8_host.py
ctest --test-dir build --output-on-failure -R '^wo_a_fp8_test$'
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/wo_a_fp8_test
```

For a CUDA red run, create an isolated baseline tree and copy only the expanded
regression test. This leaves the working checkout intact. Expect test failure
on scale 0 and reserved scale 255. Do not interpret a configure/build error as
an expected regression failure.

```sh
cuda_baseline=$(mktemp -d /tmp/revfix-cuda-before.XXXXXX)
git archive e2d8499 | tar -x -C "$cuda_baseline"
cp src/ds41/tests/wo_a_fp8_test.cu "$cuda_baseline/src/ds41/tests/wo_a_fp8_test.cu"
cmake -S "$cuda_baseline" -B "$cuda_baseline/build" -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_NATIVE_EXPERTS=OFF -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
cmake --build "$cuda_baseline/build" -j 8 --target wo_a_fp8_test
"$cuda_baseline/build/wo_a_fp8_test"
result=$?
test "$result" -eq 1
```

Existing real-weight CPU acceptance requires the box's pack and golden fixtures:

```sh
K11_PACK=/workspace/pack-3bpw K11_GOLDEN=/workspace/ci/golden/k10 MIXEDK_GOLDEN=/workspace/ci/golden/mixedk K11_THREADS=8 K11_WEIGHTS=all ./build/k11_cpu_moe_test
```

Missing fixtures are a blocked check, not a pass. Any CTest `Skipped` result is
also untested. Record the target build, test, sanitizer, and timing output before
claiming target completion. Compare `wo_a` timing with the baseline on the same
box; no performance claim is made here.

## Final scope audit

The commit-plan file list was compared with `git status --porcelain`.
The threshold expressions in all six modified acceptance suites were compared
with `e2d8499`. The protected paths were checked with `git diff --name-only`.

```text
PASS commit plan: file counts 5, 5, 4, 3, 2; all 19 task files covered
PASS existing numeric and overlap thresholds unchanged
PASS protected engine files unchanged
PASS git diff --check
```
