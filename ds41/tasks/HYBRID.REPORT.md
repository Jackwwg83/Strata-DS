# Hybrid decode report

Code is written. CUDA compilation and GPU acceptance remain unverified.
No throughput improvement is claimed. The only performance inputs below are the owner's measurements.

## State by part

| Part | Code written | Compiled here | Tested on GPU | Not done |
| --- | --- | --- | --- | --- |
| Mapped RAM arena and descriptor table | Yes | No | No | Linux build and table lifecycle test |
| Quota split and per-call descriptors | Yes | No | No | Split and graph acceptance |
| Engine integration | Yes | No | No | Real SAGE decode and overlap measurement |
| Timing and per-step print | Yes | Generator object and public Timing header | No | Validate live counts |
| Synthetic tests and CMake targets | Yes, tests first | No | No | Run both new targets on RTX 4090 |
| K10 arithmetic | Unchanged | No new build | No | Bitwise memory-location parity test |
| Local portable regression tests | Existing tests | Yes | Not applicable | None; 8/8 passed |
| Commits | Units prepared | Not applicable | Not applicable | Git metadata write is sandbox-blocked |

I could not run CUDA compilation or GPU tests, because this host is macOS arm64 and has no CUDA toolkit or NVIDIA GPU.
I could not run the SAGE forced decode, because this host has neither the RTX 4090 runtime nor `/workspace/pack-sage`.
I could not complete the full CPU build, because its x86 expert target requests AVX flags on Apple arm64.
I could not create commits, because Git cannot write the shared worktree metadata. See `HYBRID.COMMITS.md`.

## Changes

1. `HostExperts` registers its anonymous RAM arena with `cudaHostRegisterMapped | cudaHostRegisterPortable`.
   It obtains the device alias even when that alias equals the host pointer.
   A fixed device table has one `Exl3Expert` per `(layer, expert)`.
   Each descriptor comes from `VramExperts::describe_at` with the device alias and slot offset.
   Missing entries have null pointers. File mappings never enter this table.
   Registration or alias failure leaves the CPU path available and prints a diagnostic.
   Device table allocation or upload errors fail initialization explicitly.
2. `point_to_file()` clears the entry before a RAM slot is overwritten.
   `assign()` clears the old holder and publishes the new descriptor after the copy completes.
   Table uploads finish at these between-step safe points before a nonblocking stream can read them.
   `VramExperts::upload_res()` now also waits for its publication before decode or background copies start.
   Existing swap callbacks perform the RAM updates; their scheduling does not change.
3. Doorbell publish reads the VRAM residency table, RAM descriptors, and one device quota int.
   Each token keeps routing order. VRAM hits do not consume quota. File misses do not consume quota.
   The first q eligible RAM misses go to the GPU. CPU ids for both GPU paths become -1.
   The doorbell owns a device descriptor array of `max_m * topk` entries.
   Active GPU selection indices are the original route positions; inactive indices are -1.
   Thus one K10 call handles both memory tiers with the same arithmetic and reduction order.
   Legacy callers that omit descriptor tables still receive VRAM slot indices.
4. Engine init allocates a device quota array of 40 ints. Each layer passes its own element.
   `DS41_ZC_QUOTA` accepts 0..6 and defaults to 4 when mapped RAM is available.
   The engine can update these device values between steps without changing captured pointers.
   There is no automatic tuning loop. The environment is read at initialization.
   A RAM-only configuration gets its own K10 workspace; other configurations reuse the VRAM workspace.
   With no mapped RAM, routing follows the old CPU/VRAM split. With q=0, no RAM misses go to the GPU.
5. The CPU worker sees only its remaining ids. It clears its output when there are no CPU jobs.
   Publish also emits VRAM / zero-copy / CPU route-use counts under the existing seq fence.
   The worker adds CPU plus zero-copy uses to `worker_misses`, which now means non-VRAM uses.
   This preserves the existing `Timing::expert_hits` meaning without editing `step()`.
   `Timing::cpu_experts()` is RAM CPU plus file CPU uses.
   `Timing::zero_copy_experts()` is total minus VRAM minus CPU uses.
   The per-step line prints all three counts. Counts are uses, not distinct expert identities.

Default quota rationale: on the owner's RTX 4090 / PCIe 4.0 x16 measurements, four reads at 262 us cost about
1.05 ms. Two CPU experts at 500-600 us each cost about 1.0-1.2 ms. This roughly balances a layer with six RAM
misses. It does not account for VRAM work, shared experts, mixed expert sizes, launch cost, or contention.
Actual optimal quotas can differ by layer and PC. The default needs the sweep below.

`engine.cu` changes are limited to state declarations, init, `worker_loop()`, and `moe()`.
`step()`, prefill, and the reviewer's graph scheduling are unchanged.
Publish uses only its stream argument. The per-call descriptors and workspace exist before capture.
No new host synchronization, allocation, or table upload occurs in `moe()` or publish.
The current engine still supplies its existing default stream; the reviewer's stream change must pass the
same decode stream through publish, K10, shared work, and wait-add.
This change alone does not claim whole-engine CUDA graph support.

## Synthetic GPU acceptance

`hybrid_decode_test` needs no pack. It generates integer K1..K6 EXL3 projections with real 5120/2304 shapes,
random trellis bytes, signed scales, and nonzero inputs. It checks finite, nonzero results.
For every m from 1 through 8 and every q from 0 through 6, it checks:

- Mixed VRAM hits, RAM misses, and file misses; all-VRAM, all-RAM, and all-file cases.
- Exact CPU and GPU membership at each route position; inactive ids; weights and activation publication.
- Per-token quota, routing order, descriptor fields, cleared inactive descriptors, and partition counts.
- Bitwise K10 output equality against the same selected experts entirely in VRAM.
- One captured publish + K10 graph per m, replayed with changed routes, weights, residency, RAM entries, and quota.
- Separate null-table and null-quota fallback cases, including negative and oversized device quotas.

`hybrid_host_table_test` uses `fake_pack.hpp` with mixed sizes. Tiny fake experts are used only for metadata
and byte reads, never for K10 arithmetic. It checks zero budget, device aliases, untouched file entries,
repeated revoke/reassign, capacity rejection, fixed table address, and GPU reads on a nonblocking stream.
It also drives a real `VramExperts::between_steps()` swap through revoke, copy, and commit.
The existing `doorbell_test` remains the CPU handshake / wait-add regression.
Both new tests return 77 without a GPU. A skip is not GPU acceptance.

## Linux RTX 4090 commands

Run from the target repository. `STRATA_NATIVE_EXPERTS=OFF` avoids the unrelated upstream ggml fetch;
it does not disable the DS41 EXL3 CPU experts.

```sh
cd /workspace/Strata-DS
CUDA_ARCH=89
cmake -S . -B build-hybrid -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_NATIVE_EXPERTS=OFF -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
cmake --build build-hybrid -j 8 --target ds41_generate \
  hybrid_decode_test hybrid_host_table_test doorbell_test \
  host_experts_test vram_experts_test vram_lend_test
ctest --test-dir build-hybrid --output-on-failure \
  -R '^(hybrid_decode_test|hybrid_host_table_test|doorbell_test|host_experts_test|vram_experts_test|vram_lend_test)$'
compute-sanitizer --tool memcheck --error-exitcode 1 ./build-hybrid/hybrid_decode_test
compute-sanitizer --tool memcheck --error-exitcode 1 ./build-hybrid/hybrid_host_table_test
```

Record the GPU, driver, CUDA compiler, CPU, affinity, and available RAM with the test output.
Require both new targets to run, not skip. Also compile for sm_86 and sm_120 where the installed CUDA toolkit
supports them; only sm_89 runtime acceptance is available on an RTX 4090.

Use the same real comma-separated forced token file for every decode run. The commands below expect that
file at `/workspace/forced-ids.txt`. It must contain at least 200 valid tokens and at most 4096 tokens.
Do not use `--prefill`: this check must execute decode.

```sh
cd /workspace/Strata-DS
FILE=/workspace/forced-ids.txt
test -s "$FILE"
mkdir -p /workspace/hybrid-results
for q in 0 1 2 3 4 5 6; do
  DS41_ZC_QUOTA="$q" ./build-hybrid/ds41_generate \
    --pack /workspace/pack-sage --threads 30 --ram-budget-gib 100 \
    --expert-profile ds41/data/expert-profile.bin --adapt-every 0 \
    --max-seq 4096 --force-ids "$FILE" \
    > "/workspace/hybrid-results/q${q}.out" 2> "/workspace/hybrid-results/q${q}.log"
done
```

Keep residency fixed for the first sweep. Confirm the mapped-RAM startup message appears.
This parser checks each real per-step partition and prints counts and median step times:

```sh
python3 - <<'PY'
from pathlib import Path
import re
import statistics
root = Path('/workspace/hybrid-results')
rx = re.compile(r'total ([0-9.]+) ms .*vram hits (\d+)/(\d+) swaps \d+  zero-copy (\d+) cpu (\d+): ram (\d+) file (\d+)')
for q in range(7):
    rows = [tuple(map(float, m.groups())) for m in rx.finditer((root / f'q{q}.log').read_text())]
    assert len(rows) >= 200, (q, 'too few decoded positions', len(rows))
    for ms, vram, total, zc, cpu, ram, file in rows:
        assert total == 240 and vram + zc + cpu == total
        assert cpu == ram + file and 0 <= zc <= 40 * q
    if q:
        assert any(r[3] > 0 for r in rows), (q, 'zero-copy path was not exercised')
    print(q, 'steps', len(rows), 'median_ms_after_10', statistics.median(r[0] for r in rows[10:]),
          'vram_zc_cpu_uses', [int(sum(r[i] for r in rows)) for i in (1, 3, 4)])
PY
```

Then repeat q=0 and q=4 with adaptation enabled to exercise live tier updates:

```sh
for q in 0 4; do
  DS41_ZC_QUOTA="$q" ./build-hybrid/ds41_generate \
    --pack /workspace/pack-sage --threads 30 --ram-budget-gib 100 \
    --expert-profile ds41/data/expert-profile.bin --adapt-every 4 \
    --max-seq 4096 --force-ids "$FILE" \
    > "/workspace/hybrid-results/adapt-q${q}.out" 2> "/workspace/hybrid-results/adapt-q${q}.log"
done
```

Require nonzero committed swap counts in this second run. Otherwise the live swap path was not exercised.
For the no-RAM regression, run the following twice with q=0 and q=6 and compare the dump files:

```sh
for q in 0 6; do
  DS41_ZC_QUOTA="$q" ./build-hybrid/ds41_generate \
    --pack /workspace/pack-sage --threads 30 --ram-budget-gib 0 \
    --expert-profile ds41/data/expert-profile.bin --adapt-every 0 \
    --max-seq 4096 --force-ids "$FILE" --dump "/workspace/hybrid-results/no-ram-q${q}.bin" \
    > "/workspace/hybrid-results/no-ram-q${q}.out" 2> "/workspace/hybrid-results/no-ram-q${q}.log"
done
cmp /workspace/hybrid-results/no-ram-q0.bin /workspace/hybrid-results/no-ram-q6.bin
```

Compare q=0 with the pre-change engine using identical residency, pack, forced ids, and threads.
Require matching dumps for that regression. Across q>0, compare predictions and teacher-forced NLL as well as
latency. Moving work from the CPU to K10 can change rounding; whole-engine q=0 versus q>0 bitwise equality
is not the same acceptance criterion as mapped-memory versus VRAM K10 bitwise equality.
Measure overlap with a profiler before claiming the new path is faster.
No real sample, decode timing, GPU pass, or numerical parity result is available from this macOS session.

## Local commands and real output

The portable tests below are upstream regression checks. They do not compile or execute the hybrid CUDA code.

### Offline configuration (exit 0)

```sh
cmake -S . -B /tmp/strata-hybrid-cpu-build -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON -DSTRATA_NATIVE_EXPERTS=OFF
```

Actual output:

```text
-- Configuring done (0.1s)
-- Generating done (0.0s)
-- Build files have been written to: /tmp/strata-hybrid-cpu-build
```

### Portable target build (exit 0)

```sh
cmake --build /tmp/strata-hybrid-cpu-build --target gguf_reader_test gguf_split_test suffix_drafter_test controller_test draft_policy_test conv_cache_test coupled_draft_test bf16_bits_test -j 4
```

Actual output:

```text
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

### Portable tests (exit 0)

```sh
ctest --test-dir /tmp/strata-hybrid-cpu-build --output-on-failure -R '^(gguf_reader_test|gguf_split_test|suffix_drafter_test|controller_test|draft_policy_test|conv_cache_test|coupled_draft_test|bf16_bits_test)$'
```

Actual output:

```text
Test project /tmp/strata-hybrid-cpu-build
    Start 1: gguf_reader_test
1/8 Test #1: gguf_reader_test .................   Passed    0.48 sec
    Start 2: gguf_split_test
2/8 Test #2: gguf_split_test ..................   Passed    0.39 sec
    Start 3: suffix_drafter_test
3/8 Test #3: suffix_drafter_test ..............   Passed    0.40 sec
    Start 4: controller_test
4/8 Test #4: controller_test ..................   Passed    0.39 sec
    Start 5: draft_policy_test
5/8 Test #5: draft_policy_test ................   Passed    0.39 sec
    Start 6: conv_cache_test
6/8 Test #6: conv_cache_test ..................   Passed    0.38 sec
    Start 7: coupled_draft_test
7/8 Test #7: coupled_draft_test ...............   Passed    0.38 sec
    Start 8: bf16_bits_test
8/8 Test #8: bf16_bits_test ...................   Passed    5.48 sec

100% tests passed, 0 tests failed out of 8

Total Test time (real) =   8.30 sec
```

### CUDA configuration (exit 1)

```sh
cmake -S . -B /tmp/strata-hybrid-cuda-build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=89
```

Actual output:

```text
-- The CXX compiler identification is AppleClang 17.0.0.17000604
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
CMake Error at /opt/homebrew/share/cmake/Modules/Internal/CMakeCUDAFindToolkit.cmake:104 (message):
  Failed to find nvcc.

  Compiler requires the CUDA toolkit.  Please set the CUDAToolkit_ROOT
  variable.
Call Stack (most recent call first):
  /opt/homebrew/share/cmake/Modules/CMakeDetermineCUDACompiler.cmake:109 (cmake_cuda_find_toolkit)
  CMakeLists.txt:134 (enable_language)


-- Configuring incomplete, errors occurred!
```

### Full CPU build (exit 2)

```sh
cmake --build /tmp/strata-hybrid-cpu-build -j 4
```

Actual output:

```text
[ 10%] Building CXX object CMakeFiles/gguf_reader_test.dir/src/artifact/gguf_reader_test.cpp.o
[ 10%] Building CXX object CMakeFiles/strata-gguf.dir/src/artifact/gguf_reader.cpp.o
[ 10%] Building CXX object CMakeFiles/gguf_split_test.dir/tests/core/gguf_split_test.cpp.o
[ 10%] Building CXX object CMakeFiles/strata-dequant.dir/src/artifact/dequant.cpp.o
[ 12%] Linking CXX executable strata-dequant
[ 15%] Linking CXX executable strata-gguf
[ 17%] Linking CXX executable gguf_reader_test
ld: warning: search path '/opt/homebrew/opt/tcl-tk/lib' not found
ld: warning: search path '/opt/homebrew/opt/tcl-tk/lib' not found
ld: warning: search path '/opt/homebrew/opt/tcl-tk/lib' not found
[ 17%] Built target strata-dequant
[ 17%] Built target strata-gguf
[ 17%] Built target gguf_reader_test
[ 20%] Building CXX object CMakeFiles/strata-plan.dir/src/plan/plan_main.cpp.o
[ 23%] Building CXX object CMakeFiles/strata_spec.dir/src/spec/suffix_drafter.cpp.o
[ 25%] Building CXX object CMakeFiles/conv_cache_test.dir/src/program/conv_cache_test.cpp.o
[ 28%] Linking CXX executable conv_cache_test
[ 30%] Building CXX object CMakeFiles/strata_spec.dir/src/spec/controller.cpp.o
ld: warning: search path '/opt/homebrew/opt/tcl-tk/lib' not found
[ 33%] Linking CXX executable strata-plan
[ 33%] Built target conv_cache_test
[ 35%] Building CXX object CMakeFiles/strata_spec.dir/src/spec/draft_policy.cpp.o
ld: warning: search path '/opt/homebrew/opt/tcl-tk/lib' not found
[ 35%] Built target strata-plan
[ 38%] Building CXX object CMakeFiles/coupled_draft_test.dir/src/core/coupled_draft_test.cpp.o
[ 41%] Linking CXX executable gguf_split_test
ld: warning: search path '/opt/homebrew/opt/tcl-tk/lib' not found
[ 43%] Building CXX object CMakeFiles/bf16_bits_test.dir/src/kernels/bf16_bits_test.cpp.o
[ 46%] Linking CXX static library libstrata_spec.a
[ 46%] Built target gguf_split_test
[ 48%] Building CXX object CMakeFiles/strata_kernels_cpu.dir/src/kernels/cpu/expert.cpp.o
[ 51%] Linking CXX executable bf16_bits_test
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
make[1]: *** Waiting for unfinished jobs....
[ 51%] Built target strata_spec
ld: warning: search path '/opt/homebrew/opt/tcl-tk/lib' not found
[ 51%] Built target bf16_bits_test
[ 53%] Linking CXX executable coupled_draft_test
ld: warning: search path '/opt/homebrew/opt/tcl-tk/lib' not found
[ 53%] Built target coupled_draft_test
make: *** [all] Error 2
```

### Initial CPU configuration (exit 1; offline configuration above resolves the fetch dependency)

```sh
cmake -S . -B /tmp/strata-hybrid-cpu-build -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON
```

Actual output:

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
-- The C compiler identification is AppleClang 17.0.0.17000604
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
CMake Warning (dev) at /opt/homebrew/share/cmake/Modules/FetchContent.cmake:1966 (message):
  Calling FetchContent_Populate(strata_llamacpp) is deprecated, call
  FetchContent_MakeAvailable(strata_llamacpp) instead.  Policy CMP0169 can be
  set to OLD to allow FetchContent_Populate(strata_llamacpp) to be called
  directly for now, but the ability to call it with declared details will be
  removed completely in a future version.
Call Stack (most recent call first):
  CMakeLists.txt:1001 (FetchContent_Populate)
This warning is for project developers.  Use -Wno-dev to suppress it.

[ 11%] Creating directories for 'strata_llamacpp-populate'
[ 22%] Performing download step (git clone) for 'strata_llamacpp-populate'
Cloning into 'strata_llamacpp-src'...
fatal: unable to access 'https://github.com/ggml-org/llama.cpp.git/': Could not resolve host: github.com
Cloning into 'strata_llamacpp-src'...
fatal: unable to access 'https://github.com/ggml-org/llama.cpp.git/': Could not resolve host: github.com
Cloning into 'strata_llamacpp-src'...
fatal: unable to access 'https://github.com/ggml-org/llama.cpp.git/': Could not resolve host: github.com
Had to git clone more than once: 3 times.
CMake Error at /private/tmp/strata-hybrid-cpu-build/_deps/strata_llamacpp-subbuild/strata_llamacpp-populate-prefix/tmp/strata_llamacpp-populate-gitclone.cmake:50 (message):
  Failed to clone repository: 'https://github.com/ggml-org/llama.cpp.git'


make[2]: *** [strata_llamacpp-populate-prefix/src/strata_llamacpp-populate-stamp/strata_llamacpp-populate-download] Error 1
make[1]: *** [CMakeFiles/strata_llamacpp-populate.dir/all] Error 2
make: *** [all] Error 2

CMake Error at /opt/homebrew/share/cmake/Modules/FetchContent.cmake:1931 (message):
  Build step for strata_llamacpp failed: 2
Call Stack (most recent call first):
  /opt/homebrew/share/cmake/Modules/FetchContent.cmake:1622 (__FetchContent_populateSubbuild)
  /opt/homebrew/share/cmake/Modules/FetchContent.cmake:2158:EVAL:2 (__FetchContent_doPopulation)
  /opt/homebrew/share/cmake/Modules/FetchContent.cmake:2158 (cmake_language)
  /opt/homebrew/share/cmake/Modules/FetchContent.cmake:1991:EVAL:1 (__FetchContent_Populate)
  /opt/homebrew/share/cmake/Modules/FetchContent.cmake:1991 (cmake_language)
  CMakeLists.txt:1001 (FetchContent_Populate)


-- Configuring incomplete, errors occurred!
```

### Source and generator checks (exit 0)

Commands:

```sh
git diff --check
git diff --exit-code -- include/strata/ds41/kernels/k10_exl3_moe.hpp src/ds41/kernels/k10_exl3_moe.cu src/ds41/kernels/k10
clang++ -std=c++17 -Wall -Wextra -Werror -Iinclude -c src/ds41/ds41_generate.cpp -o /tmp/strata-hybrid-generate.o
```

A Python assertion also compared the complete `Engine::Impl::step()` source with `git show HEAD:src/ds41/engine.cu`.
It passed. This is a source comparison, not graph runtime validation.
The check wrapper printed this real output:

```text
git diff --check: PASS
K10 interface and math unchanged: PASS
Engine::Impl::step unchanged: PASS
ds41_generate.cpp object compilation: PASS (AppleClang, C++17, -Wall -Wextra -Werror)
CUDA code compilation and GPU execution: NOT RUN (no CUDA toolkit or NVIDIA GPU)
```
