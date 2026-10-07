# DS41 tier review fixes

Base: `e2d84995f450a21fefcc214e25e0075128b908f6` (PR #10).
Branch: `fix/ds41-revfix-tiers`.
Local host: macOS arm64, Apple Clang 17.0.0, Python 3.14.7.

Findings 1–7 are real. Each was checked against the production source.
Finding 8 belongs to another worker and was skipped.
`src/ds41/engine.cu`, `src/ds41/ds41_generate.cpp`, `src/ds41/wo_a_fp8.cu`, and its existing test are unchanged.
The reviewer reports remain unchanged and untracked.

## State

| Part | Code written | Compiled locally | Tested on RTX 4090 | Not done |
|---|---|---|---|---|
| 1. Every expert range | Yes | Yes, host CUDA substitute | No | Native CUDA build and target tests |
| 2. Range and shape overflow | Yes | Yes, host CUDA substitute | No | Native CUDA build and target tests |
| 3. Engram hash validation | Yes | Yes, host CUDA substitute | No | Native CUDA build and target tests |
| 4. Residency synchronization | Yes | Yes, host CUDA substitute | No | Working TSan run and native target tests |
| 5. Failed Engram reads | Yes | Yes | No | Linux O_DIRECT and fallback tests |
| 6. Constructor cleanup | Yes | Yes, host CUDA substitute | No | Real CUDA resource and DMA validation |
| 7. Paths with spaces | Yes | Yes; Python syntax checked | No | Target integration tests |
| 8. E8M0 code zero | Skipped by request | Not part of this work | Not part of this work | Other worker owns this finding |

Local regression result: 21 passed, 0 failed. The same tests against the base have 1 pass and 20 failures.
The base crashes on the negative hash index. The valid-input control passes on both versions.
UBSan also passes all 21 tests. A separate concurrent residency stress test passes under UBSan.
Existing tests and their tolerances were not changed.

These results use synthetic files. They are not real model inference results.
The CUDA substitute counts allocations and injects failures. It does not execute GPU copies or kernels.
On macOS, the test runner defines O_DIRECT as zero and makes posix_fadvise a no-op.
The queue, worker threads, pread calls, completions, and row copies still execute production code.
The existing Engram test prints `O_DIRECT` in this mode. That label does not prove direct I/O on macOS.

I could not run the native CUDA build, because this host has no nvcc. CMake failed while finding the toolkit.
I could not run RTX 4090 tests or Compute Sanitizer, because no target connection was supplied and this host has no NVIDIA GPU.
I could not complete AddressSanitizer, because the first test timed out. An independent empty ASan program also timed out.
I could not complete ThreadSanitizer, because its test processes exited with signal 11. An independent empty TSan program also exited with signal 11.
These sanitizer failures are not passing results. The exact runtime cause is unknown.

## Findings and tests

### 1. Only the last expert was checked

Confirmed: `map_experts()` checked only `experts_.back()` after publishing the mapping.
It now checks every expert against the file size before mmap and publication.
It checks fstat and rejects empty files. Error paths close the file descriptor.
A successful repeated map replaces the old mapping only after validation.

Regression: `early_slot` puts expert (0, 0) past EOF and leaves the last expert valid.
The test requires rejection and a null published mapping.
Base: accepted the bad slot. Fixed: rejected it.

### 2. Unsigned ranges and shape multiplication overflowed

Confirmed: component, dense, and final-expert checks used unchecked addition.
All three checks now use `offset <= limit && bytes <= limit - offset`.
Dense ranges are checked before allocation or upload.
Shape products use checked unsigned multiplication. Invalid ranks and dimensions are rejected.
Slot sizes must leave room for the tiers' 4 KiB alignment.
Component numbers must be complete unsigned tokens.

Regressions: `component_overflow`, `dense_overflow`, `slot_overflow`, and `shape_overflow`.
The three byte-range tests use values near UINT64_MAX.
The shape test uses dimensions whose product wraps to zero in the old loader.
All four were accepted by the base. All four are rejected now.

### 3. Engram hash indices and arrays were unchecked

Confirmed: a negative table index indexed a vector before its beginning.
Missing arrays and zero primes reached unchecked engine operations.
The parser now sizes arrays from the Engram table count and checks each index before use.
It requires complete multiplier, prime, and offset arrays.
It checks layer order against the table index, bounds the hash dimensions, and rejects nonpositive primes.
Each prime bucket and offset must fit its table's rows.

Regressions: `negative_index`, `missing_index`, `zero_prime`, `bad_index`, `large_index`,
`short_array`, and `layer_mismatch`.
The base crashes on `negative_index` and accepts the other bad inputs.
The fixed loader rejects all seven. The valid fixture still loads.

### 4. Lookahead reads raced with residency writes

Confirmed: `warm_file_expert()` reads both host tables from the lookahead thread.
Tier writes had no C++ synchronization with this callback.
A shared host mutex now spans each lookahead callback and each residency element write.
This includes initialization, swap eviction, swap commit, lending, restoration, RAM eviction, and RAM assignment.
The lock is held only for element publication on the writer side.
The callback can hold it through madvise. A tier write can wait for that callback.
The callback must not call a residency mutation while it holds this lock.
The GPU tables remain arrays of int32_t. Their layout and GPU publication rules are unchanged.
No engine edit was needed.

Regressions: `vram_race` and `host_race` hold a real lookahead callback open.
They check that lend/restore and point_to_file/assign cannot publish during the callback.
Both fail against the base and pass after the fix.
`residency_stress` adds repeated real table reads during 1,000 lending and RAM reassignment cycles.
It passes locally with UBSan. TSan remains required on Linux.

### 5. Engram read failures left stale completions

Confirmed: after one table failed, later tables could leave completions with tags from the old request list.
The reader now enters a failed state before submission starts.
Only a fully successful read clears that state.
A later read rejects the failed object before it can reuse requests or buffers.
Destruction joins the DirectFile workers before it frees the read buffers.
Outstanding workers write only to those owned buffers.

Regression: `engram_retry` submits two rows for each of two tables, causes the first table to fail,
then retries with one row per table. It requires the explicit failed-state error.
The base accepts reuse or consumes stale completions. The fixed reader rejects reuse.
This is the requested poison-object option. Callers must create a new reader after an I/O failure.

### 6. Failed constructors leaked resources

Confirmed: neither class ran its destructor when its constructor body threw.
Both constructors now catch exceptions and call the same cleanup function as their destructor.
VramExperts joins its copier, synchronizes its copy stream, and releases its CUDA allocations.
ExpertStream signals stop, joins each started thread, synchronizes DMA, and releases all created resources.
Null event handles are skipped. The file descriptor is closed.

Regressions: `vram_cleanup`, `stream_cleanup`, and `thread_cleanup`.
They cover a missing profile, all five VRAM constructor allocations, all eleven ExpertStream CUDA allocations,
and a C++ allocation sweep through thread/vector construction.
The C++ sweep rejects allocation points 0–16 and completes at point 17 on this host.
It covers partially started thread sets. A missing join would terminate the process.
The base leaks CUDA resources or a file descriptor. The fixed code leaves the counters and descriptor count unchanged.
OS-level pthread_create failure and real CUDA allocation failure have not been injected on the target.

### 7. Engram paths were truncated at spaces

Confirmed: the writer emitted the full absolute path, but the loader extracted one whitespace-delimited token.
The loader now uses getline for the rest of the line after the five numeric fields.
The writer rejects LF and CR in any table path before writing the index.

Regressions: `path_spaces` and `writer_paths`.
The C++ test parses a path with spaces, opens that path through EngramRows, and checks returned weight and scale bytes.
The Python test executes the production writer function with synthetic safetensors headers.
It accepts spaces and rejects LF and CR. It needs no tensor packages.
The base truncates the path and accepts a newline. The fixed paths pass.
The existing full Python pack suite also passed all nine tests with real torch and safetensors imports.

## Linux RTX 4090 commands

Run these from the target checkout. Change only `TREE` if the checkout suffix differs.
`CUDA_ARCH` is passed to CMake's actual CUDA architecture setting.

```sh
TREE=revfix-tiers
cd "/workspace/Strata-DS-${TREE}"
export CUDA_ARCH=89
cmake -S . -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
cmake --build build --parallel --target ds41_generate lookahead_test host_experts_test engram_rows_test vram_experts_test vram_lend_test expert_stream_test hybrid_host_table_test
```

First run all new regressions on native Linux file I/O. These commands still use CUDA fault injection.
The baseline command is expected to exit nonzero. Do not treat that as a fixed-tree failure.
It loads old production source into a temporary directory and leaves the checkout unchanged.

```sh
python3 tools/ds41/test_revfix.py --baseline-ref e2d84995f450a21fefcc214e25e0075128b908f6
python3 tools/ds41/test_revfix.py
UBSAN_OPTIONS=halt_on_error=1 python3 tools/ds41/test_revfix.py --sanitize=undefined
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 python3 tools/ds41/test_revfix.py --sanitize=address
TSAN_OPTIONS=halt_on_error=1 python3 tools/ds41/test_revfix.py --sanitize=thread --cases vram_race host_race residency_stress
```

Per-finding commands, after the common build above:

```sh
# Finding 1.
python3 tools/ds41/test_revfix.py --cases early_slot
# Finding 2.
python3 tools/ds41/test_revfix.py --cases component_overflow dense_overflow slot_overflow shape_overflow
# Finding 3.
python3 tools/ds41/test_revfix.py --cases negative_index missing_index zero_prime bad_index large_index short_array layer_mismatch
# Finding 4. Native CUDA behavior plus host race detection.
TSAN_OPTIONS=halt_on_error=1 python3 tools/ds41/test_revfix.py --sanitize=thread --cases vram_race host_race residency_stress
ctest --test-dir build --output-on-failure --no-tests=error -R '^(lookahead_test|host_experts_test|vram_experts_test|vram_lend_test|hybrid_host_table_test)$'
# Finding 5. Use a disk-backed checkout for O_DIRECT coverage.
python3 tools/ds41/test_revfix.py --cases engram_retry
ctest --test-dir build --output-on-failure --no-tests=error -R '^engram_rows_test$'
# Finding 6. Inject host failures, then check real CUDA lifetime and DMA paths.
python3 tools/ds41/test_revfix.py --cases vram_cleanup stream_cleanup thread_cleanup
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 ./build/expert_stream_test
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 ./build/vram_lend_test
# Finding 7. Use the project's Python environment with torch, numpy, and safetensors.
python3 tools/ds41/test_revfix.py --cases path_spaces
python3 tools/ds41/test_pack.py
```

Remaining target acceptance: the build must succeed; the listed native tests must run without skips;
ASan and TSan must complete without errors; Compute Sanitizer must report no errors.
Check the Engram test's output to identify actual O_DIRECT and fallback coverage.
A file system can accept O_DIRECT even under /dev/shm. Do not infer fallback coverage from that path alone.
No target output is claimed in this report.

## Local evidence

The first regression run was written and executed before production edits: 1 pass, 19 failures.
The final suite adds the constructor allocation sweep. Its base and fixed runs are pasted below.
Compiler command lines and temporary paths are the actual output.

### Final suite against the base

Command: `python3 tools/ds41/test_revfix.py --baseline-ref e2d8499`

```text
BASELINE e2d8499
BUILD c++ -std=c++17 -O1 -g -pthread -include /var/folders/6l/l0s8rfj164z65gfxfmvkbb4m0000gn/T/ds41-revfix-r3pf_oe0/compat.h -Isrc/ds41/tests/revfix_stubs -I/var/folders/6l/l0s8rfj164z65gfxfmvkbb4m0000gn/T/ds41-revfix-r3pf_oe0/baseline/include -Iinclude -Ithird_party/exllamav3_moe -x c++ src/ds41/tests/revfix_test.cpp /var/folders/6l/l0s8rfj164z65gfxfmvkbb4m0000gn/T/ds41-revfix-r3pf_oe0/baseline/src/ds41/pack.cpp /var/folders/6l/l0s8rfj164z65gfxfmvkbb4m0000gn/T/ds41-revfix-r3pf_oe0/baseline/src/ds41/vram_experts.cu /var/folders/6l/l0s8rfj164z65gfxfmvkbb4m0000gn/T/ds41-revfix-r3pf_oe0/baseline/src/ds41/host_experts.cpp /var/folders/6l/l0s8rfj164z65gfxfmvkbb4m0000gn/T/ds41-revfix-r3pf_oe0/baseline/src/ds41/expert_stream.cpp /var/folders/6l/l0s8rfj164z65gfxfmvkbb4m0000gn/T/ds41-revfix-r3pf_oe0/baseline/src/ds41/lookahead.cpp /var/folders/6l/l0s8rfj164z65gfxfmvkbb4m0000gn/T/ds41-revfix-r3pf_oe0/baseline/src/ds41/engram_rows.cpp /var/folders/6l/l0s8rfj164z65gfxfmvkbb4m0000gn/T/ds41-revfix-r3pf_oe0/baseline/src/platform/direct_file.cpp src/ds41/tests/revfix_stubs/allocation_failure.cpp -o /var/folders/6l/l0s8rfj164z65gfxfmvkbb4m0000gn/T/ds41-revfix-r3pf_oe0/test
MODE: CUDA fault injection; O_DIRECT emulated with buffered pread on macOS
PASS valid
FAIL early_slot: invalid input was accepted
EXIT early_slot: 1
FAIL component_overflow: invalid input was accepted
EXIT component_overflow: 1
FAIL dense_overflow: invalid input was accepted
EXIT dense_overflow: 1
FAIL slot_overflow: invalid input was accepted
EXIT slot_overflow: 1
FAIL shape_overflow: invalid input was accepted
EXIT shape_overflow: 1
EXIT negative_index: -11
FAIL missing_index: invalid input was accepted
EXIT missing_index: 1
FAIL zero_prime: invalid input was accepted
EXIT zero_prime: 1
FAIL bad_index: invalid input was accepted
EXIT bad_index: 1
FAIL large_index: invalid input was accepted
EXIT large_index: 1
FAIL short_array: invalid input was accepted
EXIT short_array: 1
FAIL layer_mismatch: invalid input was accepted
EXIT layer_mismatch: 1
FAIL path_spaces: path truncated
EXIT path_spaces: 1
FAIL engram_retry: failed reader accepted reuse or consumed stale completions
EXIT engram_retry: 1
FAIL vram_cleanup: CUDA resources leaked after missing profile
EXIT vram_cleanup: 1
FAIL stream_cleanup: file descriptor leaked
EXIT stream_cleanup: 1
FAIL vram_race: residency writer ran during the lookahead callback
EXIT vram_race: 1
FAIL host_race: residency writer ran during the lookahead callback
EXIT host_race: 1
allocation point 0 rejected
allocation point 1 rejected
FAIL thread_cleanup: allocation failure leaked CUDA resources
EXIT thread_cleanup: 1
FAIL writer_paths: writer accepted a newline in the path
RESULT 1 passed; 20 failed
```

### Fixed suite

Command: `python3 tools/ds41/test_revfix.py`

```text
BUILD c++ -std=c++17 -O1 -g -pthread -include /var/folders/6l/l0s8rfj164z65gfxfmvkbb4m0000gn/T/ds41-revfix-_3o6mshn/compat.h -Isrc/ds41/tests/revfix_stubs -Iinclude -Ithird_party/exllamav3_moe -x c++ src/ds41/tests/revfix_test.cpp src/ds41/pack.cpp src/ds41/vram_experts.cu src/ds41/host_experts.cpp src/ds41/expert_stream.cpp src/ds41/lookahead.cpp src/ds41/engram_rows.cpp src/platform/direct_file.cpp src/ds41/tests/revfix_stubs/allocation_failure.cpp -o /var/folders/6l/l0s8rfj164z65gfxfmvkbb4m0000gn/T/ds41-revfix-_3o6mshn/test
MODE: CUDA fault injection; O_DIRECT emulated with buffered pread on macOS
PASS valid
PASS early_slot
PASS component_overflow
PASS dense_overflow
PASS slot_overflow
PASS shape_overflow
PASS negative_index
PASS missing_index
PASS zero_prime
PASS bad_index
PASS large_index
PASS short_array
PASS layer_mismatch
PASS path_spaces
PASS engram_retry
PASS vram_cleanup
PASS stream_cleanup
PASS vram_race
PASS host_race
allocation point 0 rejected
allocation point 1 rejected
allocation point 2 rejected
allocation point 3 rejected
allocation point 4 rejected
allocation point 5 rejected
allocation point 6 rejected
allocation point 7 rejected
allocation point 8 rejected
allocation point 9 rejected
allocation point 10 rejected
allocation point 11 rejected
allocation point 12 rejected
allocation point 13 rejected
allocation point 14 rejected
allocation point 15 rejected
allocation point 16 rejected
allocation point 17 completed
PASS thread_cleanup
PASS writer_paths
RESULT 21 passed; 0 failed
```

### Fixed suite with UBSan

Command: `python3 tools/ds41/test_revfix.py --sanitize=undefined`

```text
BUILD c++ -std=c++17 -O1 -g -pthread -include /var/folders/6l/l0s8rfj164z65gfxfmvkbb4m0000gn/T/ds41-revfix-8n4bv47_/compat.h -Isrc/ds41/tests/revfix_stubs -Iinclude -Ithird_party/exllamav3_moe -fsanitize=undefined -fno-omit-frame-pointer -x c++ src/ds41/tests/revfix_test.cpp src/ds41/pack.cpp src/ds41/vram_experts.cu src/ds41/host_experts.cpp src/ds41/expert_stream.cpp src/ds41/lookahead.cpp src/ds41/engram_rows.cpp src/platform/direct_file.cpp src/ds41/tests/revfix_stubs/allocation_failure.cpp -o /var/folders/6l/l0s8rfj164z65gfxfmvkbb4m0000gn/T/ds41-revfix-8n4bv47_/test
MODE: CUDA fault injection; O_DIRECT emulated with buffered pread on macOS
PASS valid
PASS early_slot
PASS component_overflow
PASS dense_overflow
PASS slot_overflow
PASS shape_overflow
PASS negative_index
PASS missing_index
PASS zero_prime
PASS bad_index
PASS large_index
PASS short_array
PASS layer_mismatch
PASS path_spaces
PASS engram_retry
PASS vram_cleanup
PASS stream_cleanup
PASS vram_race
PASS host_race
allocation point 0 rejected
allocation point 1 rejected
allocation point 2 rejected
allocation point 3 rejected
allocation point 4 rejected
allocation point 5 rejected
allocation point 6 rejected
allocation point 7 rejected
allocation point 8 rejected
allocation point 9 rejected
allocation point 10 rejected
allocation point 11 rejected
allocation point 12 rejected
allocation point 13 rejected
allocation point 14 rejected
allocation point 15 rejected
allocation point 16 rejected
allocation point 17 completed
PASS thread_cleanup
PASS writer_paths
RESULT 21 passed; 0 failed
```

### Concurrent stress with UBSan

Command: `python3 tools/ds41/test_revfix.py --sanitize=undefined --cases residency_stress`

```text
BUILD c++ -std=c++17 -O1 -g -pthread -include /var/folders/6l/l0s8rfj164z65gfxfmvkbb4m0000gn/T/ds41-revfix-1xxfwrnr/compat.h -Isrc/ds41/tests/revfix_stubs -Iinclude -Ithird_party/exllamav3_moe -fsanitize=undefined -fno-omit-frame-pointer -x c++ src/ds41/tests/revfix_test.cpp src/ds41/pack.cpp src/ds41/vram_experts.cu src/ds41/host_experts.cpp src/ds41/expert_stream.cpp src/ds41/lookahead.cpp src/ds41/engram_rows.cpp src/platform/direct_file.cpp src/ds41/tests/revfix_stubs/allocation_failure.cpp -o /var/folders/6l/l0s8rfj164z65gfxfmvkbb4m0000gn/T/ds41-revfix-1xxfwrnr/test
MODE: CUDA fault injection; O_DIRECT emulated with buffered pread on macOS
PASS residency_stress
PASS writer_paths
RESULT 2 passed; 0 failed
```

### Existing host tests

Commands used:

```sh
c++ -std=c++17 -O2 -pthread -Iinclude src/ds41/tests/lookahead_test.cpp src/ds41/lookahead.cpp -o /tmp/revfix-checks/lookahead_test
/tmp/revfix-checks/lookahead_test
c++ -std=c++17 -O2 -pthread -include /tmp/revfix-checks/compat.h -Iinclude src/ds41/tests/engram_rows_test.cpp src/ds41/engram_rows.cpp src/platform/direct_file.cpp -o /tmp/revfix-checks/engram_rows_test
(cd /tmp/revfix-checks && ./engram_rows_test)
c++ -std=c++17 -O2 -pthread -include /tmp/revfix-checks/compat.h -Isrc/ds41/tests/revfix_stubs -Iinclude -Ithird_party/exllamav3_moe -x c++ src/ds41/tests/host_experts_test.cpp src/ds41/pack.cpp src/ds41/host_experts.cpp src/ds41/vram_experts.cu /tmp/revfix-checks/cpu_link_stub.cpp -o /tmp/revfix-checks/host_experts_test
(cd /tmp/revfix-checks && ./host_experts_test)
```

The compatibility header is the same macOS adapter generated by `test_revfix.py`.
The CPU link substitute defines an empty `exl3_moe_cpu_set_expert_raw` function.
This existing test uses empty CPU handles. It does not test CPU kernel computation or real CUDA descriptors.

Actual output, in the same order:

```text
selection: 50/50 exact, 0 near ties
RESULT pass
working directory, 0 threads: O_DIRECT
working directory, 64 threads: O_DIRECT
/dev/shm not available: fallback path not tested
RESULT pass
auto budget 0.0 GiB (MemAvailable 0.0 GiB)
RESULT pass
```

### Existing Python pack tests

Command:

```sh
/tmp/claude-501/-Users-jackwu-Projects-Strata-DS/5590375a-bfc7-418f-86d0-9bbf31c371cd/scratchpad/venv/bin/python tools/ds41/test_pack.py
```

Actual final output excerpt. The full run also emitted existing ResourceWarning messages for unclosed files.

```text
.
----------------------------------------------------------------------
Ran 9 tests in 1.849s

OK
```

### CUDA configure attempt

Command: `cmake -S . -B /tmp/revfix-cuda-build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=89`

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
  CMakeLists.txt:135 (enable_language)


-- Configuring incomplete, errors occurred!
```

### TSan attempt

Command: `python3 tools/ds41/test_revfix.py --sanitize=thread --cases vram_race host_race`

```text
BUILD c++ -std=c++17 -O1 -g -pthread -include /var/folders/6l/l0s8rfj164z65gfxfmvkbb4m0000gn/T/ds41-revfix-db_hhgc7/compat.h -Isrc/ds41/tests/revfix_stubs -Iinclude -Ithird_party/exllamav3_moe -fsanitize=thread -fno-omit-frame-pointer -x c++ src/ds41/tests/revfix_test.cpp src/ds41/pack.cpp src/ds41/vram_experts.cu src/ds41/host_experts.cpp src/ds41/expert_stream.cpp src/ds41/lookahead.cpp src/ds41/engram_rows.cpp src/platform/direct_file.cpp src/ds41/tests/revfix_stubs/allocation_failure.cpp -o /var/folders/6l/l0s8rfj164z65gfxfmvkbb4m0000gn/T/ds41-revfix-db_hhgc7/test
MODE: CUDA fault injection; O_DIRECT emulated with buffered pread on macOS
EXIT vram_race: -11
EXIT host_race: -11
PASS writer_paths
RESULT 1 passed; 2 failed
```

### ASan attempt

Command: `python3 tools/ds41/test_revfix.py --sanitize=address`

Actual final exception:

```text
subprocess.TimeoutExpired: Command '['/var/folders/6l/l0s8rfj164z65gfxfmvkbb4m0000gn/T/ds41-revfix-3uedcs16/test', 'valid', '/var/folders/6l/l0s8rfj164z65gfxfmvkbb4m0000gn/T/ds41-revfix-3uedcs16/valid']' timed out after 30 seconds
```

Independent smoke tests compiled a program with only `std::puts("sanitizer smoke pass")`.
A Python subprocess wrapper applied a five-second timeout. Actual output:

```text
tsan exit -11 '' ''
asan smoke timed out after 5 seconds
```

### Source and commit checks

`python3 -m py_compile tools/ds41/pack.py tools/ds41/test_revfix.py` exited zero with no output.
`git diff --check e2d8499` passed.
A byte comparison against the base and a commit size/trailer check produced:

```text
Protected files:
UNCHANGED src/ds41/engine.cu
UNCHANGED src/ds41/ds41_generate.cpp
UNCHANGED src/ds41/wo_a_fp8.cu
UNCHANGED src/ds41/tests/wo_a_fp8_test.cu
Commits:
e2224a5 5 files; trailer OK
b88d1d1 2 files; trailer OK
a902916 2 files; trailer OK
e3e1e94 5 files; trailer OK
4ba3f9f 2 files; trailer OK
PASS diff whitespace check
```

Git permitted the five code/test commits above. It then refused the report commit during git add.
The report and `REVFIX.COMMITS.md` remain uncommitted. The plan contains one two-file documentation commit.
I could not commit these final documents, because the sandbox denied creation of the worktree index lock.

Actual output:

```text
fatal: Unable to create '/Users/jackwu/Projects/Strata-DS/.git/worktrees/Strata-DS-revfix-tiers/index.lock': Operation not permitted
```
