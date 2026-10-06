# Hybrid decode: VRAM staging follow-up

The staging code and GPU acceptance tests are written. The new CUDA code is not compiled or GPU-tested here.
No staging speedup or new numerical parity result is claimed. The reviewer must run the GPU acceptance below.
This report replaces the earlier handoff for the direct mapped-memory implementation, which the reviewer committed.

## State by part

| Part | Code written | Compiled locally | Tested on GPU | Not done |
| --- | --- | --- | --- | --- |
| Fixed staging arena and uint4 copy kernel | Yes | No | No | CUDA build, exact bytes, sanitizer |
| Publish copy list and rebased K10 descriptors | Yes | No | No | Quota and mixed-size GPU acceptance |
| Engine reservation and graph fork/join | Yes | No | No | Whole-engine replay, measured overlap |
| DS41_ZC_STAGE=0 comparison path | Yes | No | No | Fixed-slot SAGE parity against stage=1 |
| Synthetic parity and copy tests | Yes, tests first | No | No | Run expert_staging_test and expanded hybrid_decode_test |
| Reviewer sweep script | Yes | Shell and embedded Python syntax checked | No | Run all 42 real cases |
| Portable regression targets | Existing | Yes | Not applicable | None: 8/8 passed |
| Generator translation unit | Unchanged | Yes, strict warnings | No | Full engine link requires CUDA |
| Commits | Units listed in HYBRID.COMMITS.md | Not applicable | Not applicable | Metadata is outside writable roots |

I could not run CUDA compilation, because this macOS arm64 host has no CUDA toolkit (`nvcc` was not found).
I could not run `expert_staging_test`, `hybrid_decode_test`, `hybrid_host_table_test`, `doorbell_test`,
`host_experts_test`, `vram_experts_test`, or `vram_lend_test`, because their DS41 engine targets require CUDA
and this host has no NVIDIA GPU.
I could not run the SAGE sweep, compute-sanitizer, or nsys, because the RTX 4090 runtime and `/workspace/pack-sage`
are not available on this host.
I could not complete the full CPU build, because its existing x86 expert code uses AVX options and x86 intrinsics
that Apple arm64 cannot compile.
No commit was attempted in this follow-up: the Git metadata is deliberately read-only in the current sandbox.

## Reviewer baseline, supplied in the task

These are prior direct-path results on RTX 4090, SAGE 1.59bpw, 200 forced tokens. They are not staging results
and were not reproduced in this session. The reviewer reports that the five hybrid/host/doorbell/lend tests passed.
Quota zero matched the old path bit for bit with fixed slots; its NLL was 1.216695.

| CPU threads | q0 ms/token | q2 | q3 | q5 |
| --- | --- | --- | --- | --- |
| 30 | 44.9 | 42.4 | 43.9 | 45.3 |
| 16 | 57.4 | 42.2 | 44.9 | — |
| 8 | 84.0 | 49.2 | 47.1 | 46.9 |

The reviewer measured CPU expert time falling from 44–67 ms to 5–11 ms.
With 16 threads, nsys showed q0 wait_add_k 40.6 ms, dense gemv 6.7 ms, and K10 gemv_mul1_jobs 2.3 ms per step.
At q3, K10 was 23.5 ms for about 58 mapped experts plus VRAM hits; wait_add_k was 4.7 ms,
dense gemv 6.8 ms, and GPU idle 4.1 ms. Direct K10 reached about 60% of PCIe bandwidth.
A uint4 streaming kernel reached 26.3 GB/s with 68 blocks: 262 us for one 6.8 MB expert.

## Implementation and ordering

1. `ExpertStaging` allocates an arena, job list, device count, nonblocking stream, and two events once.
   Engine init reads `ExpertSlot.bytes` and `comp_off[0]` for every expert. It uploads this immutable metadata.
   Each slot has `round_up(max_pack_expert_bytes, 256)` bytes. Decode reserves six slots; the helper supports
   `max_m * topk` slots. Tests exercise up to 48. SAGE therefore needs roughly 6 x 21.1 MiB for decode staging.
   Allocation happens before the automatic VRAM cache picks its capacity, including at q0.
2. Publish keeps the existing route-order quota rule. For each assigned RAM expert, it writes one compact job:
   mapped slot base, exact pack byte count, and the staging slot for the original route position.
   The mapped base comes from the live RAM descriptor minus `comp_off[0]`. This also works after RAM slot swaps.
   All nine trellis/scale pointers in the per-call descriptor move by the same offset into staging.
   VRAM hit descriptors, CPU membership, GPU selection indices, weights, and counts keep their meaning.
   Publish resets the device job count on every call, including zero-job calls. Old jobs above that count are ignored.
3. `stage_experts_k` always launches 68 blocks x 256 threads. All blocks stream each blob with aligned uint4
   loads and stores. A byte tail handles lengths that are not multiples of 16, with no rounded-up source read.
   Host RAM slots are 4 KiB aligned; staging slots are 256 B aligned. Jobs require 16-byte-aligned addresses.
   The grid and storage do not depend on live routes or quota. Only the device list and count change.
4. After publish, the main stream records `ready`; the copy stream waits on it and copies the blobs.
   The main stream computes the shared expert. The copy stream records `done`; the main stream waits before K10.
   One K10 call then handles both VRAM hits and staged RAM experts in the original route order.
   This is simpler than separate K10 calls and preserves the existing accumulation order. The cost is that VRAM
   hits also wait until this join. Shared work and copying can run concurrently; actual overlap is unmeasured.
   K10 completion precedes the next layer's publish, so no job, descriptor, staging slot, or event is reused early.
5. The fork joins back into the capture origin stream before capture ends. CUDA documents this pattern in
   [Cross-stream dependencies and events](https://docs.nvidia.com/cuda/archive/12.0.1/cuda-c-programming-guide/index.html#cross-stream-dependencies-and-events).
   There are no host reads, synchronizations, allocations, or table uploads in publish, fork, join, or `moe()`.
   `enqueue_step()` and `step()` are unchanged. K10's interface and arithmetic are unchanged.
6. `DS41_ZC_QUOTA` keeps its existing 0..6 range and default 4. `DS41_ZC_STAGE` defaults to 1;
   `DS41_ZC_STAGE=0` retains the prior direct scheduling and mapped-memory descriptors, with no staging allocation.
   Only literal 0 and 1 are accepted. No mapped RAM table means the existing CPU/VRAM path.
   When mapped RAM registration fails, a previously reserved staging arena remains unused for that engine lifetime.

**Comparison constraint:** stage=0 and stage=1 can have different automatic VRAM cache capacities.
Use the same explicit `--vram-slots` and `--adapt-every 0` for parity and throughput comparisons.
The sweep script first obtains the staged capacity, then fixes it for both modes and every quota.
A q0 comparison to the previous engine also needs that same slot count.

## Tests written before production changes

- `expert_staging_test`: exact bytes and destination guards for 0, 1, 15, 16, 17, 255, 4095, 4097,
  6,800,003, 22,151,167, and 22,151,168 bytes. It changes device counts from zero through six, permutes destination
  slots, checks eager execution, and replays a graph containing two successive fork/join copies with reused events.
  Every destination starts poisoned. The maximum size exceeds 21.1 MiB.
- `hybrid_decode_test`: real 5120/2304 shapes with compact mixed K1..K6 blobs, a nonzero first-component offset,
  and signed scales. Covers m=1..8, q=0..6, mixed tiers, all-VRAM, all-RAM, all-file, and inactive routes.
  Direct K10 is compared bit for bit with the VRAM reference. Staged K10 is compared bit for bit with direct K10.
  It checks all descriptor fields, source/size/destination jobs, device count reset, membership, and counts.
  The same captured graph replays with changed routes, weights, residency, RAM entries, and quotas.
  Staging bytes are poisoned before each eager run and replay, so stale copies cannot satisfy the check.
- Existing fallback, host-table lifecycle, CPU handshake, and adaptive swap tests remain in the reviewer commands.
  The new copy test and expanded decode test return 77 without a GPU. A skip is not acceptance.

## Exact reviewer commands

### Build and GPU correctness (repo and build directory requested by reviewer)

```bash
cd /workspace/Strata-DS-hybrid
set -euo pipefail
mkdir -p /workspace/hybrid-stage-results
nvidia-smi
nvcc --version
lscpu
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_NATIVE_EXPERTS=OFF -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j 8 --target ds41_generate expert_staging_test \
  hybrid_decode_test hybrid_host_table_test doorbell_test \
  host_experts_test vram_experts_test vram_lend_test
ctest --test-dir build --output-on-failure --no-tests=error \
  -R '^(expert_staging_test|hybrid_decode_test|hybrid_host_table_test|doorbell_test|host_experts_test|vram_experts_test|vram_lend_test)$' \
  | tee /workspace/hybrid-stage-results/ctest.log
! grep -E 'Skipped|Not Run' /workspace/hybrid-stage-results/ctest.log
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/expert_staging_test
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/hybrid_decode_test
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/hybrid_host_table_test
```

For CUDA 12.8 cross-architecture compilation, reconfigure `build` with
`-DCMAKE_CUDA_ARCHITECTURES='86;89;120'`, then repeat the same build command.
Only sm_89 execution can be accepted on the RTX 4090. Record any compilation failure separately.

### Fixed-residency performance: 42 cases, graph enabled

```bash
cd /workspace/Strata-DS-hybrid
set -euo pipefail
mkdir -p /workspace/hybrid-stage-results
bash tools/ds41/zc_stage_sweep.sh | tee /workspace/hybrid-stage-results/sweep.txt
```

The script uses `/workspace/results/quality/sage159/code_py_0.ids`, as in `/workspace/zc_sweep.sh`.
It takes exactly the first 200 ids and uses `/workspace/pack-sage`, the profile in this repository,
100 GiB RAM budget, max sequence 4096, threads 8/16/30, quotas 0/1/2/3/4/5/6, and stage 0/1.
It writes each run's stdout and stderr separately. It fixes residency, verifies 240 route uses per step,
requires nonzero RAM GPU uses at q>0, and compares per-position predictions, printed NLL, and partition counts.
It reports the generator's ms/token and median step time after ten positions. Printed NLL has six decimals;
this script alone does not prove full-logit bitwise equality.
`FORCE_IDS`, `STAGE_RESULTS`, and `VRAM_SLOTS` can override the input, output directory, and fixed slot count.
Repeat the script with a new `STAGE_RESULTS` directory to assess timing noise.

### Whole-engine bitwise evidence (dump runs are eager)

```bash
cd /workspace/Strata-DS-hybrid
set -euo pipefail
R=/workspace/hybrid-stage-results
SLOTS=$(cat "$R/slots.txt")
for q in 0 1 2 3 4 5 6; do
  for stage in 0 1; do
    DS41_ZC_STAGE=$stage DS41_ZC_QUOTA=$q DS41_GRAPH=0 ./build/ds41_generate \
      --pack /workspace/pack-sage --threads 16 --ram-budget-gib 100 \
      --expert-profile ds41/data/expert-profile.bin --vram-slots "$SLOTS" --adapt-every 0 \
      --max-seq 4096 --force-ids "$R/forced-200.ids" --dump "$R/q${q}-s${stage}.bin" \
      > "$R/dump-q${q}-s${stage}.out" 2> "$R/dump-q${q}-s${stage}.log"
  done
  cmp "$R/q${q}-s0.bin" "$R/q${q}-s1.bin"
done
```

Require exact dumps at each fixed quota. Dumps cover hidden states, routes, weights, and top logits, not every
logit. The synthetic test supplies full K10 output bitwise checks under graph replay. Compare q0 to the prior
reviewed engine at the same fixed slots as a separate regression. Different quotas can change CPU/GPU rounding.
Also compare each eager dump run's per-step predictions and printed NLL with the corresponding graph sweep run.

### Profile the overlap on the same workload

```bash
cd /workspace/Strata-DS-hybrid
set -euo pipefail
R=/workspace/hybrid-stage-results
SLOTS=$(cat "$R/slots.txt")
unset DS41_DEBUG
for stage in 0 1; do
  nsys profile --trace=cuda --cuda-graph-trace=node --force-overwrite=true \
    -o "$R/nsys-t16-q3-s${stage}" \
    env DS41_ZC_STAGE=$stage DS41_ZC_QUOTA=3 DS41_GRAPH=1 ./build/ds41_generate \
      --pack /workspace/pack-sage --threads 16 --ram-budget-gib 100 \
      --expert-profile ds41/data/expert-profile.bin --vram-slots "$SLOTS" --adapt-every 0 \
      --max-seq 4096 --force-ids "$R/forced-200.ids" \
      > "$R/nsys-s${stage}.out" 2> "$R/nsys-s${stage}.log"
  nsys stats --report cuda_gpu_kern_sum "$R/nsys-t16-q3-s${stage}.nsys-rep" \
    > "$R/nsys-s${stage}-kernels.txt"
done
```

Inspect the timeline: publish precedes each copy; shared GEMVs may overlap `stage_experts_k`; K10 follows both;
wait_add follows K10. Record copy time, K10 time, wait time, idle time, and ms/token. A faster copy alone does not
establish a faster decode step. GPU bandwidth contention may limit the overlap.

For live slot updates, repeat stage=0/1 at q3 with `--adapt-every 4` and require committed swaps in each log.
Asynchronous swap completion may choose different residency over time, so use the static runs for exact parity.
For no-RAM fallback, repeat stage=0/1 with `--ram-budget-gib 0`, q6, fixed slots, and compare eager dumps.
These extra runs and the prior-engine q0 comparison remain reviewer acceptance, not completed work here.

## Local checks: real output from this follow-up

The portable checks below do not compile the staging CUDA code or validate GPU execution.

### CPU-only configuration (exit 0)

```sh
cmake -S . -B /tmp/strata-hybrid-cpu-build -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON -DSTRATA_NATIVE_EXPERTS=OFF
```

Actual output:

```text
-- Configuring done (0.1s)
-- Generating done (0.0s)
-- Build files have been written to: /tmp/strata-hybrid-cpu-build
```

### Portable build (exit 0)

```sh
cmake --build /tmp/strata-hybrid-cpu-build --target gguf_reader_test gguf_split_test suffix_drafter_test controller_test draft_policy_test conv_cache_test coupled_draft_test bf16_bits_test -j 4
```

Actual output:

```text
[100%] Built target gguf_reader_test
[100%] Built target gguf_split_test
[ 66%] Built target strata_spec
[100%] Built target suffix_drafter_test
[ 66%] Built target strata_spec
[100%] Built target controller_test
[ 66%] Built target strata_spec
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
1/8 Test #1: gguf_reader_test .................   Passed    0.00 sec
    Start 2: gguf_split_test
2/8 Test #2: gguf_split_test ..................   Passed    0.01 sec
    Start 3: suffix_drafter_test
3/8 Test #3: suffix_drafter_test ..............   Passed    0.01 sec
    Start 4: controller_test
4/8 Test #4: controller_test ..................   Passed    0.00 sec
    Start 5: draft_policy_test
5/8 Test #5: draft_policy_test ................   Passed    0.00 sec
    Start 6: conv_cache_test
6/8 Test #6: conv_cache_test ..................   Passed    0.00 sec
    Start 7: coupled_draft_test
7/8 Test #7: coupled_draft_test ...............   Passed    0.00 sec
    Start 8: bf16_bits_test
8/8 Test #8: bf16_bits_test ...................   Passed    5.05 sec

100% tests passed, 0 tests failed out of 8

Total Test time (real) =   5.09 sec
```

### CUDA configuration (exit 1)

```sh
cmake -S . -B /tmp/strata-stage-cuda-build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_NATIVE_EXPERTS=OFF -DCMAKE_CUDA_ARCHITECTURES=89
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

Actual output excerpts (first 25 and last 5 lines; intervening x86-intrinsic errors omitted):

```text
[ 10%] Built target gguf_split_test
[ 10%] Built target gguf_reader_test
[ 20%] Built target strata-gguf
[ 20%] Built target strata-dequant
[ 33%] Built target strata-plan
[ 35%] Built target strata_spec
[ 43%] Built target conv_cache_test
[ 46%] Built target coupled_draft_test
[ 53%] Building CXX object CMakeFiles/strata_kernels_cpu.dir/src/kernels/cpu/expert.cpp.o
[ 53%] Built target bf16_bits_test
[ 58%] Built target controller_test
[ 64%] Built target suffix_drafter_test
clang++: error: unsupported option '-mavx512f' for target 'arm64-apple-darwin25.5.0'
clang++: error: unsupported option '-mavx512bw' for target 'arm64-apple-darwin25.5.0'
clang++: error: unsupported option '-mavx512vl' for target 'arm64-apple-darwin25.5.0'
clang++: error: unsupported option '-mavx512dq' for target 'arm64-apple-darwin25.5.0'
clang++: error: unsupported option '-mavx512vnni' for target 'arm64-apple-darwin25.5.0'
clang++: error: unsupported option '-mavx512vbmi' for target 'arm64-apple-darwin25.5.0'
clang++: error: unsupported option '-mf16c' for target 'arm64-apple-darwin25.5.0'
clang++: error: unsupported option '-mfma' for target 'arm64-apple-darwin25.5.0'
make[2]: *** [CMakeFiles/strata_kernels_cpu.dir/src/kernels/cpu/expert.cpp.o] Error 1
make[2]: *** Waiting for unfinished jobs....
[ 69%] Building CXX object CMakeFiles/strata_kernels_cpu.dir/src/kernels/cpu/pool.cpp.o
[ 69%] Building CXX object CMakeFiles/strata_kernels_cpu.dir/src/kernels/cpu/expert_layout.cpp.o
[ 74%] Built target draft_policy_test
[...]
fatal error: too many errors emitted, stopping now [-ferror-limit=]
20 errors generated.
make[2]: *** [CMakeFiles/strata_kernels_cpu.dir/src/kernels/cpu/pool.cpp.o] Error 1
make[1]: *** [CMakeFiles/strata_kernels_cpu.dir/all] Error 2
make: *** [all] Error 2
```

### Source and syntax checks (exit 0)

```sh
git diff --check
git diff --exit-code -- include/strata/ds41/kernels/k10_exl3_moe.hpp src/ds41/kernels/k10_exl3_moe.cu src/ds41/kernels/k10
clang++ -std=c++17 -Wall -Wextra -Werror -Iinclude -c src/ds41/ds41_generate.cpp -o /tmp/strata-stage-generate.o
bash -n tools/ds41/zc_stage_sweep.sh
```

A Python source assertion compared `engine.cu` from `enqueue_step()` through end-of-file with HEAD.
Both Python heredocs in the sweep script were passed to Python `compile()` for syntax checking.
The check wrappers printed this actual output:

```text
ds41_generate.cpp object compilation: PASS (AppleClang, C++17, -Wall -Wextra -Werror)
zc_stage_sweep.sh shell syntax: PASS
enqueue_step / step source unchanged: PASS
K10 interface and arithmetic unchanged: PASS
zc_stage_sweep.sh embedded Python syntax: PASS
```

`git diff --check` and the K10 diff emitted no output and returned zero.
The shell script was not run against a synthetic or mock generator. Its real end-to-end run requires the GPU box.
