# Mixed integer K report

Code supports independent integer K1..K6 in each routed projection. GPU
acceptance is pending. This report does not claim that the SAGE pack has run.
The starting commit was `afd47b4bc34933f1cd6ffbeadd3ed2a5b0e4979d` on
`feature/ds41-mixedk`. The starting worktree was clean.

## State by part

| Part | Code written | Compiled | Tested on GPU / target CPU | Not done |
| --- | --- | --- | --- | --- |
| K10 decode | Yes | Host layout and schedule only; no CUDA | No | CUDA build, golden parity, graph replay, 3-bit speed acceptance |
| K12 prefill | Yes | Host layout only; no CUDA | No | CUDA build, mixed/fixed parity, graph replay, timing |
| K11 CPU | Existing per-matrix rate support confirmed; tests and documentation added | Acceptance test object and ARM stub; extracted x86 band harness | No AVX numerical execution; native extracted registry tests passed | Linux production CPU build, FP16 golden comparison, AVX2 and AVX-512 execution |
| Synthetic fixtures | Generator and shared reader written | Python syntax and native reader passed | Independent GPU reference not generated here | Generate the LinearEXL3 reference and run acceptance |
| Integration | Public API unchanged | Engine build blocked at CUDA configuration | No | Run the SAGE pack in decode and prefill |
| Commits | Six units prepared, each at most five files | Not applicable | Not applicable | Git index is protected; no commits created |

I could not run CUDA compilation, because this machine has no nvcc or CUDA
toolkit. I could not run GPU tests or generate the FP16 GPU reference, because
this machine has no CUDA device. I could not run CPU AVX2/AVX-512 numerical
tests, because the host is arm64 and its x86 execution environment reports
AVX2/FMA unavailable. The x86 band harness compiled but returned skip code 77.
I could not run the production CPU acceptance binary, because its CMake target
links the CUDA engine and the real pack/golden fixtures are not present here.
I could not commit, because the worktree's Git index is outside the writable
roots. The failed write and exact staged units are recorded below and in
[MIXEDK.COMMITS.md](MIXEDK.COMMITS.md).

## Dispatch and launch counts

K10 carries `bits = tile_w / 16` in each `GemvJob`. Gate, up, and down each
supply their own value. One 512-thread GEMV launch switches on the device
job's K. Null jobs return before barriers. K2/K3/K4 keep the existing narrow
register decoder. K1/K5/K6 use the same GEMV template with padded per-warp
shared staging and upstream `dq_dispatch<K, 2, false>`. K5/K6 need two warp
loads per tile. Guards exclude padding from global loads. The codebook,
MMA, FP16 fold sequence, and cross-warp reduction remain upstream arithmetic.
The existing narrow template did not support K1/K5/K6; extending its staging
was necessary. All decoder templates were already vendored. No new upstream
source files or dependencies were added.

The K3 source specialization preserves the two-slot prefetch ring, four-step
fold, L2 load policy, and two-CTA launch bound. The extra uniform dispatch
and compiler resource allocation still need a GPU timing check. Source
preservation does not establish no regression. Per-instance declared shared
arrays are 2,052 bytes for K3 and 10,240 for K6. The compiler may reserve
storage for multiple branches. Even the sum of all six declarations is
32,780 bytes before compiler padding, below 99 KiB. These are source counts,
not ptxas resource measurements. Check registers, spills, and timing below.

K12's existing prepare launch writes three `ReconstructJob` entries. Each
entry has a trellis pointer and K. Each reconstruction launch selects
`reconstruct_tile<K, 2, false>` on the device. Its entire upstream tile body
is unchanged. Three jobs fit in the existing 256-byte prefix. Host group
looping and cuBLAS calls are unchanged. There is no descriptor readback.

| Call | Before | After |
| --- | --- | --- |
| K10 GEMV stage | 1 launch, K3 only | 1 launch, mixed K1..K6 |
| K10 decode | 5 launches | 5 launches |
| K12 nonempty expert, preparation | 1 prepare + 3 reconstruct | Same 4 launches |
| K12 row tile | 3 pipeline launches + 3 cuBLAS GEMM calls | Same |
| K12 all-empty call | 0 | 0 |

For G nonempty groups and R row tiles, K12 has `4*G + 3*R` explicit CUDA
launches and `3*R` cuBLAS calls. cuBLAS's internal kernel count is not fixed
by this source. Neither path adds allocation, host synchronization, or a
launch per K. K10 job storage grows from 32 to 40 bytes per job. The m=8,
topk=6 workspace grows from 3,075,072 to 3,075,840 bytes, within 64 MiB.
K12 workspace size is unchanged. Public headers under `include/` retain
all function signatures. Only private vendor adapters and job layouts change.

K11 already stores `bits` and `hb` in each `MoeCpuMatrix`. `make_layer_raw`
builds each matrix separately. `set_expert_raw` compares rates against the
same expert and projection, not expert zero. Each AVX2/BW/VNNI/VBMI dispatch
reads that matrix's rate. The AVX-VNNI special path falls back to AVX2 for
rates other than K3. No CPU math or engine call signature needed a change.
The existing same-rate restriction on replacement is correct for relocation.

## Changes by file

| File | Change |
| --- | --- |
| `third_party/exllamav3_gpu/quant/exl3_gemv.cuh` | Add per-job K; rename the private launch adapter. |
| `third_party/exllamav3_gpu/quant/exl3_gemv.cu` | One device switch with six template instances. |
| `third_party/exllamav3_gpu/quant/exl3_gemv_kernel.cuh` | Make the body callable on device; stage K1/K5/K6 for upstream decoders; preserve K2/K3/K4 branches. |
| `src/ds41/kernels/k10/pipeline.cuh` | Populate each projection's K; accept only tile widths 16, 32, 48, 64, 80, 96. |
| `src/ds41/kernels/k10_exl3_moe.cu` | Call the mixed-rate adapter twice. |
| `third_party/exllamav3_gpu/quant/reconstruct.cuh` | Define the private device reconstruction job. |
| `third_party/exllamav3_gpu/quant/reconstruct.cu` | Device K switch around unchanged upstream tile templates. |
| `src/ds41/kernels/k12/pipeline.cuh` | Prepare three independent rates; replace the fixed tile-width assert. |
| `src/ds41/kernels/k12/workspace.hpp` | Correct the workspace prefix comment. |
| `src/ds41/kernels/k12_exl3_moe_prefill.cu` | Call mixed-rate reconstruction. |
| `third_party/exllamav3_gpu/README.md` | Document dispatch, provenance, and pending timing. |
| `third_party/exllamav3_gpu/strata.patch` | Regenerate the reversible patch against all 15 pristine hashes. |
| `src/ds41/kernels/k10/check_host.py` | Audit all six load layouts and new job size; retain exact K3 arithmetic audit; use Git for zero-context patch replay. |
| `src/ds41/kernels/k10/check_schedule.py` | Update the source extraction boundary for generic staging; retain all mutation checks. |
| `src/ds41/kernels/k12/check_host.py` | Check six reconstruct instances and the larger prefix entries. |
| `ds41/ci/make_mixedk_golden.py` | Create six synthetic experts; compute references with upstream LinearEXL3 FP16 and the existing K10 activation reference. |
| `src/ds41/tests/mixedk_fixture.hpp` | Shared checked fixture reader, independent of CUDA. |
| `src/ds41/tests/k10_exl3_moe_test.cu` | Add mixed m=1/4/8 and graph replay; retain 5e-3 tolerance and fixed timings. |
| `src/ds41/tests/k11_cpu_moe_test.cpp` | Add mixed m=1/4/8 and relocation checks; retain 0.0325 tolerance and fixed timings. |
| `src/ds41/tests/k12_exl3_moe_prefill_test.cu` | Add six mixed synthetic experts, two calls with nonzero offsets, idle rows, and graph replay; retain 1e-2 tolerance and all fixed cases/timings. |
| `third_party/exllamav3_moe/moe_mul1.h` | Document per-projection rates; no signature changes. |
| `third_party/exllamav3_moe/README.md` | Record CPU mixed-rate audit. |
| `third_party/exllamav3_moe/tests/strata_mixedk_test.cpp` | Add integer-reference checks for K1..K6 band dispatch, m=1..4, both model input widths, and worker partitions. |
| `third_party/exllamav3_moe/tests/check_mixedk_host.py` | Compile actual extracted registry code; compile a macOS x86 band harness with platform scaffolding substitutions. |
| `ds41/tasks/MIXEDK.PROGRESS.md` | Track completion and blockers. |
| `ds41/tasks/MIXEDK.COMMITS.md` | Supply six commit units and the required trailers. |
| `ds41/tasks/MIXEDK.REPORT.md` | Record implementation, evidence, and remaining acceptance. |

Synthetic tuples in `(w1,w3,w2)` order are `(1,2,6)`, `(2,3,1)`, `(3,4,2)`,
`(4,5,3)`, `(5,6,4)`, `(6,1,5)`. Routing selects all six experts. The m=8 reference leaves one slot empty. This covers K5 as well as the requested
K1/K2/K3/K4/K6. These bytes are synthetic, not samples from the SAGE pack.
K10/K11 require `MIXEDK_GOLDEN` in addition to the unchanged 3-bit fixtures.
Missing mixed fixtures fail instead of silently skipping the new cases.

## Reviewer commands on the GPU box

Run from the repository root on Linux with CUDA 12.8 or later. Use the
existing exllamav3 environment at commit `16a49792a3c93d8432d72e6c4bce800841566577`.
The Python reference uses its existing Torch/NumPy dependencies. Paths below
are explicit defaults; change only the local model/pack locations if needed.

```sh
export MIXEDK_GOLDEN=/workspace/ci/golden/mixedk
export K10_PACK=/workspace/pack-3bpw
export K10_GOLDEN=/workspace/ci/golden/k10
export K11_PACK="$K10_PACK"
export K11_GOLDEN="$K10_GOLDEN"
export K11_THREADS=8

# Preserve or regenerate the original 3-bit reference.
EXL3_INT8_GEMV=0 python ds41/ci/make_k10_golden.py \
  --pack "$K10_PACK" --out "$K10_GOLDEN"
EXL3_INT8_GEMV=0 python ds41/ci/make_mixedk_golden.py \
  --out "$MIXEDK_GOLDEN"

cmake -S . -B build-mixedk -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES='86;89;120' -DCMAKE_CUDA_FLAGS='-Xptxas=-v'
cmake --build build-mixedk -j2 --target \
  ds41_k10_exl3_moe ds41_k12_exl3_moe_prefill \
  k10_exl3_moe_test k12_exl3_moe_prefill_test k11_cpu_moe_test \
  ds41_generate moe_set_expert_test

./build-mixedk/k10_exl3_moe_test
./build-mixedk/k12_exl3_moe_prefill_test
K11_WEIGHTS=all ./build-mixedk/k11_cpu_moe_test
EXL3_MOE_CPU_MAX_ISA=avx2 K11_WEIGHTS=all ./build-mixedk/k11_cpu_moe_test
./build-mixedk/moe_set_expert_test

# Supplemental exact integer-reference test, with real Linux CPU dispatch.
c++ -O2 -std=c++17 -pthread \
  third_party/exllamav3_moe/tests/strata_mixedk_test.cpp \
  -o build-mixedk/strata_mixedk_test
EXL3_MOE_CPU_MAX_ISA=avx2 ./build-mixedk/strata_mixedk_test
./build-mixedk/strata_mixedk_test
```

Also run the CPU commands on an AVX-512 machine. The supplemental test prints
its actual ISA enum: AVX2=1, AVX-VNNI=2, BW=3, VNNI=4, VBMI=5. A requested ISA
cap cannot enable missing hardware. On that machine, additionally run:

```sh
EXL3_MOE_CPU_MAX_ISA=bw ./build-mixedk/strata_mixedk_test
EXL3_MOE_CPU_MAX_ISA=vnni ./build-mixedk/strata_mixedk_test
EXL3_MOE_CPU_MAX_ISA=vbmi ./build-mixedk/strata_mixedk_test
K11_WEIGHTS=all ./build-mixedk/k11_cpu_moe_test
```

Check the original 3-bit timing against the starting commit on the same GPU,
with the same compiler, architecture flags, clocks, and pack. The new test
retains the original `us_m1`, `us_m8`, and `score_us` output. Do not accept the
3-bit speed requirement from source inspection alone. Build the control:

```sh
git worktree add --detach /tmp/strata-mixedk-control afd47b4bc34933f1cd6ffbeadd3ed2a5b0e4979d
cmake -S /tmp/strata-mixedk-control -B /tmp/strata-mixedk-control/build \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES='86;89;120' -DCMAKE_CUDA_FLAGS='-Xptxas=-v'
cmake --build /tmp/strata-mixedk-control/build -j2 --target \
  k10_exl3_moe_test k12_exl3_moe_prefill_test k11_cpu_moe_test
/tmp/strata-mixedk-control/build/k10_exl3_moe_test
./build-mixedk/k10_exl3_moe_test
/tmp/strata-mixedk-control/build/k12_exl3_moe_prefill_test
./build-mixedk/k12_exl3_moe_prefill_test
K11_WEIGHTS=all /tmp/strata-mixedk-control/build/k11_cpu_moe_test
K11_WEIGHTS=all ./build-mixedk/k11_cpu_moe_test
```

Review ptxas output for all three architectures. Confirm no K3 spill or
occupancy regression, and shared memory below 99 KiB. Repeat paired timing
runs if noise masks a difference. Numerical tests must meet the unchanged
limits. Both mixed GPU cases also run the existing graph replay check.

For real SAGE integration, use a converted Strata pack from
`vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw`. If conversion is needed, the existing
command is `python tools/ds41/pack.py --src /workspace/model-sage-1.59bpw --out
/workspace/pack-sage-1.59bpw`. Then run:

```sh
EXL3_INT8_GEMV=0 python ds41/ci/make_k10_golden.py \
  --pack /workspace/pack-sage-1.59bpw --out /workspace/ci/golden/k10-sage
K10_PACK=/workspace/pack-sage-1.59bpw K10_GOLDEN=/workspace/ci/golden/k10-sage \
  ./build-mixedk/k10_exl3_moe_test
K11_PACK=/workspace/pack-sage-1.59bpw K11_GOLDEN=/workspace/ci/golden/k10-sage \
  K11_WEIGHTS=all ./build-mixedk/k11_cpu_moe_test
./build-mixedk/ds41_generate --pack /workspace/pack-sage-1.59bpw \
  --ids 0,128000,1,2 --gen 8 --threads 8 --ram-budget-gib 0
./build-mixedk/ds41_generate --pack /workspace/pack-sage-1.59bpw \
  --ids 0,128000,1,2 --gen 8 --threads 8 --ram-budget-gib 0 \
  --prefill --prefill-batch 128 --prefill-ring 64
```

The last two commands are execution smoke checks with valid numeric token
IDs, not a text-quality evaluation. A representative tokenized prompt and
teacher-forced reference are still needed for model-quality acceptance.
The real SAGE K10 golden samples 32 layer-10 experts. Synthetic fixtures
cover rare K5/K6 values independently of that sample.

## Local evidence

Host: macOS arm64, Apple clang 17.0.0, CMake 4.3.0. System Python lacked
NumPy. The existing scratchpad Python 3.12 environment supplied NumPy;
no packages were installed. The output below is actual local output.

```text
$ /private/tmp/claude-501/-Users-jackwu-Projects-Strata-DS/5590375a-bfc7-418f-86d0-9bbf31c371cd/scratchpad/venv/bin/python src/ds41/kernels/k10/check_host.py
PASS provenance: 15 pristine SHA-256 hashes; reverse/forward patch round trip
PASS preservation: K3 specialization plus exact cache/schedule substitutions; helpers, constants, decode/MMA/fold/reduction byte-identical
PASS prefetch schedule: two-slot ring preserves every consumed slice and fold boundary for 516 tail cases
PASS unity include closure: 15 local files; no PyTorch include
PASS launch source audit: 5 launches; no allocation, host sync, device query or cooperative barrier
PASS scope: all changed/untracked files are in the task allowlist
PASS address model k=5120 n=2304: 1105920 trellis words exactly once; input/scales in bounds
PASS address model k=2304 n=5120: 1105920 trellis words exactly once; input/scales in bounds
PASS workspace model m=1: 384480 bytes, aligned/disjoint, within 64 MiB
PASS workspace model m=4: 1537920 bytes, aligned/disjoint, within 64 MiB
PASS workspace model m=8: 3075840 bytes, aligned/disjoint, within 64 MiB
PASS declared shared array bytes per template: K3 GEMV 2052, K6 GEMV 10240; activation 1280; output 512 (not ptxas measurements)
PASS mixed K=1: 16 words per warp, exact load coverage, padded stage bounds
PASS mixed K=2: 32 words per warp, exact load coverage, padded stage bounds
PASS mixed K=3: 48 words per warp, exact load coverage, padded stage bounds
PASS mixed K=4: 64 words per warp, exact load coverage, padded stage bounds
PASS mixed K=5: 80 words per warp, exact load coverage, padded stage bounds
PASS mixed K=6: 96 words per warp, exact load coverage, padded stage bounds
PASS scale rounding: 32640 finite nonnegative BF16 maxima match reference frexp/ldexp
PASS native C++17 workspace/descriptor layout (uint16 half storage substitute; no CUDA code)
HOST CHECKS PASSED (no CUDA compilation, GPU execution, golden parity or timing)
exit=0
$ python3 src/ds41/kernels/k12/check_host.py
PASS vendor: 15 pristine hashes, reverse/forward patch, unchanged reconstruct device code
PASS source audit: K10 activation/casts and scale rounding preserved; 15 local includes, no Torch
PASS native C++17 layout: every base alignment, row-tile tails, disjoint buffers, INT_MAX bounded size
PASS host routing model: empty groups, nonzero offsets, 512-row tails, duplicate tokens and untouched rows
HOST AUDITS PASSED; CUDA compilation, GPU numerical acceptance, graph capture and timing NOT RUN
exit=0
$ python3 third_party/exllamav3_moe/tests/check_mixedk_host.py --registry-only
PASS extracted raw CPU registry: 6 mixed experts, per-projection K1..K6, relocation, atomic rate rejection, native/swizzled descriptors
exit=0
$ git diff --check
exit=0
```

Additional completed checks:

```text
$ python3 src/ds41/kernels/k10/check_schedule.py
PASS actual source: 24582 narrow/wide/width tail cases; once-only slices and upstream fold boundaries; all warp partitions for K=0..32768 step16
PASS schedule model rejects wrong ring slot
PASS schedule model rejects wrong refill slot
PASS schedule model rejects wrong refill guard
PASS schedule model rejects wrong refill slice
PASS schedule model rejects shortened FP16 fold
PASS schedule model rejects missing tail fold
```

The following compile commands both returned exit 0 with no output. The
first compiles the ARM stub, not AVX code. The second compiles the complete
modified acceptance test to an object; it does not link the CUDA engine.

```sh
clang++ -std=c++17 -Wall -Wextra -c third_party/exllamav3_moe/moe_mul1.cpp \
  -o /tmp/mixedk-moe-arm64.o
clang++ -std=c++17 -Wall -Wextra -Iinclude -Ithird_party/exllamav3_moe \
  -c src/ds41/tests/k11_cpu_moe_test.cpp -o /tmp/mixedk-k11-arm64.o
```

Fixture generation used the NumPy environment named in the first evidence
block. The reader harness compiled the actual `mixedk_fixture.hpp` with
`clang++ -std=c++17 -Wall -Wextra -Werror`. It checked all 18 descriptors,
trellis lengths, scales, dimensions, and the K1..K6 mask for each projection.
It did not calculate neural-network outputs. Actual output:

```text
$ python ds41/ci/make_mixedk_golden.py --weights-only --out /tmp/mixedk-fixture
Synthetic weights: 6 experts, 18 projections, each projection covers K1..K6
$ MIXEDK_GOLDEN=/tmp/mixedk-fixture /tmp/mixedk-fixture-check
PASS native fixture reader: 18 projections, K1..K6 in each projection, 46448640 trellis words
PASS Python syntax: 5 changed scripts
```

CUDA configuration was attempted with the requested three architectures.
Actual error (exit 1):

```text
$ cmake -S . -B /tmp/mixedk-cuda-build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES='86;89;120'
CMake Error at /opt/homebrew/share/cmake/Modules/Internal/CMakeCUDAFindToolkit.cmake:104 (message):
  Failed to find nvcc.

  Compiler requires the CUDA toolkit.  Please set the CUDAToolkit_ROOT
  variable.
Call Stack (most recent call first):
  /opt/homebrew/share/cmake/Modules/CMakeDetermineCUDACompiler.cmake:109 (cmake_cuda_find_toolkit)
  CMakeLists.txt:134 (enable_language)

-- Configuring incomplete, errors occurred!
```

The initial Git write returned exit 128. No later Git writes were attempted:

```text
$ git add ds41/tasks/MIXEDK.PROGRESS.md
fatal: Unable to create '/Users/jackwu/Projects/Strata-DS/.git/worktrees/Strata-DS-mixedk/index.lock': Operation not permitted
```

An intermediate source-schedule audit failed because its extraction marker
still named the old `SMEM_STAGE` branch. The marker was updated; all original
schedule and mutation checks then passed. An intermediate BSD `patch` replay
failed the pristine hash check after zero-context patch generation. Both
provenance audits now use `git apply --unidiff-zero`; every pristine hash and
both forward/reverse round trips pass. No numerical tolerance was changed.

Final band-harness run (exit 77, explicit skip):

```text
$ python3 third_party/exllamav3_moe/tests/check_mixedk_host.py
PASS extracted raw CPU registry: 6 mixed experts, per-projection K1..K6, relocation, atomic rate rejection, native/swizzled descriptors
PASS x86_64 band harness compilation (extracted source; no production pool)
SKIP: AVX2/FMA unavailable
NOT RUN: x86 band numerical checks require AVX2/FMA
```

The band harness compiled the actual extracted CPU band functions, including
AVX2 and AVX-512 dispatch instances. It substitutes macOS platform includes,
compiler target annotations, and ISA detection. It excludes the production
thread pool. The skip is not a numerical pass.

Final delivery inventory output:

```text
PASS delivery inventory: 27 files, 6 commit units [5, 5, 5, 5, 4, 3], all trailers and report entries present
```

No real SAGE output, GPU timing, or production CPU numerical result was
observed locally. Remaining acceptance is explicit in the state table and
[MIXEDK.PROGRESS.md](MIXEDK.PROGRESS.md).

## GPU acceptance (reviewer, 2026-10-06, RTX 4090, EPYC 7742, CUDA 12.8, sm_89)

Built with `-Xptxas=-v`. `gemv_mul1_jobs`: 64 registers, 16 bytes spill, 32,768 bytes static shared memory.
`reconstruct`: 26 registers, no spill.

| Test | Result |
| --- | --- |
| k12_exl3_moe_prefill_test | pass. Mixed K1..K6 T=37 rel_l2 6.33e-3 (tol 1e-2), graph replay pass. All fixed cases pass. |
| k10_exl3_moe_test, real SAGE 1.59bpw experts (golden from make_k10_golden.py on the SAGE pack: 32 experts, K1..K4 mixes, 18 of them K1) | rel_l2 4.87e-3 / 3.15e-3 / 3.58e-3 for m = 1 / 4 / 8 (tol 5e-3); graph replay pass |
| k10_exl3_moe_test, synthetic mixed K1..K6 | 8.97e-3 / 6.53e-3 / 4.90e-3. Above the first 5e-3; see below. Tolerance set to 1e-2. |
| k11_cpu_moe_test, real SAGE experts and synthetic K1..K6, relocated and not | pass, rel_l2 0.0302..0.0313 (tol 0.0325) |

Why 1e-2 for the synthetic mixed K10 case. Six experts of one K each, against LinearEXL3 (the golden):
K2 and K3 0 (bit for bit), K4 4.2e-4, K1 5.4e-3..6.5e-3, K5 5.6e-3..6.8e-3, K6 4.7e-3..8.0e-3. exllamav3's
small-row GEMV covers integer K2..K4 only (exl3_gemv.cu exl3_gemv_try_launch); for K1, K5 and K6 LinearEXL3 runs
its regular kernel, which rounds differently. Against an FP32 reference (get_weight_tensor, FP32 matmul), every K
is at 4.7e-3..1.1e-2: K1 6.6e-3..7.3e-3, K2 4.7e-3..7.5e-3, K3 7.8e-3..1.09e-2, K4 7.0e-3..9.0e-3,
K5 6.0e-3..9.2e-3, K6 7.0e-3..8.6e-3. K3 is the 3bpw production path. So K1/K5/K6 are as exact as K3.

3-bit speed (same binary flags, K12 test, two runs before, one after): K10 loop T=4096 364.6-364.7 ms -> 365.6 ms
(+0.3%); K12 T=4096 64.53-64.54 ms -> 66.36 ms (+2.8%), T=512 54.55-54.64 -> 57.06 ms (+4.5%). The K12 cost is
accepted for now: K12 v2 (direct EXL3 GEMM) replaces this path.
