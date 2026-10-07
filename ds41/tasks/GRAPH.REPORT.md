# Decode graph kernel report

## Result and status

The kernel-side implementation and synthetic tests are written. `engine.cu` is unchanged.
This is not an end-to-end graph integration result. The reviewer still owns `Engine::step()` integration.
No CUDA source was compiled or executed on this host.

| Part | Code written | CUDA compiled | Tested on GPU | Remaining acceptance |
| --- | --- | --- | --- | --- |
| 1. Ops streams and device token/position/row copy | Yes | No | No | Build; run `graph_ops_test` and `graph_stream_ops_test` |
| 2. K5 device length and fixed capacity | Yes | No | No | Bitwise parity and replay tests; measure long-context speed |
| 3. K3 device attention length | Yes | No | No | Bitwise parity and replay tests |
| 4. K7/K8 explicit workspace init | Yes | No | No | Capture before any eager compute, after explicit init |
| 5. Device logits argmax | Yes | No | No | Compare all replay results with host `std::max_element` |
| 6. K10 and doorbell capture review | Source checked; synthetic K10 test added | No | No | Run `graph_workspace_test` and existing `doorbell_test` |
| Synthetic graph tests and CMake registration | Four tests plus shared helper | No | No | Build and run all four tests on RTX 4090 |
| Portable existing C++ tests | Unchanged | Eight native CPU targets built | Not applicable | Eight passed; these do not validate CUDA |
| Commits | Plan written | Not applicable | Not applicable | Git index is outside the permitted write roots |

I could not run CUDA compilation, because this ARM64 macOS host has no `nvcc` or CUDA toolkit.
I could not run the GPU tests or Compute Sanitizer, because this host has no CUDA GPU.
I could not finish the full native CPU build, because existing CPU kernels include x86 intrinsics on ARM64.
I could not create commits, because the sandbox blocked the worktree's `index.lock`.
The exact commit units are in [GRAPH.COMMITS.md](GRAPH.COMMITS.md).
The pre-existing untracked `DECODE.STUDY.md` was not changed or added to the commit plan.

## Interfaces and integration contract

All 21 existing functions in `ops.hpp` now have a trailing `cudaStream_t stream = 0`.
All 29 launches in `ops.cu` use the supplied stream. Existing calls retain their default-stream behavior.
There are no allocations or host copies in these ops.

New ops:

- `embed_device(table, token_dev, h, stream)` reads the token on each execution.
- `window_index_device(pos_dev, idx, stream)` reads the window position on each execution.
- `rope_device(v, n_vec, stride, cs_table, pos_dev, offset, inverse, stream)` reads row
  `int64_t(*pos_dev) + offset`. Each table row has `kRopeDim` floats. Pass the base of the selected
  plain or YaRN table. A negative row does no work, for the first incomplete ratio-2 group.
- `row_copy_device(dst, src, row_bytes, pos_dev, div, mod, stream)` uses byte addressing:
  `dst + ((*pos_dev / div) % mod) * row_bytes`. It also supports unaligned row sizes.
  Use positive constant `div` and `mod`, a nonnegative position, and nonoverlapping buffers.
- `argmax_logits(logits, token_dev, stream)` reduces all 129280 floats without scratch allocation.
  It selects the lowest index on ties. It matches the current host `std::max_element` behavior for
  infinities, signed zeros, and NaNs: index-zero NaN wins; other NaNs are ignored.

`indexer_topk_device` and `candidate_blocks_device` take `pos_dev`, constant `ratio`, and `t_cap`.
The device computes `t = (int64_t(*pos_dev) + 1) / ratio`. The contract is `0 <= t <= t_cap`.
Only the live prefixes are written. A zero live length performs no reads of score/key inputs and no output writes.
For positive capacity the captured nodes still launch and return on the device at zero live length.
A zero capacity submits no nodes. Output IDs remain ascending and include the constant offset.

The score grid depends only on capacity. The scalar scorer is recorded for up to 512 entries.
For a larger capacity, the tensor scorer is also recorded. Uniform device guards select exactly the old
`t <= 512` or `t > 512` arithmetic path. The inactive scorer returns without writing.
Selection always uses the existing scratch-free radix implementation. Its device setup clamps `k` to `t`.
Candidate selection uses the same radix code, including the forced last block and the `-inf` exclusion.
This avoids the old parallel selector's metadata layout when a large capacity has a short live prefix.
It also handles `k == t` without a host branch.

This is a deliberate performance tradeoff. Large-length selection uses one CTA instead of the old parallel
selector. Numerical parity is covered by the new tests, but has not been observed on a GPU here.
Long-context latency has not been measured. Do not claim a speedup from this change.

The header permits any fixed nonnegative capacity. For one graph across all positions, choose the maximum
compressed length for the allocated context. Optional buckets use `max(1, next_power_of_two(t))`.
Each bucket needs its own captured graph. Changing a bucket or a pointer requires a separate graph.

`sparse_attn_decode_device` handles one query. It reads a device `int t`, then computes
`n_idx = kWindow + min(kIndexTopK, t)`. The index buffer has capacity 640. The grid is constant.
The existing attention body processes only the live indices and retains its original arithmetic order.
The engine must stage or compute this `t` on the same stream before attention. This API reads `t`, not `pos`.
The existing multi-query, host-length API is retained.

Call `kernels::hc_init()` and `kernels::router_init()` on the selected device before capture.
They allocate the existing maximum-size, process-lifetime scratch buffers. Repeated calls do nothing.
Legacy eager first-use allocation remains for compatibility. Calls and graph replays that share one device's
scratch must not overlap, even on different streams. Initializers must never run inside capture.

Keep every device scalar and buffer at a fixed address through all replays. Publish changed values on the
capture stream before graph launch, or include copies from fixed pinned staging addresses in the graph.
Do not change a scalar while the graph is running. Launch arguments use only constants, capacities, and pointers.
Compressor completion and any CPU preparation remain the reviewer's engine-integration responsibility.

K10 already uses caller-owned scratch. Its five launches, including the GEMV wrapper calls, preserve the
supplied stream. No allocation, host copy, or host synchronization occurs in its decode entry point.
Doorbell construction allocates mapped memory outside capture. `publish` and `wait_add` only launch on the
supplied stream. Reset only after the previous replay finishes. Its existing test creates a nonblocking
stream and replays one captured graph twice. Neither production implementation needed a change.

## Tests written

The first three tests and registration were written before the implementation. The fourth test and additional
boundary cases were added during review. All data is synthetic. No model pack or golden fixture is needed.
The helper synchronizes pageable test uploads before any nonblocking-stream work. These test-only host waits
are outside capture. Every graph is captured once before its replay loop.

- `graph_ops_test`: bitwise host-parameter parity and direct-device parity for embedding, window indices,
  both RoPE directions and offsets -1/0/3, ring and compressed row copies. Positions cross 128/256/512 and
  return to earlier values. Argmax uses changing logits, ties, infinities, NaNs, and the last vocabulary entry.
- `graph_stream_ops_test`: every legacy op is captured on a nonblocking stream and compared bit for bit with
  its default-stream call. Outputs are poisoned before each replay. In-place inputs are reset in the graph.
- `graph_index_attn_test`: K5 capacities 1/32/512/4097/8192; exhaustive live lengths 0..32 in the small bucket;
  larger boundary lengths include 511/512/513 and 4095/4096/4097. It covers ratios 1/2, incomplete groups,
  k=0/1/512/cap+1, masked and all-masked scores, full selection, t=0, and unchanged output tails.
  Candidate tests use blocks 1/8/31, random FP32 scores, ties, and all `-inf` scores.
  K3 uses changing lengths through 512 and beyond, with mixed window/compressed/empty rows.
  Invalid tail indices expose accidental reads beyond the live length.
- `graph_workspace_test`: K7/K8 capture happens after explicit init and before any eager compute.
  Inputs change between replays. The K10 graph uses synthetic packed K3 weights and changes its device
  expert selection from all absent to active slots. It checks direct-call parity and finite, nonzero active output.
- Existing `doorbell_test`: retained for the mapped CPU handoff on a non-default stream.

## Linux RTX 4090 commands

Run after these files are present in `/workspace/Strata-DS`. Native GGML experts are disabled here to avoid
an unrelated network fetch. This does not disable the DS41 EXL3 path. No new dependency is required.

```sh
cd /workspace/Strata-DS
export CUDA_ARCH=89
cmake -S . -B build \
  -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_NATIVE_EXPERTS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)" --target \
  ds41_generate graph_ops_test graph_stream_ops_test graph_index_attn_test graph_workspace_test \
  k3_sparse_attn_test k5_indexer_test k7_hc_test k8_router_test doorbell_test prefill_ops_test
ctest --test-dir build --output-on-failure --no-tests=error \
  -R '^(graph_ops_test|graph_stream_ops_test|graph_index_attn_test|graph_workspace_test|k3_sparse_attn_test|k5_indexer_test|k7_hc_test|k8_router_test|doorbell_test|prefill_ops_test)$'
for test in graph_ops_test graph_stream_ops_test graph_index_attn_test graph_workspace_test; do
  compute-sanitizer --tool memcheck --error-exitcode 1 "build/$test" || exit 1
done
compute-sanitizer --tool synccheck --error-exitcode 1 build/graph_index_attn_test
```

Require actual execution with no skipped tests. CTest exit code alone is insufficient if it reports GPU skips.
The original `k10_exl3_moe_test` needs a model pack and golden files. The new `graph_workspace_test` covers
synthetic K10 graph replay instead. No real model output or full `Engine::step()` capture was validated here.
For the rule-5 compile checks, use the same CMake command with architectures 86 and 120 in separate build
folders. CUDA 12.8 is required for the sm_120 build. Those builds were not run here either.

## Actual local checks

The following text is real command output from this session. CPU checks do not establish CUDA correctness.

### Environment

```text
$ uname -sm
Darwin arm64
$ nvcc --version
zsh:1: command not found: nvcc
```

### Configure attempts

Initial command:

```sh
cmake -S . -B /private/tmp/ds41-graph-cpu-build \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
```

The default native-experts dependency could not be fetched. Output excerpt:

```text
Cloning into 'strata_llamacpp-src'...
fatal: unable to access 'https://github.com/ggml-org/llama.cpp.git/': Could not resolve host: github.com
Cloning into 'strata_llamacpp-src'...
fatal: unable to access 'https://github.com/ggml-org/llama.cpp.git/': Could not resolve host: github.com
Cloning into 'strata_llamacpp-src'...
fatal: unable to access 'https://github.com/ggml-org/llama.cpp.git/': Could not resolve host: github.com
Had to git clone more than once: 3 times.
CMake Error at strata_llamacpp-subbuild/strata_llamacpp-populate-prefix/tmp/strata_llamacpp-populate-gitclone.cmake:50 (message):
  Failed to clone repository: 'https://github.com/ggml-org/llama.cpp.git'

```

Offline configure command:

```sh
cmake -S . -B /private/tmp/ds41-graph-cpu-build \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_NATIVE_EXPERTS=OFF -DCMAKE_BUILD_TYPE=Release
```

```text
-- Configuring done (0.1s)
-- Generating done (0.0s)
-- Build files have been written to: /private/tmp/ds41-graph-cpu-build
```

### Full CPU build attempt

```sh
cmake --build /private/tmp/ds41-graph-cpu-build -j 4
```

Exit status: 2. Output excerpt:

```text
[ 66%] Built target controller_test
In file included from /Users/jackwu/Projects/Strata-DS-graph/src/kernels/cpu/pool.cpp:9:
/Library/Developer/CommandLineTools/usr/lib/clang/17/include/immintrin.h:14:2: error: "This header is only meant to be used on x86 and x64 architecture"
   14 | #error "This header is only meant to be used on x86 and x64 architecture"
      |  ^
fatal error: too many errors emitted, stopping now [-ferror-limit=]
20 errors generated.
make[2]: *** [CMakeFiles/strata_kernels_cpu.dir/src/kernels/cpu/pool.cpp.o] Error 1
make[1]: *** [CMakeFiles/strata_kernels_cpu.dir/all] Error 2
make: *** [all] Error 2
```

### Portable targets and CTest

```sh
cmake --build /private/tmp/ds41-graph-cpu-build --target \
  gguf_reader_test gguf_split_test suffix_drafter_test controller_test draft_policy_test \
  conv_cache_test coupled_draft_test bf16_bits_test -j 4
ctest --test-dir /private/tmp/ds41-graph-cpu-build --output-on-failure \
  -R '^(gguf_reader_test|gguf_split_test|suffix_drafter_test|controller_test|draft_policy_test|conv_cache_test|coupled_draft_test|bf16_bits_test)$'
```

Both commands exited 0. Full output:

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
Test project /private/tmp/ds41-graph-cpu-build
    Start 1: gguf_reader_test
1/8 Test #1: gguf_reader_test .................   Passed    0.42 sec
    Start 2: gguf_split_test
2/8 Test #2: gguf_split_test ..................   Passed    0.40 sec
    Start 3: suffix_drafter_test
3/8 Test #3: suffix_drafter_test ..............   Passed    0.38 sec
    Start 4: controller_test
4/8 Test #4: controller_test ..................   Passed    0.40 sec
    Start 5: draft_policy_test
5/8 Test #5: draft_policy_test ................   Passed    0.37 sec
    Start 6: conv_cache_test
6/8 Test #6: conv_cache_test ..................   Passed    0.38 sec
    Start 7: coupled_draft_test
7/8 Test #7: coupled_draft_test ...............   Passed    0.39 sec
    Start 8: bf16_bits_test
8/8 Test #8: bf16_bits_test ...................   Passed    5.39 sec

100% tests passed, 0 tests failed out of 8

Total Test time (real) =   8.13 sec
```

### Source audit

A temporary Python audit checked public signatures, stream arguments, the K10 launch path, unchanged engine
and doorbell files, test registration, and `git diff --check`. This audit is not a CUDA compiler. Its first run
had an incorrect expected API count (27 instead of 26); the audit was corrected to compare against HEAD.
The corrected command was `python3 /private/tmp/ds41_graph_audit.py`. Output:

```text
PASS: 26 ops declarations and definitions use an explicit stream.
PASS: all 29 ops launches use the supplied stream.
PASS: K10 and its GEMV wrapper use stream launches without host allocation, copy, or sync.
PASS: engine.cu and doorbell.cu are unchanged.
PASS: all four synthetic graph tests are registered in CMake.
PASS: git diff --check.
This is a source audit. It does not compile or execute CUDA.
```
