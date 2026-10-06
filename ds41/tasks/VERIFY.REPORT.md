# DS41 verifier delivery report

Date: 2026-10-06. Worktree: `Strata-DS-verify`. Branch: `feature/ds41-verify`.
Base HEAD: `e4afe65a5ddab062d92785941d31cdc977ce5b52`.
This is source delivery with local CPU evidence. It is not a GPU acceptance result or a speed claim.
The existing untracked `DECODE.STUDY.md` was preserved.

## State by part

| Part | Code written | Compiled | Tested on GPU | Not done |
| --- | --- | --- | --- | --- |
| Engine verify / prefix commit | Yes | No CUDA build here | No | CUDA build, real logits parity, state rollback acceptance |
| T-row dense / K7 / K8 / K10 / hybrid CPU experts | Yes, T=1..4 | No CUDA build here | No | Real grouped CPU parity and overlap measurement |
| Window graph cache | Yes, key = T / parity / two capacities | No CUDA build here | No | Graph replay and capture checks on the 4090 |
| Suffix lookup | Yes, upstream algorithm with minimum-match bound fixed | Yes, AppleClang C++17 | Not required for the lookup | Real-model acceptance rate and speed |
| Generation CLI | Yes | C++ object compiled; synthetic executable linked | No | Link and run against the CUDA engine |
| Six pack acceptance executables | Yes, common source and six CMake targets | C++ object compiled; not linked to CUDA | No | All six model-backed executions |
| Local suffix / acceptance and CLI tests | Yes | Yes | These tests are synthetic | No real model evidence from these tests |
| Commits | Commands prepared, at most five files per unit | N/A | N/A | Git staging is blocked by the sandbox; see VERIFY.COMMITS.md |
| DSpark | Not part of this task | No | No | Owner decision and a later task |

I could not run CUDA compilation, because this host is macOS arm64 and has no nvcc.
I could not run the six SAGE engine tests or the real ds41_generate binary, because this host has no CUDA GPU or SAGE pack.
I could not create commits, because git could not create index.lock in the protected main repository git directory.

## Implementation and boundaries

- `Engine::verify(window, pos, logits)` returns the next token after each causal input prefix.
  A successful window stays pending until `commit(1..T)`. Step, prefill, and a second verify reject a pending window.
  Invalid sizes, positions, token IDs, and commit counts throw before changing the transaction.
- The API caps T at four. The CPU rows 5..8 fix belongs to another worktree.
  `--spec-max 5..8` prints the cap and uses four. There is no separate all-GPU T=8 path in this delivery.
- Each layer has a private 128-slot ring. Attention writes it in row order and reads that row's index list and length.
  Commit copies only accepted KV rows into the formal ring. It never replaces the whole ring with the last draft state.
- Ratio-2 partial KV and scores use private storage and a snapshot after each row.
  Commit copies the snapshot at `n_keep-1`. Rejected compressor writes never touch the formal partial group.
- Compressed KV and index keys use append space beyond the committed length. Each query has its own causal length.
  The original bytes of up to four append rows are saved on the GPU. Commit restores all rejected append bytes.
  This avoids copying a full long-context compressed cache. Physical tentative appends exist during verify, but
  no other engine operation may read them until commit. These are not committed rows.
- Candidate masks and attention IDs have independent storage per query. Shared-source layers retain the same row's
  index list. The formal step candidate buffer is untouched. Step reconstructs its scratch after commit.
- Engram reads use the tentative prefix while staging. The formal compressed-ID history is restored before launch.
  Commit appends only accepted inputs. The suffix drafter receives raw prompt IDs and emitted target IDs.
- Dense FP8 projections, Engram projection, HC mixes, routing, shared experts, and K10 use m=T.
  Attention, compressor projections, the indexer, grouped output projection, and the head retain decode row math.
  Every layer makes one m=T doorbell publication. The CPU raw API groups rows by expert.
  VRAM hits and per-row mapped-RAM quota use the same partition as step.
- A separate verifier doorbell and CPU drive thread leave `moe()` and `worker_loop()` unchanged.
  The CPU drive thread is created once per window; its overhead has not been measured.
  Only committed routes update the residency usage counts. Residency changes happen between completed windows.
- The first verifier window is eager. Later windows capture/replay one graph on `st`, keyed by T, start parity,
  ratio-1 capacity, and ratio-2 capacity. Pinned staging addresses stay fixed. Device parameters hold positions.
  Commit currently uses eager stream-ordered copies and kernels, then one host wait. Its GPU body is capturable,
  but this delivery does not cache a commit graph. CUDA failure poisons the transaction and blocks reuse.
- `verify.cu` is included after the private Impl definition in `engine.cu`. It is not a separate CMake translation unit.
  This keeps the implementation in a new file without moving the other workers' private engine code.
- `--spec suffix --spec-max T` emits per-window position, T, accepted drafts, and emitted tokens.
  It limits windows by remaining output count and context space. Optional `--eos-id` stops at the same token as plain greedy.
  Both paths emit exactly `--gen` tokens unless EOS stops them. This also fixes the old plain path's extra output token.
  Decode tokens/s excludes the first prediction produced by the prompt. It includes verify, commit, draft, and loop overhead.
  Forced-token and dump modes reject suffix speculation explicitly.

## Acceptance criteria and limitations

`verify_rows_parity`, `verify_commit_prefix`, `verify_ring_rollback`, `verify_compressor`, `verify_engram`, and
`spec_end_to_end` are separate executables. Each takes `--pack`. Without it they return 77 before creating an Engine.
They run baseline and verifier in separate child processes so two dense packs need not fit in VRAM together.

The first five compare all vocabulary logits and every greedy next token. They repeat eight transactions with step
continuations. They cover all T up to `--t` and all keep counts, except rows parity, which keeps every row.
Ring cases start at 125, 127, and 128. Compressor cases start at 2 and 3. Engram cases start at 6 and 127.
A graph-enabled test fails if it never reuses an existing verifier graph. Invalid and overlapping transaction calls
are checked. `spec_end_to_end` compares 24 generated tokens from plain and suffix paths for each requested T.
The command-level comparison below also checks the production CLI.

T=1 requires bitwise equal logits. `--exact` extends this requirement to every T. All T require exact greedy IDs.
For the default multi-row check, maximum absolute logit error must be <= 0.05 and relative L2 error <= 0.002.
These are conservative acceptance limits, not measured DS41 errors. The reason to permit a numerical bound is the
CPU expert path: rows=1 uses five phases, while grouped rows use six and transform/accumulate in a different order.
K1 and K7 retain per-row reduction structure; attention and the head use the same single-row kernels.
The relative limit is tighter than the existing K10 0.005 and K12 0.01 relative-L2 limits. Those expert-output limits
are not a proof about final logits. Exact greedy equality remains mandatory. Do not widen these limits merely to pass.
Run `--exact` first where possible and inspect any failure against the grouped CPU math.

No measured acceptance rate, GPU memory footprint, GPU latency, or speedup is available.
The tests default to fixed residency and force `DS41_ZC_QUOTA=0`. A nonzero-quota run is an additional integration
check, not a substitute for the required fixed-residency acceptance.
The long-prefix mode prefills the same prefix in both processes. It tests continuation from a common prefill state;
it does not prove prefill itself equals step for that prefix.

## Local evidence

The following is actual output from this host. The CLI mock uses a synthetic periodic target. It exercises the real
CLI generation loop and acceptance code, but it supplies a fake Engine. Its tokens are not SAGE output.

```text
$ cmake -S . -B /private/tmp/ds41-verify-cpu-build -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON -DSTRATA_NATIVE_EXPERTS=OFF
-- Configuring done (0.1s)
-- Generating done (0.0s)
-- Build files have been written to: /private/tmp/ds41-verify-cpu-build
exit=0

$ cmake --build /private/tmp/ds41-verify-cpu-build --target ds41_suffix_drafter_test ds41_generate_suffix_mock_test -j 4
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
exit=0

$ ctest --test-dir /private/tmp/ds41-verify-cpu-build -R '^ds41_(suffix_drafter|generate_suffix_mock)_test$' --output-on-failure
Test project /private/tmp/ds41-verify-cpu-build
    Start 1: ds41_suffix_drafter_test
1/2 Test #1: ds41_suffix_drafter_test .........   Passed    0.17 sec
    Start 2: ds41_generate_suffix_mock_test
2/2 Test #2: ds41_generate_suffix_mock_test ...   Passed    0.38 sec

100% tests passed, 0 tests failed out of 2

Total Test time (real) =   0.55 sec
exit=0

$ /private/tmp/ds41-verify-cpu-build/ds41_suffix_drafter_test
RESULT pass suffix_drafter acceptance=zero,partial,full overlap=1 longest=1 ties=1
exit=0

$ /private/tmp/ds41-verify-cpu-build/ds41_generate_suffix_mock_test
RESULT pass generate_suffix_mock windows=110 multi=45 rejected=9 gen_bounds=1 eos=1 prefill=1
exit=0

$ clang++ -std=c++17 -Wall -Wextra -Werror -Iinclude -c src/ds41/ds41_generate.cpp -o /private/tmp/ds41_generate.o
exit=0

$ clang++ -std=c++17 -Wall -Wextra -Werror -Iinclude -c src/ds41/tests/verify_engine_test.cpp -o /private/tmp/ds41_verify_engine_test.o
exit=0

$ clang++ -std=c++17 -Wall -Wextra -Werror -Iinclude src/ds41/tests/generate_suffix_mock_test.cpp src/ds41/suffix_drafter.cpp -o /private/tmp/ds41_generate_mock_werror
exit=0

$ /private/tmp/ds41_generate_mock_werror
RESULT pass generate_suffix_mock windows=110 multi=45 rejected=9 gen_bounds=1 eos=1 prefill=1
exit=0

$ git diff --check
exit=0
```

The final engine-test object was recompiled after adding invalid-call checks. It exited 0 with no compiler output.
A source check also produced:

```text
PASS static checks: enqueue has no allocation or host wait; moe and worker_loop are unchanged
```

This source check is not a CUDA compiler or a runtime capture test.
UndefinedBehaviorSanitizer built and ran the suffix test:

```text
$ clang++ -std=c++17 -O1 -g -fsanitize=undefined -fno-omit-frame-pointer -Wall -Wextra -Werror -Iinclude src/ds41/suffix_drafter.cpp src/ds41/tests/suffix_drafter_test.cpp -o /private/tmp/ds41_suffix_ubsan
exit=0
$ /private/tmp/ds41_suffix_ubsan
RESULT pass suffix_drafter acceptance=zero,partial,full overlap=1 longest=1 ties=1
exit=0
```

After adding the explicit `--spec-max 8` cap and exact context-boundary case, the final rebuild and CTest run produced:

```text
[ 25%] Building CXX object CMakeFiles/strata_ds41_suffix.dir/src/ds41/suffix_drafter.cpp.o
[ 50%] Linking CXX static library libstrata_ds41_suffix.a
[ 50%] Built target strata_ds41_suffix
[ 75%] Linking CXX executable ds41_suffix_drafter_test
ld: warning: search path '/opt/homebrew/opt/tcl-tk/lib' not found
[100%] Built target ds41_suffix_drafter_test
[ 50%] Built target strata_ds41_suffix
[ 75%] Building CXX object CMakeFiles/ds41_generate_suffix_mock_test.dir/src/ds41/tests/generate_suffix_mock_test.cpp.o
[100%] Linking CXX executable ds41_generate_suffix_mock_test
ld: warning: search path '/opt/homebrew/opt/tcl-tk/lib' not found
[100%] Built target ds41_generate_suffix_mock_test
Test project /private/tmp/ds41-verify-cpu-build
    Start 1: ds41_suffix_drafter_test
1/2 Test #1: ds41_suffix_drafter_test .........   Passed    0.44 sec
    Start 2: ds41_generate_suffix_mock_test
2/2 Test #2: ds41_generate_suffix_mock_test ...   Passed    0.41 sec

100% tests passed, 0 tests failed out of 2

Total Test time (real) =   0.85 sec
spec: cap T at 4 until CPU rows 5..8 are validated
RESULT pass generate_suffix_mock windows=134 multi=60 rejected=11 gen_bounds=1 eos=1 prefill=1
```

Two local checks did not complete normally:

1. The first CMake configure kept the default `STRATA_NATIVE_EXPERTS=ON`. Its unrelated ggml dependency fetch failed:

   ```text
   fatal: unable to access 'https://github.com/ggml-org/llama.cpp.git/': Could not resolve host: github.com
   Failed to clone repository: 'https://github.com/ggml-org/llama.cpp.git'
   -- Configuring incomplete, errors occurred!
   ```

   The successful build above explicitly disabled that dependency.
2. The combined AddressSanitizer/UndefinedBehaviorSanitizer executable produced no output and did not return during
   repeated waits. It was interrupted (exit 130). Its binary existed, but there is no passing ASan result.
   UBSan alone completed as shown above.

## RTX 4090 build and tests

These commands are prepared, not executed here. Run them in Bash on the GPU box. CUDA_ARCH is 89. Native experts
are disabled only to avoid an unrelated llama.cpp dependency; the DS41 EXL3 CPU engine is still built.

```bash
cd /workspace/Strata-DS-verify
export CUDA_ARCH=89
export DS41_ZC_QUOTA=0
export DS41_GRAPH=1
cmake -S . -B build \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_NATIVE_EXPERTS=OFF -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
cmake --build build -j 8 --target ds41_generate ds41_suffix_drafter_test \
  ds41_generate_suffix_mock_test verify_rows_parity verify_commit_prefix \
  verify_ring_rollback verify_compressor verify_engram spec_end_to_end
ctest --test-dir build -R '^ds41_(suffix_drafter|generate_suffix_mock)_test$' --output-on-failure
# These must be SKIPPED with return code 77. They are not pack acceptance.
ctest --test-dir build -R '^(verify_rows_parity|verify_commit_prefix|verify_ring_rollback|verify_compressor|verify_engram|spec_end_to_end)$' --output-on-failure

mkdir -p /workspace/results/verify
VERIFY_FIXED=(--pack /workspace/pack-sage --threads 32 \
  --expert-profile ds41/data/expert-profile.bin --vram-slots 64 --ram-budget-gib 0)
set -o pipefail
for VERIFY_MODE in 0 1; do
  for VERIFY_TEST in verify_rows_parity verify_commit_prefix verify_ring_rollback verify_compressor verify_engram spec_end_to_end; do
    DS41_GRAPH="$VERIFY_MODE" "build/$VERIFY_TEST" "${VERIFY_FIXED[@]}" --t 4 \
      --ids /workspace/results/quality/sage159/code_py_0.ids \
      2>&1 | tee "/workspace/results/verify/${VERIFY_TEST}-graph${VERIFY_MODE}.log"
    test "${PIPESTATUS[0]}" -eq 0 || exit 1
  done
done
# Exact mode is an extra diagnostic for multi-row reduction differences.
build/verify_rows_parity "${VERIFY_FIXED[@]}" --t 4 --exact \
  --ids /workspace/results/quality/sage159/code_py_0.ids \
  2>&1 | tee /workspace/results/verify/rows-exact.log
# Inspect a nonzero result. Do not replace the default greedy-equality requirement.

# Cross indexer capacity and candidate-block boundaries. Both children prefill the same real prefix.
for VERIFY_START in 1023 1024 4095 4096 16383 16384 260000; do
  build/verify_commit_prefix "${VERIFY_FIXED[@]}" --t 4 --prefill-prefix \
    --start "$VERIFY_START" --max-seq 260256 --ids /workspace/long260k.ids \
    2>&1 | tee "/workspace/results/verify/long-${VERIFY_START}.log"
  test "${PIPESTATUS[0]}" -eq 0 || exit 1
done
```

The long cases reload one pack for each baseline and verifier case. They can take substantial time. They deliberately
check all keep counts. A test failure leaves its trace files in `/tmp/ds41-verify-*` for inspection.
For a GPU-only comparison of row math, use a fully resident configuration only if that pack fits. The fixed 64-slot
configuration above intentionally tests the hybrid path. Do not compare runs with automatic residency.

Production CLI comparison and measured output (same residency in both runs):

```bash
cd /workspace/Strata-DS-verify
export DS41_ZC_QUOTA=0
export DS41_GRAPH=1
python3 - <<'PY'
from pathlib import Path
import re
source = Path('/workspace/results/quality/sage159/code_py_0.ids')
ids = [int(x) for x in re.findall(r'-?\d+', source.read_text())]
assert ids
Path('/workspace/results/verify/prompt.ids').write_text(','.join(map(str, ids[:128])))
PY
for VERIFY_SPEC in none suffix; do
  build/ds41_generate --pack /workspace/pack-sage \
    --ids @/workspace/results/verify/prompt.ids --gen 256 --threads 32 \
    --max-seq 4096 --expert-profile ds41/data/expert-profile.bin \
    --vram-slots 64 --ram-budget-gib 0 --adapt-every 0 \
    --spec "$VERIFY_SPEC" --spec-max 4 \
    > "/workspace/results/verify/generate-${VERIFY_SPEC}.log" \
    2> "/workspace/results/verify/generate-${VERIFY_SPEC}.stderr"
  test "$?" -eq 0 || exit 1
done
python3 - <<'PY'
from pathlib import Path
base = Path('/workspace/results/verify')
def tokens(name):
    text = (base / f'generate-{name}.log').read_text()
    rows = [s for s in text.splitlines() if s.startswith('generated:')]
    assert len(rows) == 1
    return [int(x) for x in rows[0].split()[1:]]
a, b = tokens('none'), tokens('suffix')
assert len(a) == len(b) == 256, (len(a), len(b))
assert a == b, next((i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), None)
print('RESULT pass real_cli_greedy_equal tokens=256')
for name in ('none', 'suffix'):
    for line in (base / f'generate-{name}.log').read_text().splitlines():
        if line.startswith('decode_tokens '): print(name, line)
PY
# Repeat with --prefill in both CLI invocations to check the prefill-to-verify transition.
# Repeat on the full long prompt with --ids @/workspace/long260k.ids --prefill --max-seq 262144.
```

Remaining acceptance is explicit: compile the CUDA translation unit, run every pack test with eager and graph modes,
inspect exact-mode differences, verify the long-context boundaries, and compare production CLI tokens and timings.
Only after these pass can the verifier be called GPU-tested. DSpark remains outside this delivery.
