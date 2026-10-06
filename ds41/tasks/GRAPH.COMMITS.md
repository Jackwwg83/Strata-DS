# Commit plan

Git could not write the worktree index. Both `git add` and `git commit` failed:

```text
fatal: Unable to create '/Users/jackwu/Projects/Strata-DS/.git/worktrees/Strata-DS-graph/index.lock': Operation not permitted
```

No commits were created. Run these units from this worktree after the sandbox restriction is removed.
Each unit has at most five files. Each commit message ends with the required trailer.
Do not include the pre-existing untracked `ds41/tasks/DECODE.STUDY.md`.

1. `test(ds41): cover device parameters in captured decode kernels` (5 files)
   - `src/ds41/tests/graph_test_util.hpp`
   - `src/ds41/tests/graph_ops_test.cu`
   - `src/ds41/tests/graph_index_attn_test.cu`
   - `src/ds41/tests/graph_workspace_test.cu`
   - `src/ds41/tests/graph_stream_ops_test.cu`
2. `test(ds41): register synthetic decode graph tests` (1 file)
   - `cmake/ds41_engine.cmake`
3. `feat(ds41): add streamed ops and device decode parameters` (2 files)
   - `include/strata/ds41/ops.hpp`
   - `src/ds41/ops.cu`
4. `feat(ds41): read decode lengths on the device` (4 files)
   - `include/strata/ds41/kernels/k5_indexer.hpp`
   - `src/ds41/kernels/k5_indexer.cu`
   - `include/strata/ds41/kernels/k3_sparse_attn.hpp`
   - `src/ds41/kernels/k3_sparse_attn.cu`
5. `feat(ds41): expose capture workspace initialization` (4 files)
   - `include/strata/ds41/kernels/k7_hc.hpp`
   - `src/ds41/kernels/k7_hc.cu`
   - `include/strata/ds41/kernels/k8_router.hpp`
   - `src/ds41/kernels/k8_router.cu`
6. `docs(ds41): record graph kernel changes and validation limits` (3 files)
   - `ds41/tasks/GRAPH.PROGRESS.md`
   - `ds41/tasks/GRAPH.REPORT.md`
   - `ds41/tasks/GRAPH.COMMITS.md`

Use this command format for each unit, with only that unit's paths:

```sh
git add <paths>
git commit -m '<title>' -m 'Co-Authored-By: Codex <noreply@openai.com>'
```

The initial three CUDA tests and CMake registration were written before the implementation.
The legacy-stream coverage test and extra boundary checks were added during review.
Units 1 and 2 refer to APIs introduced in units 3 through 5. Compile after unit 5.
