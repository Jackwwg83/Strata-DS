# Hybrid commit units

No commit was created. The first test commit attempt failed:

```text
fatal: Unable to create '/Users/jackwu/Projects/Strata-DS/.git/worktrees/Strata-DS-hybrid/index.lock': Operation not permitted
```

The shared Git metadata is outside this sandbox's writable roots. The source worktree is writable.
Do not stage the pre-existing untracked `ds41/tasks/DECODE.STUDY.md`.
The tests were written before the production changes. The final test files also include review fixes.
Each unit below touches at most five files. Run these commands in this worktree when Git metadata is writable.

```sh
git add src/ds41/tests/hybrid_decode_test.cu src/ds41/tests/hybrid_host_table_test.cu cmake/ds41_engine.cmake
git commit -m 'test(ds41): cover hybrid expert routing and mapped memory' \
  -m 'Co-Authored-By: Codex <noreply@openai.com>'

git add include/strata/ds41/host_experts.hpp src/ds41/host_experts.cpp src/ds41/vram_experts.cu
git commit -m 'feat(ds41): publish mapped RAM expert descriptors at tier safe points' \
  -m 'Co-Authored-By: Codex <noreply@openai.com>'

git add include/strata/ds41/doorbell.hpp src/ds41/doorbell.cu
git commit -m 'feat(ds41): split RAM misses with a device quota' \
  -m 'Co-Authored-By: Codex <noreply@openai.com>'

git add src/ds41/engine.cu include/strata/ds41/engine.hpp src/ds41/ds41_generate.cpp
git commit -m 'feat(ds41): run hybrid decode and report expert counts' \
  -m 'Co-Authored-By: Codex <noreply@openai.com>'

git add ds41/tasks/HYBRID.PROGRESS.md ds41/tasks/HYBRID.REPORT.md ds41/tasks/HYBRID.COMMITS.md
git commit -m 'docs(ds41): record hybrid validation and GPU acceptance commands' \
  -m 'Co-Authored-By: Codex <noreply@openai.com>'
```
