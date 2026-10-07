# Hybrid staging commit units

The reviewer committed the prior direct-path work and merged `feature/ds41`.
This follow-up starts at `e4afe65` on `feature/ds41-hybrid`. No new commit was made in this session.
Git metadata is read-only under the current sandbox, so no metadata write was attempted.
The earlier report recorded an `index.lock: Operation not permitted` failure for this shared worktree.
Source changes are present and reviewable. Preserve the pre-existing untracked `ds41/tasks/DECODE.STUDY.md`.

The tests were edited before production code. Units 1 and 2 below belong together for compilation;
the test-first unit references the new staging API. Each commit touches at most five files.
Run these commands only where Git metadata is writable:

```sh
cd /workspace/Strata-DS-hybrid
git add src/ds41/tests/expert_staging_test.cu src/ds41/tests/hybrid_decode_test.cu
git commit -m 'test(ds41): check staged expert bytes and decode parity' \
  -m 'Co-Authored-By: Codex <noreply@openai.com>'

git add include/strata/ds41/expert_staging.hpp src/ds41/expert_staging.cu \
  include/strata/ds41/doorbell.hpp src/ds41/doorbell.cu cmake/ds41_engine.cmake
git commit -m 'feat(ds41): stream mapped expert blobs into fixed VRAM slots' \
  -m 'Co-Authored-By: Codex <noreply@openai.com>'

git add src/ds41/engine.cu
git commit -m 'feat(ds41): overlap expert staging with shared decode work' \
  -m 'Co-Authored-By: Codex <noreply@openai.com>'

git add tools/ds41/zc_stage_sweep.sh ds41/tasks/HYBRID.REPORT.md \
  ds41/tasks/HYBRID.PROGRESS.md ds41/tasks/HYBRID.COMMITS.md
git commit -m 'docs(ds41): record staging limits and fixed-residency GPU checks' \
  -m 'Co-Authored-By: Codex <noreply@openai.com>'
```
