# Commit plan

No commit was created. The sandbox denied staging on 2026-10-06:

```text
$ git add include/strata/ds41/verify.hpp src/ds41/tests/suffix_drafter_test.cpp src/ds41/tests/verify_engine_test.cpp
fatal: Unable to create '/Users/jackwu/Projects/Strata-DS/.git/worktrees/Strata-DS-verify/index.lock': Operation not permitted
```

The worktree git directory is outside the writable roots. No permission bypass was attempted.
Run the following from `/Users/jackwu/Projects/Strata-DS-verify` after reviewing the files.
Each commit contains at most five files. Do not add the owner's untracked `DECODE.STUDY.md`.

```sh
git add include/strata/ds41/verify.hpp include/strata/ds41/engine.hpp \
  include/strata/ds41/suffix_drafter.hpp src/ds41/suffix_drafter.cpp \
  src/ds41/tests/suffix_drafter_test.cpp
git commit -m "Add verifier contract and suffix lookup tests" \
  -m "Co-Authored-By: Codex <noreply@openai.com>"

git add src/ds41/verify.cu src/ds41/engine.cu src/ds41/tests/verify_engine_test.cpp
git commit -m "Add tentative multi-row verification and prefix commit" \
  -m "Co-Authored-By: Codex <noreply@openai.com>"

git add src/ds41/ds41_generate.cpp src/ds41/tests/generate_suffix_mock_test.cpp \
  CMakeLists.txt cmake/ds41_suffix.cmake cmake/ds41_engine.cmake
git commit -m "Use suffix windows in generation and register verifier tests" \
  -m "Co-Authored-By: Codex <noreply@openai.com>"

git add ds41/tasks/VERIFY.PROGRESS.md ds41/tasks/VERIFY.REPORT.md ds41/tasks/VERIFY.COMMITS.md
git commit -m "Record local evidence and GPU acceptance commands" \
  -m "Co-Authored-By: Codex <noreply@openai.com>"
```
