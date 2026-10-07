# Review fix commit plan

Branch: `fix/ds41-revfix-kernels`. Base: `e2d8499` from `feature/ds41`, PR #10.
No commit was made. The sandbox rejected `git add` before it could write the index:

```text
fatal: Unable to create '/Users/jackwu/Projects/Strata-DS/.git/worktrees/Strata-DS-revfix-kernels/index.lock': Operation not permitted
```

Run these commands from the repository after reviewing the changes. Each unit has at most five files.
Do not stage the pre-existing untracked review inputs, `rev_kernels_out.md` and `rev_tiers_out.md`.

```sh
git add third_party/exllamav3_moe/moe_mul1.cpp \
  third_party/exllamav3_moe/README.md \
  third_party/exllamav3_moe/tests/revfix_test.cpp \
  third_party/exllamav3_moe/tests/check_revfix_host.py \
  third_party/exllamav3_moe/tests/check_mixedk_host.py
git commit -m "ds41: validate CPU experts and fix mixed-rate staging ownership" \
  -m "Co-Authored-By: Codex <noreply@openai.com>"

git add src/ds41/tests/test_validation.hpp \
  src/ds41/tests/test_validation_test.cpp \
  src/ds41/tests/k2_fp8_gemm_test.cu \
  src/ds41/tests/k1c_fp8_gemv_test.cu \
  src/ds41/tests/k15_hc_prefill_test.cu
git commit -m "ds41: reject non-finite errors in kernel acceptance tests" \
  -m "Co-Authored-By: Codex <noreply@openai.com>"

git add src/ds41/tests/prefill_ops_test.cu \
  src/ds41/tests/k5_indexer_test.cu \
  src/ds41/tests/k14_indexer_prefill_test.cu \
  cmake/ds41_engine.cmake
git commit -m "ds41: check unique top-k IDs and register review regressions" \
  -m "Co-Authored-By: Codex <noreply@openai.com>"

git add src/ds41/wo_a_fp8.cu \
  src/ds41/tests/wo_a_fp8_test.cu \
  src/ds41/tests/check_wo_a_fp8_host.py
git commit -m "ds41: decode E8M0 boundaries with BF16 reference rounding" \
  -m "Co-Authored-By: Codex <noreply@openai.com>"

git add ds41/tasks/REVFIX.REPORT.md ds41/tasks/REVFIX.COMMITS.md
git commit -m "ds41: record review fix evidence and target validation commands" \
  -m "Co-Authored-By: Codex <noreply@openai.com>"
```
