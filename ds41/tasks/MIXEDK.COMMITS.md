# Mixed-K commit plan

No commits were created. The sandbox denies writes to the worktree index at
`/Users/jackwu/Projects/Strata-DS/.git/worktrees/Strata-DS-mixedk/index`.
The first `git add` failed while creating `index.lock`. Do not bypass this
protection. Run these commands after the worktree has normal Git write access.
Each unit has at most five files. All messages have the requested trailer.
No new upstream files were needed, so there is no pristine import to stage.

```sh
git add third_party/exllamav3_gpu/quant/exl3_gemv.cu \
  third_party/exllamav3_gpu/quant/exl3_gemv.cuh \
  third_party/exllamav3_gpu/quant/exl3_gemv_kernel.cuh \
  src/ds41/kernels/k10/pipeline.cuh src/ds41/kernels/k10_exl3_moe.cu
git commit -m 'Dispatch decode GEMV jobs by projection bit width' \
  -m 'Co-Authored-By: Codex <noreply@openai.com>'

git add third_party/exllamav3_gpu/quant/reconstruct.cu \
  third_party/exllamav3_gpu/quant/reconstruct.cuh \
  src/ds41/kernels/k12/pipeline.cuh src/ds41/kernels/k12/workspace.hpp \
  src/ds41/kernels/k12_exl3_moe_prefill.cu
git commit -m 'Dispatch prefill reconstruction from device projection jobs' \
  -m 'Co-Authored-By: Codex <noreply@openai.com>'

git add third_party/exllamav3_gpu/strata.patch third_party/exllamav3_gpu/README.md \
  src/ds41/kernels/k10/check_host.py src/ds41/kernels/k10/check_schedule.py \
  src/ds41/kernels/k12/check_host.py
git commit -m 'Audit mixed-rate loads and preserve vendor provenance' \
  -m 'Co-Authored-By: Codex <noreply@openai.com>'

git add ds41/ci/make_mixedk_golden.py src/ds41/tests/mixedk_fixture.hpp \
  src/ds41/tests/k10_exl3_moe_test.cu src/ds41/tests/k11_cpu_moe_test.cpp \
  src/ds41/tests/k12_exl3_moe_prefill_test.cu
git commit -m 'Extend MoE acceptance with mixed projection rates' \
  -m 'Co-Authored-By: Codex <noreply@openai.com>'

git add third_party/exllamav3_moe/moe_mul1.h third_party/exllamav3_moe/README.md \
  third_party/exllamav3_moe/tests/strata_mixedk_test.cpp \
  third_party/exllamav3_moe/tests/check_mixedk_host.py
git commit -m 'Verify CPU matrix rates and expert relocation' \
  -m 'Co-Authored-By: Codex <noreply@openai.com>'

git add ds41/tasks/MIXEDK.PROGRESS.md ds41/tasks/MIXEDK.REPORT.md \
  ds41/tasks/MIXEDK.COMMITS.md
git commit -m 'Record mixed-rate verification and GPU acceptance steps' \
  -m 'Co-Authored-By: Codex <noreply@openai.com>'
```
