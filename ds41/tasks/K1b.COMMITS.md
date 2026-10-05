# Reviewer commit plan

No commit, index write, or push was attempted. The reviewer can commit the final working-tree files in
these two groups; each group contains at most five files.

1. `ds41: add allocation-free FP8 GEMV with split-K and row reuse`
   - `include/strata/ds41/fp8_gemv.hpp`
   - `src/ds41/kernels/fp8_gemv.cu`
   - `src/ds41/kernels/fp8_gemv_parity.cpp`

   Includes the new API, compatibility wrapper, shape dispatch, host model, graph/reuse parity checks,
   q-only timings and old/new comparison table. CUDA build and GPU acceptance are pending.

2. `docs(ds41): record K1b host validation and GPU handoff`
   - `ds41/tasks/K1b.QUESTIONS.md`
   - `ds41/tasks/K1b.PROGRESS.md`
   - `ds41/tasks/K1b.REPORT.md`
   - `ds41/tasks/K1b.COMMITS.md`

   Records assumptions, actual local check output, unmeasured GPU acceptance and this commit plan.
