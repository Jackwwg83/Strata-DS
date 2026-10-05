# K1b progress

- [x] Read K1b, K1, existing FP8/BF16 kernels and parity tests, CMake registration and Python reference.
- [x] Confirm requested branch `feature/ds41-k1b-fp8-gemv-speed` and initially clean worktree (`cb99f4a`).
- [x] Implement allocation-free quantization/GEMV entry points and compatibility wrapper.
- [x] Implement shape-selected in-block split-K and register reuse across activation/output rows.
- [x] Extend host numerical/layout checks and GPU parity, graph replay, reuse and q-only timing checks.
- [x] Compile optimized host C++17 with clang++ and warnings as errors; pass 88 numerical cases and 288 layout geometries.
- [x] Compile/run UBSan host checks: 56 boundary cases and exhaustive conversion/layout checks pass.
- [x] Review CUDA indexing, barriers, alignment, scale sharing, integer bounds and unchanged quantization numerics.
- [x] Record actual check output, separate implementation/build/GPU statuses and old/pending-new measurement table in `K1b.REPORT.md`.
- [x] Record assumptions and reviewer commit groups; no commit, index write or push attempted.
- [ ] ASan runtime validation: instrumented build succeeds, but execution times out after 30 seconds without output. Not a passing test; cause unverified.
- [ ] Direct Python reference execution: `import torch` fails with `ModuleNotFoundError`; no PyTorch installed in the available Python. Reference was read and independent C++ reference checks pass.
- [ ] CUDA compilation, GPU numerics/graphs/sanitizers and RTX 4090 speed acceptance: not compiled: no CUDA toolkit on this machine; no NVIDIA GPU. Reviewer commands and pending measurements are in the report.
- [x] Final allowed-file-only diff and whitespace verification: exactly 7 permitted files; `git diff --check` and new-file whitespace checks pass.
