# Hybrid decode progress

Every item was reviewed before delivery. A checked implementation item means code is written, not GPU-tested.
See `HYBRID.REPORT.md` for the state of each part and the real local output.

- [x] Read the task rules, decode review, tier code, doorbell, K10, and upstream mapped-memory precedent.
- [x] Write synthetic split, memory parity, table lifecycle, and graph replay tests before production changes.
- [x] Register the RAM arena and maintain the fixed device descriptor table at tier safe points.
- [x] Publish a deterministic CPU / VRAM / zero-copy split and per-call K10 descriptors for m=1..8.
- [x] Integrate per-layer device quotas and per-step counts. Keep K10 math and `step()` unchanged.
- [x] Cover quota 0..6, no tiers, null quota, all tiers, inactive routes, and mixed expert sizes in tests.
- [x] Add a test for the real adaptive VRAM swap callbacks and nonblocking-stream RAM reads.
- [x] Check the source diff and compile the generator object with strict warnings on macOS.
- [x] Build and run all eight available portable CTest targets. All eight passed.
- [x] Attempt full CPU and CUDA configurations/builds. Record the real failures and limitations.
- [x] Document exact Linux build, synthetic GPU tests, forced decode quota sweep, and count checks.
- [x] Attempt the first commit. Record the sandbox refusal and five commit units, each at most five files.
- [x] Preserve the user's untracked `DECODE.STUDY.md`.
- [x] Review all items and separate written, compiled, GPU-tested, and blocked work.

Environment-blocked acceptance:

- [ ] Compile the changed CUDA engine and new tests. No CUDA toolkit is installed on this macOS arm64 host.
- [ ] Run GPU split, bitwise K10 parity, table lifecycle, and graph replay tests. No NVIDIA GPU is available.
- [ ] Run SAGE forced decode for q=0..6, validate real counts, compare outputs, and measure overlap/latency.
      The RTX 4090 and `/workspace/pack-sage` are not available here.
- [ ] Complete the entire native CPU build. Its x86 expert target requires AVX compiler options unavailable on arm64.
- [ ] Create Git commits. The shared `.git/worktrees/Strata-DS-hybrid` directory is sandbox-protected.
      `HYBRID.COMMITS.md` contains the exact commands and required co-author trailers.
