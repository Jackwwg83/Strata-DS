# Verifier progress

Every item below was reviewed before delivery. Source completion is separate from GPU acceptance.

- [x] Read task rule 7, study section 3 / plan 4.3, decode hooks, and upstream verifier / suffix / policy / generation.
- [x] Check worktree and branch. Preserve the owner's untracked DECODE.STUDY.md.
- [x] Write suffix and six pack acceptance test sources before the implementation.
- [x] Implement tentative verifier state and prefix commit in a new verify.cu file.
- [x] Batch dense, HC, router, hybrid GPU / CPU experts per layer. Cap windows at four rows.
- [x] Capture by T, parity, and capacities. Keep position parameters on the device. Use an eager first window.
- [x] Port suffix lookup and integrate generation, acceptance, EOS, output limits, and timing.
- [x] Build and run relevant local CPU checks. Record actual output in VERIFY.REPORT.md.
- [x] Compile the generation and pack-test C++ objects with warnings as errors.
- [x] Run synthetic CLI tests with full, partial, and rejected proposals, output bounds, EOS, and prefill.
- [x] Review ring rollback, compressor snapshots, causal index lists, shared sources, candidate masks, and history.
- [x] Keep moe() and worker_loop() unchanged. A source comparison against HEAD confirmed both bodies are identical.
- [x] Prepare at-most-five-file commit units and the required Co-Authored-By trailer in VERIFY.COMMITS.md.
- [ ] Create commits. Git staging cannot create the protected main repository's worktree index.lock.
- [x] Write exact RTX 4090 build/test and real CLI comparison commands in VERIFY.REPORT.md.
- [ ] Compile CUDA and link the real Engine executables. This macOS arm64 host has no nvcc.
- [ ] Run GPU parity, prefix, wrap, compressor, Engram, graph replay, and real suffix acceptance.
      This host has no CUDA GPU or SAGE pack. The report lists each command and its required result.
- [ ] Measure real acceptance rate, GPU memory, and tokens/s. These require the real GPU runs above.
- [ ] Complete AddressSanitizer execution. The combined ASan/UBSan run hung without output and was interrupted.
      UBSan alone passed. No passing ASan result is claimed.

Scope limits are explicit: T=5..8 is capped until the CPU worktree fix is validated. Commit uses eager capturable
GPU operations, not a cached commit graph. DSpark is excluded. No model-backed validation or speedup is claimed.
