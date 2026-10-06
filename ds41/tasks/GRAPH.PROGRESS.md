# Decode graph kernel progress

- [x] Read project rules, decode review section 1, and upstream staging code.
- [x] Check branch and preserve existing work. DECODE.STUDY.md was untracked on entry and remains untouched.
- [x] Write the first synthetic parity and capture-once replay tests before implementation.
- [x] Add explicit streams to all 21 legacy ops and add device token/position/row-copy ops.
- [x] Add capacity-based device-length K5 entry points and document the bucket rule.
- [x] Add device-length K3 decode attention with a fixed capacity of 640 indices.
- [x] Add explicit K7 and K8 workspace initialization before capture.
- [x] Add device logits argmax with stable ties and host-compatible NaN handling.
- [x] Check K10 and doorbell source paths for non-default capture streams.
- [x] Add synthetic K10 replay coverage. Retain the existing doorbell replay test.
- [x] Add a non-default stream capture test for every legacy op.
- [x] Run available local checks. Eight portable C++ targets built and their tests passed.
- [x] Attempt the full native build. Record its ARM64/x86 intrinsic failure.
- [x] Record exact Linux build/test commands and real local output in GRAPH.REPORT.md.
- [x] Attempt a test-first commit. Record the sandbox failure and <=5-file units in GRAPH.COMMITS.md.
- [x] Audit every item and report code, compile, and GPU test states separately.

Pending acceptance, with explicit blockers:

- [ ] Compile changed CUDA files and new tests. This macOS ARM64 host has no nvcc or CUDA toolkit.
- [ ] Run bitwise parity, capture replay, and sanitizer tests on a GPU. This host has no CUDA GPU.
- [ ] Measure K5 large-context selection latency. The new path uses one scratch-free selection CTA;
      no performance result is available without a GPU.
- [ ] Create commits. The sandbox denies writes to the worktree index in the parent repository.
- [ ] Integrate and verify Engine::step() capture. This is assigned to the reviewer; engine.cu was not changed.

No pending item above blocks further authorized local work. All local implementation, source review,
portable checks, and handoff documentation are complete. CUDA and end-to-end success are not claimed.
