# Hybrid decode staging progress

Checked items mean the work described is done. They do not imply CUDA compilation or GPU acceptance.
See [HYBRID.REPORT.md](HYBRID.REPORT.md) for separate states and real local output.

- [x] Confirm branch `feature/ds41-hybrid`, merged stream/graph code, and reviewer baseline.
- [x] Preserve the pre-existing untracked `DECODE.STUDY.md`.
- [x] Write copy byte tests and expand K10 parity tests before production changes.
- [x] Allocate fixed staging storage from the pack's largest expert before automatic cache sizing.
- [x] Publish compact device copy jobs and rebase only RAM-assigned call descriptors.
- [x] Add a fixed-grid uint4 copy with exact byte tails.
- [x] Fork a copy stream, compute the shared expert on the main stream, then join before one K10 call.
- [x] Keep `DS41_ZC_QUOTA`; add default-on staging and `DS41_ZC_STAGE=0` direct comparison.
- [x] Cover q0..6, m1..8, mixed blob sizes, nonzero component offsets, changed graph routes, and repeated forks.
- [x] Poison staging before parity replays and check copy counts, jobs, descriptors, and output bits.
- [x] Verify source invariants: K10 arithmetic and `enqueue_step()` / `step()` remain unchanged.
- [x] Build all eight portable regression targets and run all eight tests successfully.
- [x] Compile the generator object with strict warnings; check shell and embedded Python syntax.
- [x] Attempt CUDA configuration and full CPU build; record the actual failures.
- [x] Supply exact build, sanitizer, fixed-slot 42-case sweep, dump parity, and nsys commands.
- [x] Update the report and commit units. Keep prior reviewer measurements separate from new results.

Not done in this environment:

- [ ] Compile the new CUDA source and tests for sm_89 / sm_86 / sm_120. No CUDA toolkit is installed.
- [ ] Run new and existing DS41 GPU acceptance tests and sanitizer. No NVIDIA GPU is available.
- [ ] Run SAGE forced decode, compare direct/staged dumps, and validate graph replay in the full engine.
- [ ] Measure copy bandwidth, overlap, and ms/token at 8 / 16 / 30 threads and q0..6.
- [ ] Exercise adaptive swaps and no-RAM fallback in real decode with staging enabled.
- [ ] Create follow-up commits. Git metadata is deliberately outside the sandbox's writable roots.
