# Current status: v0.3 prefill kit / 2026-10-04

Implemented: exact-schema source arithmetic; local multi-shard component header/MUL1 audit; concrete TP1 phase/state/host/device planner and seed enumeration; route-exact stable wave/row-tile CPU execution reference; identity-matched measured-record selector; optional real-weight GPU component probe script.

Executed this turn: Python old+new unit tests, plan/CLI generation, hash validation, GPU-probe syntax/help (NO GPU execution), C++17 legacy interface syntax. Read `results/prefill/validation-summary.json` for actual count/environment.

Not executed: complete HF checkpoint header/payload scan (direct source access failed), real weights/GPU numerical or peak measurements, full V4.1 forward, actual model-performance campaign, cloud rental, GitHub publication. Source-derived expert sizes are deliberately not labeled payload-audited. All whole-engine backend implementations remain disabled.

Full GPU kernel/attention/I/O implementation tasks are PF-000..005 plus retained SD/U39 tasks. New CPU code is an implementation artifact, not a claim of missing kernel completion.

---
## Historical v0.2.1/v0.2 status (not current test counts)

# v0.2.1 design update status / 2026-10-04

New deliverables: v0.1.39 source-pin review, 12 planned delta tasks, 8 runtime design profiles, phase/ring/IO arithmetic and plan validators. None is a CUDA engine, HTTP server, scheduler, kernel or cloud controller implementation. Real backends remain disabled. New validation evidence is under `results/upgrade0139/`; root results below are v0.2 historical evidence. Existing 11 recipe files and model pins were preserved.

The previous status follows for context:

# Implementation status / 2026-10-04

This is a design-and-offline-tooling delivery, not a real inference engine release.

## Implemented and locally tested

- 11 recipe files (8 core + 3 optional) with immutable JSON fingerprints.
- Conservative design budget calculator. It always reports `can_launch=false`.
- Separate conditional RAM/GPU hit-rate I/O model, including H2D even on all-RAM hits.
- Bounded safetensors header audit; no tensor payload allocation.
- Synchronous buffered `pread` row reader with a payload-bounded LRU, deduplication and fail-closed bounds.
- Thread-safe cache-control ledger with reservations, leases and generation fencing.
- Planned benchmark matrix generation, not execution.
- Token-event summary with per-trial wall-time goodput; synthetic provenance preserved.
- Read-only local host/cgroup observation and offline Vast search/cost proposals.
- Python Protocol and C++ expert-backend contracts.

## Explicitly not implemented / not tested

- EXL3 MUL1 native CUDA/CPU kernels or V4.1 single-discrete-GPU bridge.
- Real GGUF runtime bridge, real model tensor partitioning or whole-model execution.
- Production O_DIRECT/io_uring reader, genuine CUDA events, DMA overlap or hard RSS enforcement.
- Native FP8/E8M0 Engram math hookup or TP2 page15-to-TP1 asset transformation.
- Production continuous batching, HTTP serving, model-state snapshots or remote lifecycle controller.
- Any cloud instance creation, model download, model-quality experiment or GPU throughput measurement.

`contracts.BACKENDS` deliberately blocks every real backend. The CPU state ledger uses a test-level completion signal, NOT a CUDA event. Row-reader `pread_bytes` is not physical SSD traffic. All sample timing and price data are labeled synthetic.

The prior v0.1 package's 93-passed/2-skipped result is not reused as evidence for v0.2. New execution evidence lives in results/unittest.txt and results/validation-summary.json.
