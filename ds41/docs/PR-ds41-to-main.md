# DeepSeek V4.1 Flash on Strata: engine, prefill, decode (feature/ds41 → main)

Draft PR description. Not opened yet: the owner decides.

## What this branch adds

- A DeepSeek V4.1 Flash engine (`src/ds41/`): EXL3 experts (3bpw and SAGE 1.59bpw with mixed K1..K6 per projection),
  mHC, compressed sparse attention with the indexer, Engram, the MoE with VRAM / RAM / file tiers.
- Prefill: layer-major batched prefill; every expert streamed once per pass (upstream's Stager / issuer); upstream's
  resident budget and O_DIRECT file tier.
- Decode: one CUDA graph per step; hybrid experts (approved deviation from upstream): part of each layer's VRAM misses
  is staged into VRAM over PCIe and computed by the GPU while the CPU computes the rest; wo_a kept as FP8.
- Speculative decoding infrastructure: a multi-token verifier (T <= 4, transactional window ring, compressor,
  compressed KV and Engram history) and a suffix drafter (`--spec suffix`, off by default: +5% on repetitive text,
  -10% on other text while decode is PCIe bound). 12 of 12 acceptance tests pass; speculative output equals plain
  greedy output token for token with fixed residency.
- Tools: `tools/ds41/pack.py` (incl. `--dense-only`), quality and speed scripts in `ds41/bench/scripts/`.

## Measured (RTX 4090, EPYC 7742, container held at 119.9 GiB, SAGE 1.59bpw)

| | 32K | 64K | 128K | 256K |
| --- | --- | --- | --- | --- |
| Prefill tok/s | 1,100 | 1,159 | 1,263 | 967 |

Decode, 16 threads: 40.3 ms/token at 72% VRAM hits (code text), 53-77 ms at 52% (repository text). Before the decode
work: 62.2 ms (16 threads), 81.4 ms (8 threads). Report: `ds41/docs/m3-report-v3.html`.

Quality (5 documents, teacher-forced nll): SAGE 1.59 2.509 (engine) / 2.513 (FP16 prototype); 3bpw 2.228.

## Tests

Unit and GPU tests per component (pack, tiers, mixed-K kernels, graph kernels, hybrid staging, doorbell, verifier
rows parity); bit-exactness checks for every decode change with fixed residency (nll 1.216695 / 1.216216 unchanged).

## Not in this PR / open

- Installer and server support for DeepSeek V4.1 (setup.py, serve/) are not done.
- DSpark (DeepSeek's MTP drafter, 7.9 GB) is not wired in: its placement is an owner decision.
- 3bpw not re-measured with the new decode path (the 3bpw box's host is offline at Vast).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
