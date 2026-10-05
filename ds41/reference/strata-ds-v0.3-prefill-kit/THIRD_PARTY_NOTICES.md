# Third-party boundary

This design kit contains original offline tools and design contracts, not copied upstream runtime source or binaries. Model identifiers, byte counts, source revisions and small factual descriptions are attributed in docs/SOURCES.md.

The package's MIT license does not relicense any model, dependency, external runtime or derived code a future implementation adds. In particular, the referenced coolbho3k/MiaAI Engram reader and integration include AGPL-3.0-only material. Preserve their notices, corresponding source and applicable obligations if incorporating it. Do not assume an IPC boundary automatically resolves obligations. Review the resulting distribution before public release.

The original v0.1 release remains separate. No user private repository contents, credentials or model weights are included.


## v0.3 source audit
The new original prefill planner, format auditor and CPU scheduler describe the public DeepSeek V4.1 / EXL3 MUL1 interfaces. Consult specs/prefill/sources.lock.json and docs/PREFILL_SOURCES.md for exact source identities. No upstream GPU binary or model weights are bundled. The optional GPU probe uses a user's separately installed trusted ExLlamaV3; its dependencies and any reused/modified upstream runtime remain subject to their own licenses. Our MIT license does not relabel upstream AGPL implementation.
