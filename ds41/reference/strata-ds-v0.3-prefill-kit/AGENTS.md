# v0.3 prefill priority addendum

Read `docs/DEEPSEEK_PREFILL.zh-CN.md` FIRST for prefill. Source schema H5120/F2304/E384/K6; exact TP1 MUL1 payload13,315,596B, component-aligned slot13,316,352B. Verify real checkpoint via the supplied auditor, never assume the entire Hub payload was audited in this delivery. Explicit T/G/R and query/key slabs replace Qwen constants.

Reuse every selected expert for all its row tiles. No whole-bank reconstruction, no implicit missing-expert skip, no unqualified CED decoder truncation. Source bytes, designed buffer bytes, reserved allowances and measured peaks must remain distinguishable. A passed CPU planner cannot qualify a GPU backend. `can_launch` remains false until a real engine exists, not because another abstract plan is needed.

# v0.2.1 priority addendum

Read `docs/UPSTREAM_v0.1.39_DELTA.zh-CN.md` and `specs/upstream-v0139.backlog.json` before the prior contract below. Merge prerequisite augmentations with the original SD task DAG. Keep 11 hardware recipes and EXL3-first strategy. Source pin: Strata v0.1.39 at 6f32ec070f23ced9f50e704d854d775da52591ab; no blind PR cherry-picks.

Mandatory new invariants: unknown phase bytes block; required dense/head/logits before cache; post-residency file working set, not entire shards; ring bytes from actual packed bundle stride; allocated slots distinct from offered load; unsupported sampling/tool/schema features rejected, not silently ignored; no Base64 mislabeled as encryption. Default one slot and no speculation. Exactness claims must name kernel/placement/sampling conditions. No shared-host sysctl/drop_caches. New profiles are plans, not executable backend configs.

# Coding-agent contract

## Goal
Build and compare DeepSeek V4.1 Flash recipes on 16/24 GiB discrete NVIDIA GPU with 128/192/256/384 GiB host RAM. coolbho3k EXL3 3bpw is QUALITY PRIMARY. Q2 is a cost/quality comparison, not an automatic replacement. Engram belongs on SSD with bounded row caching.

## First reads
README.md -> docs/OVERALL_DESIGN.zh-CN.md -> docs/CODE_DESIGN.zh-CN.md -> docs/ACCEPTANCE.zh-CN.md -> docs/VAST_RUNBOOK.zh-CN.md -> specs/backlog.json.

## Mandatory invariants
- Never silently change base model, quant tensors, tokenizer/template, router top-k, expert count or KV quantization.
- Never equate file size, active payload, allocated RAM, physical RAM and usable device memory.
- Never preload/clone a full Engram table. Do not remove OOM guards instead of fixing the loader.
- No full expert pool dequantization. A fallback must use bounded tiles and a distinct backend label.
- No CPU EXL3 performance assumption. The initial EXL3 executor is GPU packed streaming.
- Every resident and in-flight allocation is budgeted. Don't evict leased slots; fence stale completions.
- Resident-only parity and SSD offload parity are separate from quantization quality.
- A synthetic test is never a benchmark; SSE chunks are never counted as tokens.
- APP_CACHE_CAP_ONLY / CGROUP_CAP / PHYSICAL_HARDWARE are distinct evidence labels.
- Do not treat a TP2/GB10 upstream image as a single-card x86 runtime.
- Preserve source and model licenses. No AGPL code copied into a purported MIT-only release.

## Cloud permission
This delivery authorizes no rental. Read-only search is separate from paid writes/downloads. Require local approved budget, exact offer/image/recipe and a deadline before spending. Never send keys to ChatGPT or commit them. No unattended full matrix, no indefinite retry, no destroy-all. Stop does not necessarily end disk charges. Save logs then verify provider destruction.

## Development ordering
Follow L0-L9 in the code design; implement backlog tasks in dependency order. CPU tests first, then representative real GPU expert, then whole-model oracle, then memory/performance. Produce evidence for each stage. Do not flip BACKENDS.implemented merely to make tests green.

## Reporting
Each change: changed files, tests executed, skips, real-model/GPU status, limits, rollback, next blocked dependency. Keep unsupported facts as unknown. Do not promise background completion.
