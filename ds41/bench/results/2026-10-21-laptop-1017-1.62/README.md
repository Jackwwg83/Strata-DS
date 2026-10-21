# SAGE 1.62 bpw (V4.1 Flash 1017) against SAGE 1.59 bpw (V4.1 Flash) on the RTX 5090 Laptop, 2026-10-21

**Decision:** the laptop recipe stays on SAGE 1.59 bpw. The 1017 pack is as fast but predicts real text worse on
every kind measured; task accuracy was not measured.

**What ran.** The laptop of [docs/DEEPSEEK_LAPTOP.md](../../../../docs/DEEPSEEK_LAPTOP.md) (RTX 5090 Laptop 24 GB,
64 GB RAM). One engine build for both packs (feature/ds41-spark at db2e3eed, which runs the 1.62 pack's 8-bit EXL3
head as a BF16 head). Default settings (route and SSD tail skips on). The models:
`vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw` (eca94a38) and `vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw` (2a605935,
gated; sha256 of every file checked against the repository's). Script: [t162.sh](t162.sh).

## Speed

4 short prompts x 3 rounds, 256 tokens, a fresh `ds41_generate` per run, the pack's page cache dropped first, one
shuffled order. Decode tokens/s, median of 3 ([speed.txt](speed.txt)):

| Pack | agent | code | zh_chat | en_explain | all 12 runs |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAGE 1.59 | 30.8 | 30.4 | 31.9 | 31.0 | 30.8 |
| SAGE 1.62 (1017) | 32.4 | 32.0 | 29.4 | 30.3 | 31.2 |
| SAGE 1.62 (1017), DSpark every step | 28.4 | 28.4 | 26.0 | 27.6 | 27.5 |

DSpark makes the laptop slower: a verify pass reads more experts, and on this laptop those go over PCIe or to the CPU.

## Quality

200 segments of 256 tokens of real code, prose and chat (`kld200.list`, from `ds41/proto/corpus/docs.jsonl`),
teacher-forced; mean NLL per token (lower is better) and the 95% bootstrap interval of the difference
([nll_1.62_vs_1.59.json](nll_1.62_vs_1.59.json)):

| Kind | 1.59 | 1.62 (1017) | Difference |
| --- | ---: | ---: | ---: |
| Python | 1.821 | 1.899 | +0.078 [0.055, 0.100] |
| C++ | 1.936 | 2.024 | +0.089 [0.069, 0.111] |
| Chinese | 3.811 | 4.060 | +0.249 [0.213, 0.284] |
| English | 2.744 | 2.860 | +0.116 [0.093, 0.139] |
| chat | 1.675 | 1.784 | +0.109 [0.086, 0.133] |
| all | 2.397 | 2.525 | +0.128 [0.114, 0.142] |

The two packs differ in both the checkpoint (V4.1 Flash against its 1017 release) and the quantization, and this
measurement cannot separate them. The quantizer's own numbers put 1.62 closer to its reference (KL 0.098 against
the 1017 FP4 weights) than 1.59 to its own (0.189), so the checkpoint is the more likely cause. Not measured.
