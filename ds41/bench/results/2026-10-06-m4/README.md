# M4 results (RAM tier, file tier, lookahead), 2026-10-06

Machine: Vast instance 54324174. RTX 4090 24 GB, Intel Core i9-14900K (8 P-cores + 16 E-cores, AVX2, no AVX-512),
container limits 64 GiB RAM (the file cache counts against it) and about 30 CPUs. Pack: EXL3 3bpw, experts 190.5 GiB
in `experts.bin`. Engine: feature/ds41 at e7941c3 plus the measured flags. 854 VRAM expert slots (10.6 GiB).
**This is a 64 GiB machine with an AVX2 CPU, not the 128 GB / AVX-512 target.**

## Correctness

`m4_eval.log`: with the VRAM split fixed (`--adapt-every 0`), the per-step dumps with no RAM tier and with a 60 GiB RAM
tier are byte-identical (code_py_0, 200 tokens, nll 1.063326). Moving an expert between tiers changes no output.

## RAM budget (m4_sweep.sh, cold file cache before each run, adaptive VRAM tier and lookahead on, 16 threads)

| RAM tier | documents ms/token | chat ms/token | RAM share | file share | SSD share (doc / chat) |
| --- | --- | --- | --- | --- | --- |
| 0 GiB | **294.6** | 419.0 | 0 | 0.34 / 0.47 | 0.094 / 0.206 |
| 16 GiB | 323.1 | **396.6** | 0.06 / 0.07 | 0.28 / 0.40 | 0.105 / 0.196 |
| 32 GiB | 319.6 | 471.3 | 0.12 / 0.13 | 0.24 / 0.35 | 0.111 / 0.197 |
| 48 GiB | 337.9 | 516.4 | 0.17 / 0.21 | 0.19 / 0.29 | 0.111 / 0.186 |

Documents: code_py_0, 200 teacher-forced tokens. Chat: the first 48 tokens of zh_0, then 128 generated tokens.
A static, profile-ranked RAM tier serves few accesses (the VRAM tier holds the hottest experts) and takes room from
the file cache, which keeps what the current text uses. The SSD share hardly changes with the budget.
Before the sweep, `m4_eval.log` measured an automatic budget (60 GiB of 64): 552 ms/token against 357 without a RAM
tier (adaptive VRAM tier off in that pair). So the engine's default is now no RAM tier, as upstream's (setup opts in).

## Fetch-now and lookahead (m4_fetch.sh, chat, RAM tier 0)

| mode | ms/token |
| --- | --- |
| both on | 417.4 |
| both off | 431.4 |
| fetch-now only | 425.0 |
| lookahead only | 457.1 |

The differences are within run-to-run noise. Lookahead: 48% of the experts it warmed were used by the next layer.

## Where the time goes

CPU expert time is 90% of the step. In the last steps of a document, 157 CPU experts with 18 read from the SSD took
237 ms: about 1.5 ms per expert, about 9 GB/s, against 44 GB/s measured for the same kernel on a 7950X (AVX-512).
CPU threads (documents, RAM tier 0, VRAM split fixed): 8 threads (the P-cores) 337.5, 16: 374.7, 28: 359.5 ms/token.
On this CPU the AVX2 path of the CPU expert kernel is the bottleneck, then the SSD.
