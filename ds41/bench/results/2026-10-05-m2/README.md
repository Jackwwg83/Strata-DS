# M2 engine results, 2026-10-05

Machine: Vast instance 54293474. RTX 4090 24 GB, AMD EPYC 7642 (48 cores, but the container quota is 23 CPUs),
503 GB RAM, CUDA 12.8. Pack: EXL3 3bpw (coolbho3k rev 650cae2c), `/workspace/pack-3bpw`. The GPU test queue
(`ds41/ci/runner.sh`) shared the machine; GPU jobs were serialized with `flock gpu.lock`, but its builds (`-j96`)
competed for the CPU quota. **The CPU expert times here do not represent the target PC (7950X class).**

## Files

| File | What |
| --- | --- |
| `m2a_verify.{sh,log}` | M2a (doorbell CPU experts): dumps byte-identical to M1 on code_py_0 and zh_0 |
| `m2b_verify.{sh,log}`, `results/m2b/*.out` | M2b (VRAM tier + adaptive swaps): hit rate, speed, nll |
| `m2b_accuracy.{sh,log}` | exact-FP16-expert prototype nll (5 docs; the chat_0 doc failed: KeyError 'text') |
| `m2c_verify.{sh,log}`, `results/m2c/*.out` | M2c (K1 GEMV, K3/K7 interfaces): nll, route agreement |
| `m2c_eval.{sh,log}` | M2c on 5 docs vs the FP16 reference, no tier and tier; warm-cache timing |
| `k3_engine_check.{sh,log}`, `k5_engine_check.log` | engine after merging K3-10, then K3-11 + K5-11 |
| `prof_kern_cuda_gpu_kern_sum.csv` | nsys kernel summary, 40 steps, tier on (before K3/K5 merges) |
| `ci/state/results/*.tsv` | every GPU-queue result of the day (task branch, pass/fail, timings) |

The `results/m2b_acc/fp16/*.npz` oracle dumps are not in git (`*.npz` is ignored).

## Main numbers

Teacher-forced nll, 200 tokens per doc. Reference: the prototype with exact EXL3 experts (GPU LinearEXL3 with
`EXL3_INT8_GEMV=0`). No tier: every routed expert on the CPU (exllamav3 moe_mul1, int8 activations). Tier: 896 VRAM
slots (11.1 GiB) from `ds41/data/expert-profile.bin` plus adaptive swaps; hits use the GPU FP16 path (K10).

| doc | reference | no tier | tier (hit rate) |
| --- | --- | --- | --- |
| code_py_0 | 1.077 | 1.056 | 1.038 (66.8%) |
| zh_0 | 3.644 | 3.648 | 3.592 (51.0%) |
| en_0 | 1.943 | 1.959 | 1.931 (53.0%) |
| code_cpp_1 | 2.146 | 2.147 | 2.192 (52.0%) |
| zh_2 | 2.326 | 2.381 | 2.416 (59.8%) |
| mean | 2.227 | 2.238 | 2.234 |

Decode, code_py_0 (forced tokens, warm page cache, 8 CPU threads), ms per token:

| engine | no tier | tier |
| --- | --- | --- |
| M1 | 665 | - |
| M2a (doorbell) | 298 | - |
| M2b (VRAM tier) | 279 | 156 |
| M2c (K1 GEMV) | 268-276 | 125-127 |
| + K3-11, K5-11 | - | 126.6 |

Chat decode (zh_mail prompt, 128 generated tokens, M2b): static profile 7.2% hits, 402 ms/token; adaptive 51.3%
hits, 188 ms/token.

GPU kernel time per step (nsys, 40 steps, tier on): about 19 ms of kernels: FP8 GEMV 6.6 ms, hyper-connection
mixes 3.1 ms, wo_a (stored as BF16) 2.9 ms, K10 experts 2.5 ms, BF16 GEMV (mostly the head) 1.8 ms, attention
0.4 ms. The rest of the step is the GPU waiting for the CPU experts.

CPU threads (40 steps, tier): 16 threads 152 ms, 32 threads 178 ms per token. More threads were slower: the container
quota is 23 CPUs and the queue's builds ran at the same time. Not a property of the kernel.
