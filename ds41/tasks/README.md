# ds41 open kernel tasks

Strata-DS ports the Strata engine to DeepSeek V4.1 Flash on one consumer GPU (RTX 4090 / 3090 / 16 GB cards) plus
128 GB RAM. The tasks below are self-contained GPU kernels with a fixed interface and a fixed acceptance test. Many
implementations of the same task may compete; the fastest one that passes wins and is merged.

| Task | Kernel | Header | Ranking metric (lower is better) |
| --- | --- | --- | --- |
| [K1c](K1c.md) | faster FP8 GEMV for decode (all dense weights) | `include/strata/ds41/fp8_gemv.hpp` | `score_us` |
| [K2](K2.md) | FP8 block-scaled GEMM for prefill | `include/strata/ds41/kernels/k2_fp8_gemm.hpp` | `score_us` |
| [K3](K3.md) | sparse attention, decode and verify windows | `include/strata/ds41/kernels/k3_sparse_attn.hpp` | `score_us` |
| [K5](K5.md) | indexer scores, candidate blocks, top-k | `include/strata/ds41/kernels/k5_indexer.hpp` | `score_us` |
| [K7](K7.md) | hyper-connection mix + collapse | `include/strata/ds41/kernels/k7_hc.hpp` | `score_us` |
| [K10](K10.md) | routed experts on the GPU from EXL3 weights (vendor exllamav3) | `include/strata/ds41/kernels/k10_exl3_moe.hpp` | `score_us` |
| [K8](K8.md) | MoE router on the GPU | `include/strata/ds41/kernels/k8_router.hpp` | `score_us` |
| [K11](K11.md) | CPU EXL3 expert kernel (AVX2 first), vendored exllamav3 | `third_party/exllamav3_moe/moe_mul1.h` | `score_us` |

## Rules for every task

1. Work on a branch named `task/<TASK>/<your-name>` (for example `task/K3/dots-07`), created from
   `feature/ds41`. Never push to `main` or `feature/ds41`.
2. Change only the files the task lists under CI-FILES. The interface header and the acceptance test are fixed;
   a branch that changes them fails automatically.
3. Every push is built and tested on an RTX 4090 (CUDA 12.8, sm_89) by the queue in `ds41/ci/runner.sh`, one branch
   at a time. The result (pass/fail, timings, log tail) is posted as a comment on the task's GitHub issue, usually
   within 10-20 minutes. Read it and iterate.
4. The test prints `RESULT pass|fail key=value ...`. Pass means numerical parity with the reference. Ranking uses
   `score_us` among passing results.
5. Must also compile for sm_86 (RTX 3090) and sm_120 (RTX 50): do not require FP8 tensor cores or features newer
   than sm_86 without a fallback. Shared memory per block: at most 99 KB (consumer GPUs). C++17, CUDA 12.8, no new
   dependencies.
6. Do not change the numerics to pass. The reference functions in `src/ds41/ops.cu` define the math; they restate
   DeepSeek's `ds41/proto/ref/model.py`.
7. The engine captures each decode step as one CUDA graph on a non-default stream (as upstream Strata does). So a call
   must not synchronize with the host (no `cudaMemcpy` to or from host memory, no `cudaStreamSynchronize`, no
   reading device results on the host) and must not allocate or free memory (no `cudaMalloc`, `cudaMallocAsync`,
   `cudaMallocFromPoolAsync`, `cudaFree*`). Scratch memory: allocate it once, on the first call, keep it for the
   life of the process (one buffer per device, sized for the largest call the interface allows), and reuse it.
   The engine always makes one eager call before it captures. Use only the stream argument.

Machine-read fields for the CI runner are at the top of each task file.
