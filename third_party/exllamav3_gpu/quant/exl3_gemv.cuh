#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace strata_exl3 {

// One row, already input-Hadamard transformed. C is FP32 before output Hadamard.
// Stored on device; B == nullptr skips a slot without touching A or C.
struct GemvJob {
    const half* A;
    const uint16_t* B;
    float* C;
    int k;
    int n;
};

// K10: 3-bit mul1, narrow configuration, one row per job. max_n sizes grid.x;
// each job's n must be a multiple of 128 and no larger than max_n.
void gemv_mul1_3bit(const GemvJob* jobs, int count, int max_n, cudaStream_t stream);

}  // namespace strata_exl3
