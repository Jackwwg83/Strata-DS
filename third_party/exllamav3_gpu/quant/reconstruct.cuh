#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace strata_exl3 {

// packed_ptr is a DEVICE pointer to one trellis pointer. Output is row-major
// [k][n] fp16 in the Hadamard basis, without suh/svh or either Hadamard.
// k must be divisible by 16, n by 128; the trellis uses 3-bit mul1 (tile_w=48).
void reconstruct_mul1_3bit(half* unpacked, const uint16_t* const* packed_ptr,
                           int k, int n, cudaStream_t stream);

}  // namespace strata_exl3
