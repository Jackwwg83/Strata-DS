// include/strata/ds41/kernels/k18_exl3_linear.hpp - task K18: one dense projection from EXL3 (mul1) weights, as
// exllamav3's LinearEXL3 computes it, for the dense layers of an EXL3 checkpoint (attention, shared expert, head).
// Implementation in src/ds41/kernels/k18_exl3_linear.cu.
#pragma once

#include "strata/ds41/kernels/k10_exl3_moe.hpp"

#include <cuda_bf16.h>

namespace strata::ds41::kernels {

/// Rows up to this many use exllamav3's small-row GEMV (one job per row); more rows reconstruct the matrix and
/// multiply with cuBLAS (K12's prefill path).
constexpr int kExl3GemvRows = 8;

/// Workspace bytes for exl3_linear with m rows of a k -> n projection
size_t exl3_linear_workspace_bytes(int m, int k, int n);

/// y[r][0..n) = W x[r] for r < m, with W the EXL3 projection w (k -> n; k, n multiples of 128):
///   a = H(x[r] * suh) in fp16, c = W_q a in FP32, y = bf16(H(c) * svh), H the 128-block Hadamard / sqrt(128).
/// x [m][ldx] bf16, y [m][ldy] bf16. No allocation, no host synchronization; capturable when m <= kExl3GemvRows
/// (the cuBLAS path needs its handle created before capture, as K12).
void exl3_linear(const __nv_bfloat16* x, int ldx, int m, const Exl3Proj& w, __nv_bfloat16* y, int ldy,
                 void* workspace, size_t workspace_bytes, cudaStream_t stream);

/// The projection's matrix as BF16, out [n][k] (row j: W's row for output j), so that out x = y as exl3_linear computes
/// it, up to BF16 rounding. Through the reconstruct path, so K1..K8 (the GEMV path takes K1..K6). Allocates its own
/// scratch and synchronizes the stream: for loading (an 8-bit head becomes a BF16 head).
void exl3_dequant_bf16(const Exl3Proj& w, __nv_bfloat16* out, cudaStream_t stream);

}  // namespace strata::ds41::kernels
