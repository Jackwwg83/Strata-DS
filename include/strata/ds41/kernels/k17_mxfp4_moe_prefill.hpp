// include/strata/ds41/kernels/k17_mxfp4_moe_prefill.hpp - task K17: routed experts for a prefill chunk (many
// tokens), rows grouped by expert, from DeepSeek's FP4 (MXFP4) weights. K12's interface, K16's math.
// Implementation in src/ds41/kernels/k17_mxfp4_moe_prefill.cu.
#pragma once

#include "strata/ds41/kernels/k16_mxfp4_moe.hpp"

namespace strata::ds41::kernels {

/// Workspace bytes that mxfp4_moe_prefill needs for calls with at most max_rows rows
size_t mxfp4_moe_prefill_workspace_bytes(int max_rows, int max_groups);

/// The caller sorts the (token, slot) assignments of a chunk by expert (K12's contract: tok, w, off on the host,
/// experts with descriptors as K16 reads them). For every row r of group g, e = experts[g], x_t = x[tok[r]]:
///   K16's math: g = bf16(W1 x_t), u = bf16(W3 x_t), the clamps, h = bf16(silu(g) * u * w[r]), FP8 block-32 QDQ,
///   out[tok[r]] += bf16(W2 h)
/// Each expert is dequantized to BF16 once (exact: an E2M1 code times a power of 2), then cuBLAS multiplies in FP32;
/// the FP8 activations are exact in BF16 too, so only the order of the additions differs from fp4_gemm.
/// One-time setup per device at the first call (the cuBLAS handle). After that no allocation, no host sync.
void mxfp4_moe_prefill(const __half* x, const int32_t* tok, const float* w, const int32_t* off, int n_groups,
                       const Exl3Expert* experts, float* out, void* workspace, size_t workspace_bytes,
                       cudaStream_t stream);

}  // namespace strata::ds41::kernels
