// include/strata/ds41/kernels/k16_mxfp4_moe.hpp - task K16: routed experts on the GPU from DeepSeek's own FP4 (MXFP4)
// weights, as DeepSeek-V4.1-Flash-1017 ships them (pack expert_format mxfp4). Implementation in
// src/ds41/kernels/k16_mxfp4_moe.cu.
#pragma once

#include "strata/ds41/kernels/k10_exl3_moe.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace strata::ds41::kernels {

/// An MXFP4 expert travels in the descriptor the tiers already pass around (the ExpertSlot places of pack.hpp). Per
/// projection: p.trellis = the weight [n][k/2] bytes (E2M1 codes, two per byte along k, the low nibble first) and
/// p.suh = the scale [n][k/32] E8M0 bytes; k and n as for EXL3. p.svh and p.tile_w are not read.
/// Weight value = E2M1(code) * 2^(scale - 127).
__host__ __device__ inline const uint8_t* mxfp4_codes(const Exl3Proj& p) {
    return reinterpret_cast<const uint8_t*>(p.trellis);
}
__host__ __device__ inline const uint8_t* mxfp4_scales(const Exl3Proj& p) {
    return reinterpret_cast<const uint8_t*>(p.suh);
}

/// Workspace bytes for m * topk slots (16-byte aligned memory)
size_t mxfp4_moe_workspace_bytes(int slots);

/// DeepSeek's Expert.forward (inference/model.py, fp4_gemm) for m tokens (1..8), topk slots each. For every token t
/// and slot j with sel[t][j] >= 0, e = experts[sel[t][j]]:
///   g = bf16(W1 x_t), u = bf16(W3 x_t)   (per 32-block: the products summed in FP32, times the block's scale;
///                                          the blocks summed in FP32)
///   g = min(g, 10); u = clamp(u, -10, 10); h = bf16(silu(g) * u * w[t][j])
///   h = FP8 E4M3 block-32 quantize-dequantize with a power-of-2 scale (as K10 and ops::act_quant_inplace)
///   y = bf16(W2 h); out[t] += y, slot after slot (FP32)
/// x [m][5120] fp16 (already FP8-quantized by the caller), sel [m][topk] int32, w [m][topk] f32, experts: device array.
/// out [m][5120] f32 is accumulated into. No allocation, no synchronization (CUDA-graph capturable).
void mxfp4_moe_decode(const __half* x, int m, const int32_t* sel, const float* w, int topk, const Exl3Expert* experts,
                      float* out, void* workspace, size_t workspace_bytes, cudaStream_t stream);

}  // namespace strata::ds41::kernels
