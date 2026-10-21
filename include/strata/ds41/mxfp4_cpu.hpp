// include/strata/ds41/mxfp4_cpu.hpp - routed MXFP4 experts on the CPU: the experts that the GPU does not compute
// (read from the SSD into RAM during the step), for packs with expert_format mxfp4. The CPU counterpart of K16, with
// its math (kernels/k16_mxfp4_moe.hpp); the layer registry follows exllamav3's moe_mul1 (make, set, forward, free).
// Plain C++: it builds on any CPU, the DGX Spark's ARM cores included.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::ds41 {

/// One projection: codes [n][k/2] (E2M1, two per byte along k, the low nibble first), scales [n][k/32] (E8M0)
struct Mxfp4CpuProj {
    const uint8_t* codes = nullptr;
    const uint8_t* scales = nullptr;
};
/// w1 gate (5120 -> 2304), w3 up (5120 -> 2304), w2 down (2304 -> 5120)
struct Mxfp4CpuExpert {
    Mxfp4CpuProj w1, w3, w2;
};

/// Register a layer of n experts. The caller keeps the bytes alive and unchanged until free or set_expert.
int64_t mxfp4_cpu_make_layer(const Mxfp4CpuExpert* experts, int n);
void mxfp4_cpu_free_layer(int64_t handle);
/// Point one expert at other bytes. Only between forwards of that layer.
void mxfp4_cpu_set_expert(int64_t handle, int expert, const Mxfp4CpuExpert& e);

/// The routed experts of `rows` tokens (1..8): out[t] = sum over the token's slots j with sel[t][j] >= 0 of K16's
/// y = bf16(W2 qdq(bf16(silu(min(g, 10)) * clamp(u, 10) * w[t][j]))), g = bf16(W1 x_t), u = bf16(W3 x_t), added in
/// slot order. x [rows][5120] fp16 bits, sel [rows][topk], w [rows][topk], out [rows][5120] (overwritten).
/// Each expert's weights are read once for all the rows that use it. `threads` (>= 1) includes the caller. One
/// caller at a time: the calls share one thread pool.
void mxfp4_cpu_forward(int64_t handle, const uint16_t* x, const int32_t* sel, const float* w, float* out, int rows,
                       int topk, int threads);

}  // namespace strata::ds41
