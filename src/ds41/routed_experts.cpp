// src/ds41/routed_experts.cpp - see include/strata/ds41/routed_experts.hpp.
#include "strata/ds41/routed_experts.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/kernels/k12_exl3_moe_prefill.hpp"
#include "strata/ds41/kernels/k16_mxfp4_moe.hpp"
#include "strata/ds41/kernels/k17_mxfp4_moe_prefill.hpp"
#include "strata/ds41/mxfp4_cpu.hpp"

#include "moe_mul1.h"   // third_party/exllamav3_moe

#include <stdexcept>
#include <vector>

namespace strata::ds41 {
namespace {

/// exllamav3's descriptor of one EXL3 projection (c0 = 0 gate, 4 up, 8 down) of the expert at `bytes`
MoeCpuMatrixDesc exl3_desc(const ExpertSlot& s, const uint8_t* bytes, int c0, int k, int n) {
    MoeCpuMatrixDesc d;
    d.trellis = (const uint16_t*) (bytes + s.comp_off[c0]);
    d.suh = (const at::Half*) (bytes + s.comp_off[c0 + 1]);
    d.svh = (const at::Half*) (bytes + s.comp_off[c0 + 2]);
    d.k_tiles = k / 16;
    d.n_tiles = n / 16;
    d.tile_w = (int) (s.comp_bytes[c0] / ((uint64_t) d.k_tiles * d.n_tiles * 2));
    return d;
}

/// the MXFP4 expert at `bytes`: each projection's codes in place c0, its scales in place c0 + 1 (pack.hpp)
Mxfp4CpuExpert fp4_desc(const ExpertSlot& s, const uint8_t* bytes) {
    auto proj = [&](int c0) { return Mxfp4CpuProj{bytes + s.comp_off[c0], bytes + s.comp_off[c0 + 1]}; };
    return {proj(0), proj(4), proj(8)};
}

}  // namespace

void routed_moe_decode(ExpertFormat f, const __half* x, int m, const int32_t* sel, const float* w, int topk,
                       const kernels::Exl3Expert* experts, float* out, void* workspace, size_t workspace_bytes,
                       cudaStream_t stream) {
    if (f == ExpertFormat::Mxfp4)
        kernels::mxfp4_moe_decode(x, m, sel, w, topk, experts, out, workspace, workspace_bytes, stream);
    else
        kernels::exl3_moe_decode(x, m, sel, w, topk, experts, out, workspace, workspace_bytes, stream);
}

size_t routed_moe_prefill_workspace_bytes(ExpertFormat f, int max_rows, int max_groups) {
    return f == ExpertFormat::Mxfp4 ? kernels::mxfp4_moe_prefill_workspace_bytes(max_rows, max_groups)
                                    : kernels::exl3_moe_prefill_workspace_bytes(max_rows, max_groups);
}

void routed_moe_prefill(ExpertFormat f, const __half* x, const int32_t* tok, const float* w, const int32_t* off,
                        int n_groups, const kernels::Exl3Expert* experts, float* out, void* workspace,
                        size_t workspace_bytes, cudaStream_t stream) {
    if (f == ExpertFormat::Mxfp4)
        kernels::mxfp4_moe_prefill(x, tok, w, off, n_groups, experts, out, workspace, workspace_bytes, stream);
    else
        kernels::exl3_moe_prefill(x, tok, w, off, n_groups, experts, out, workspace, workspace_bytes, stream);
}

int64_t cpu_make_layer(ExpertFormat f, const ExpertSlot* const* slots, const uint8_t* const* bytes, int n,
                       float swiglu_limit) {
    if (f == ExpertFormat::Mxfp4) {
        std::vector<Mxfp4CpuExpert> e(n);
        for (int i = 0; i < n; ++i) e[i] = fp4_desc(*slots[i], bytes[i]);
        return mxfp4_cpu_make_layer(e.data(), n);
    }
    std::vector<MoeCpuMatrixDesc> gate(n), up(n), down(n);
    for (int i = 0; i < n; ++i) {
        gate[i] = exl3_desc(*slots[i], bytes[i], 0, kDim, kMoeInter);   // w1: 5120 -> 2304
        up[i] = exl3_desc(*slots[i], bytes[i], 4, kDim, kMoeInter);     // w3
        down[i] = exl3_desc(*slots[i], bytes[i], 8, kMoeInter, kDim);   // w2: 2304 -> 5120
    }
    return exl3_moe_cpu_make_layer_raw(gate.data(), up.data(), down.data(), n, 0, swiglu_limit, 0);
}

void cpu_free_layer(ExpertFormat f, int64_t handle) {
    if (f == ExpertFormat::Mxfp4) mxfp4_cpu_free_layer(handle);
    else exl3_moe_cpu_free_layer(handle);
}

void cpu_set_expert(ExpertFormat f, int64_t handle, int e, const ExpertSlot& slot, const uint8_t* bytes) {
    if (f == ExpertFormat::Mxfp4) {
        mxfp4_cpu_set_expert(handle, e, fp4_desc(slot, bytes));
        return;
    }
    const MoeCpuMatrixDesc g = exl3_desc(slot, bytes, 0, kDim, kMoeInter);
    const MoeCpuMatrixDesc u = exl3_desc(slot, bytes, 4, kDim, kMoeInter);
    const MoeCpuMatrixDesc d = exl3_desc(slot, bytes, 8, kMoeInter, kDim);
    exl3_moe_cpu_set_expert_raw(handle, e, &g, &u, &d, 0);
}

void cpu_forward(ExpertFormat f, int64_t handle, const uint16_t* x, const int32_t* ids, const float* w, float* out,
                 int rows, int topk, int threads) {
    if (f == ExpertFormat::Mxfp4) {
        mxfp4_cpu_forward(handle, x, ids, w, out, rows, topk, threads);
        return;
    }
    constexpr int kMax = 64 * 32;
    if (rows * topk > kMax) throw std::invalid_argument("cpu_forward: too many slots");
    c10::Half wh[kMax];
    for (int i = 0; i < rows * topk; ++i) {
        const __half hv = __float2half_rn(w[i]);
        wh[i] = c10::Half(__half_as_ushort(hv), c10::Half::from_bits());
    }
    exl3_moe_cpu_forward_raw(handle, (const at::Half*) x, ids, wh, out, rows, topk, threads);
}

void cpu_pool_prime(ExpertFormat f, int threads) {
    if (f == ExpertFormat::Exl3) exl3_moe_cpu_pool_prime(threads);
}

}  // namespace strata::ds41
