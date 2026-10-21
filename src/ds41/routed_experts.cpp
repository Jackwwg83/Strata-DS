// src/ds41/routed_experts.cpp - see include/strata/ds41/routed_experts.hpp.
#include "strata/ds41/routed_experts.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/kernels/k12_exl3_moe_prefill.hpp"
#include "strata/ds41/kernels/k16_mxfp4_moe.hpp"
#include "strata/ds41/kernels/k17_mxfp4_moe_prefill.hpp"
#include "strata/ds41/kernels/k19_exl3_smoe.hpp"
#include "strata/ds41/mxfp4_cpu.hpp"

#include "moe_mul1.h"   // third_party/exllamav3_moe

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

/// exllamav3's CPU kernel (moe_mul1) is x86 only. Elsewhere (the DGX Spark's ARM CPU) an EXL3 pack has no CPU
/// experts: on unified memory the GPU computes every expert it can reach, and a use left to the CPU (a miss with no
/// free RAM slot) adds nothing; the first such call says so, with a count at exit.
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
constexpr bool kExl3Cpu = true;
#else
constexpr bool kExl3Cpu = false;
#endif
std::atomic<long long> dropped{0};
struct DropReport {
    ~DropReport() {
        if (dropped.load())
            std::fprintf(stderr, "ds41: %lld EXL3 expert uses left to the CPU were dropped (no CPU kernel here)\n",
                         dropped.load());
    }
} drop_report;

/// K19 (sudoingX's streaming kernels) for EXL3 experts on an integrated GPU (GB10: 199-224 GB/s, K10 159-181);
/// DS41_EXL3_K19=0 / 1 forces K10 / K19
bool use_k19() {
    static const bool on = [] {
        if (const char* v = std::getenv("DS41_EXL3_K19")) return v[0] != '0';
        int dev = 0, integrated = 0;
        return cudaGetDevice(&dev) == cudaSuccess &&
               cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, dev) == cudaSuccess && integrated != 0;
    }();
    return on;
}

}  // namespace

void check_routed_experts(ExpertFormat f, const ExpertSlot* const* slots, int n, int group, const char* what) {
    if (f != ExpertFormat::Exl3 || !use_k19()) return;
    for (int e = 0; e < n; ++e) {
        const float* b = slots[e]->bits;
        const auto k = [](float x) { return x == (float) (int) x ? (int) x : 0; };   // half K: not K19's
        if (!kernels::k19_takes(k(b[0]), k(b[1]), k(b[2])))
            throw std::runtime_error("ds41: " + std::string(what) + " " + std::to_string(group) + " expert " +
                                     std::to_string(e) + " has gate / up / down at " + std::to_string(b[0]) + " / " +
                                     std::to_string(b[1]) + " / " + std::to_string(b[2]) +
                                     " bits, which K19 does not decode (DS41_EXL3_K19=0: K10)");
    }
}

void routed_moe_decode(ExpertFormat f, const __half* x, int m, const int32_t* sel, const float* w, int topk,
                       const kernels::Exl3Expert* experts, float* out, void* workspace, size_t workspace_bytes,
                       cudaStream_t stream) {
    if (f == ExpertFormat::Mxfp4)
        kernels::mxfp4_moe_decode(x, m, sel, w, topk, experts, out, workspace, workspace_bytes, stream);
    else if (use_k19())
        kernels::exl3_smoe_decode(x, m, sel, w, topk, experts, out, workspace, workspace_bytes, stream);
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
    if (f == ExpertFormat::Mxfp4) {
        kernels::mxfp4_moe_prefill(x, tok, w, off, n_groups, experts, out, workspace, workspace_bytes, stream);
        return;
    }
    if (!use_k19()) {
        kernels::exl3_moe_prefill(x, tok, w, off, n_groups, experts, out, workspace, workspace_bytes, stream);
        return;
    }
    // K19's GPU: a group of up to kSmallGroup rows runs as K21 (each expert read once per 8 rows, ~63 us for 13 MB),
    // batched up to kExl3SmoePrefillRows rows a call; a larger group as K12 (the expert dequantized once, then cuBLAS).
    // DGX Spark, a 240-token prompt: the routed experts took 3.28 s with K12 alone.
    constexpr int kSmallGroup = 128;
    static void* ws21 = nullptr;
    static const size_t ws21_bytes = kernels::exl3_smoe_workspace_bytes(1, 1);
    if (!ws21 && cudaMalloc(&ws21, ws21_bytes) != cudaSuccess) {
        cudaGetLastError();
        throw std::runtime_error("K21: cannot allocate its workspace");
    }
    int g = 0;
    while (g < n_groups) {
        if (off[g + 1] - off[g] > kSmallGroup) {
            kernels::exl3_moe_prefill(x, tok, w, off + g, 1, experts + g, out, workspace, workspace_bytes, stream);
            ++g;
            continue;
        }
        int e = g;   // groups g..e-1: small, at most kExl3SmoePrefillRows rows in all
        while (e < n_groups && off[e + 1] - off[e] <= kSmallGroup && off[e + 1] - off[g] <= kernels::kExl3SmoePrefillRows) ++e;
        kernels::exl3_smoe_prefill(x, tok, w, off + g, e - g, experts + g, out, ws21, ws21_bytes, stream);
        g = e;
    }
}

int64_t cpu_make_layer(ExpertFormat f, const ExpertSlot* const* slots, const uint8_t* const* bytes, int n,
                       float swiglu_limit) {
    if (f == ExpertFormat::Mxfp4) {
        std::vector<Mxfp4CpuExpert> e(n);
        for (int i = 0; i < n; ++i) e[i] = fp4_desc(*slots[i], bytes[i]);
        return mxfp4_cpu_make_layer(e.data(), n);
    }
    if (!kExl3Cpu) return -1;   // no CPU kernel: no layer (cpu_forward adds nothing for it)
    std::vector<MoeCpuMatrixDesc> gate(n), up(n), down(n);
    for (int i = 0; i < n; ++i) {
        gate[i] = exl3_desc(*slots[i], bytes[i], 0, kDim, kMoeInter);   // w1: 5120 -> 2304
        up[i] = exl3_desc(*slots[i], bytes[i], 4, kDim, kMoeInter);     // w3
        down[i] = exl3_desc(*slots[i], bytes[i], 8, kMoeInter, kDim);   // w2: 2304 -> 5120
    }
    return exl3_moe_cpu_make_layer_raw(gate.data(), up.data(), down.data(), n, 0, swiglu_limit, 0);
}

void cpu_free_layer(ExpertFormat f, int64_t handle) {
    if (handle < 0) return;
    if (f == ExpertFormat::Mxfp4) mxfp4_cpu_free_layer(handle);
    else exl3_moe_cpu_free_layer(handle);
}

void cpu_set_expert(ExpertFormat f, int64_t handle, int e, const ExpertSlot& slot, const uint8_t* bytes) {
    if (handle < 0) return;
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
    if (handle < 0) {   // no CPU kernel (EXL3 on ARM): these uses add nothing
        long long n = 0;
        for (int i = 0; i < rows * topk; ++i) n += ids[i] >= 0;
        if (n && dropped.fetch_add(n) == 0)
            std::fprintf(stderr, "ds41: an EXL3 expert use reached the CPU, which has no EXL3 kernel here: dropped\n");
        std::memset(out, 0, sizeof(float) * (size_t) rows * kDim);
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
    if (f == ExpertFormat::Exl3 && kExl3Cpu) exl3_moe_cpu_pool_prime(threads);
}

}  // namespace strata::ds41
