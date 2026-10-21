// src/ds41/mxfp4_cpu.cpp - see include/strata/ds41/mxfp4_cpu.hpp.
//
// Two pool runs per call: the gate/up rows of every used expert (each row read once for all its tokens), then each
// part computes the activations it needs and its share of the down rows. A table turns a code byte into its two
// E2M1 values, so a 32-block costs 16 table loads and 32 FMAs in FP32; the products are exact, as in fp4_gemm.
#include "strata/ds41/mxfp4_cpu.hpp"

#include "strata/ds41/parallel.hpp"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace strata::ds41 {
namespace {

constexpr int H = 5120, F = 2304, kMaxRows = 8;

struct Layer {
    std::vector<Mxfp4CpuExpert> experts;
};

std::mutex g_mu;
std::unordered_map<int64_t, std::unique_ptr<Layer>> g_layers;
int64_t g_next = 1;
std::unique_ptr<ThreadPool> g_pool;
size_t g_pool_threads = 0;

Layer& layer_of(int64_t handle) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_layers.find(handle);
    if (it == g_layers.end()) throw std::invalid_argument("mxfp4_cpu: unknown layer handle");
    return *it->second;
}

ThreadPool& pool(int threads) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_pool || g_pool_threads != (size_t) threads) {
        g_pool.reset();
        g_pool.reset(new ThreadPool((size_t) threads));
        g_pool_threads = (size_t) threads;
    }
    return *g_pool;
}

/// byte -> its two E2M1 values (low nibble first)
struct Table {
    float v[256][2];
    Table() {
        static const float e2m1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                       -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
        for (int b = 0; b < 256; ++b) {
            v[b][0] = e2m1[b & 15];
            v[b][1] = e2m1[b >> 4];
        }
    }
};
const Table kTable;

inline float e8m0(uint8_t e) {
    uint32_t u = e ? (uint32_t) e << 23 : 0x00400000u;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
inline float bf16r(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    std::memcpy(&f, &u, 4);
    return f;
}
inline float round_pow2(float a) {
    uint32_t u;
    std::memcpy(&u, &a, 4);
    u = (u + 0x007fffffu) & 0x7f800000u;
    std::memcpy(&a, &u, 4);
    return a;
}

/// one output row against `nx` activation vectors (each k floats, `stride` apart): acc[i] for vector i
inline void row_dot(const uint8_t* codes, const uint8_t* scales, int k, const float* x, size_t stride, int nx,
                    float* acc) {
    for (int i = 0; i < nx; ++i) acc[i] = 0.0f;
    for (int b = 0; b < k / 32; ++b) {
        const uint8_t* q = codes + 16 * b;
        const float s = e8m0(scales[b]);
        for (int i = 0; i < nx; ++i) {
            const float* a = x + i * stride + 32 * b;
            float p0 = 0.0f, p1 = 0.0f;
            for (int j = 0; j < 16; ++j) {
                const float* t = kTable.v[q[j]];
                p0 = std::fma(t[0], a[2 * j], p0);
                p1 = std::fma(t[1], a[2 * j + 1], p1);
            }
            acc[i] = std::fma(p0 + p1, s, acc[i]);
        }
    }
}

}  // namespace

int64_t mxfp4_cpu_make_layer(const Mxfp4CpuExpert* experts, int n) {
    if (!experts || n <= 0) throw std::invalid_argument("mxfp4_cpu: no experts");
    auto l = std::make_unique<Layer>();
    l->experts.assign(experts, experts + n);
    std::lock_guard<std::mutex> lk(g_mu);
    const int64_t h = g_next++;
    g_layers[h] = std::move(l);
    return h;
}

void mxfp4_cpu_free_layer(int64_t handle) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_layers.erase(handle);
}

void mxfp4_cpu_set_expert(int64_t handle, int expert, const Mxfp4CpuExpert& e) {
    Layer& l = layer_of(handle);
    if (expert < 0 || expert >= (int) l.experts.size()) throw std::invalid_argument("mxfp4_cpu: bad expert");
    l.experts[expert] = e;
}

void mxfp4_cpu_forward(int64_t handle, const uint16_t* x, const int32_t* sel, const float* w, float* out, int rows,
                       int topk, int threads) {
    if (rows < 1 || rows > kMaxRows || topk < 1 || topk > 32 || threads < 1 || !x || !sel || !w || !out)
        throw std::invalid_argument("mxfp4_cpu: invalid arguments");
    Layer& l = layer_of(handle);
    const int n_exp = (int) l.experts.size();
    // the used experts and, per expert, the slots (token * topk + j) that use it
    std::vector<int> uniq;
    std::vector<std::vector<int>> users;
    for (int s = 0; s < rows * topk; ++s) {
        const int id = sel[s];
        if (id < 0) continue;
        if (id >= n_exp) throw std::invalid_argument("mxfp4_cpu: expert id out of range");
        auto it = std::find(uniq.begin(), uniq.end(), id);
        if (it == uniq.end()) {
            uniq.push_back(id);
            users.emplace_back();
            it = uniq.end() - 1;
        }
        users[it - uniq.begin()].push_back(s);
    }
    std::fill_n(out, (size_t) rows * H, 0.0f);
    if (uniq.empty()) return;

    std::vector<float> xf((size_t) rows * H);
    for (size_t i = 0; i < xf.size(); ++i) {
        __half_raw r;
        r.x = x[i];
        xf[i] = __half2float(__half(r));
    }
    const int slots = rows * topk;
    std::vector<float> gu((size_t) slots * 2 * F), hidden((size_t) slots * F), y((size_t) slots * H);
    // each expert's tokens' activations side by side, so a weight row is read once for all of them
    std::vector<std::vector<float>> xs(uniq.size());
    for (size_t u = 0; u < uniq.size(); ++u) {
        xs[u].resize(users[u].size() * (size_t) H);
        for (size_t i = 0; i < users[u].size(); ++i)
            std::memcpy(xs[u].data() + i * H, xf.data() + (size_t) (users[u][i] / topk) * H, H * 4);
    }

    ThreadPool& tp = pool(threads);
    const size_t parts = tp.size();
    const int gu_rows = (int) uniq.size() * 2 * F;
    tp.run(parts, [&](size_t p) {
        float acc[kMaxRows];
        for (int r = (int) (gu_rows * p / parts); r < (int) (gu_rows * (p + 1) / parts); ++r) {
            const int u = r / (2 * F), rr = r % (2 * F);
            const Mxfp4CpuProj& pr = rr < F ? l.experts[uniq[u]].w1 : l.experts[uniq[u]].w3;
            const int row = rr % F;
            row_dot(pr.codes + (size_t) row * (H / 2), pr.scales + (size_t) row * (H / 32), H, xs[u].data(), H,
                    (int) users[u].size(), acc);
            for (size_t i = 0; i < users[u].size(); ++i)
                gu[(size_t) users[u][i] * 2 * F + rr] = bf16r(acc[i]);
        }
    });
    for (int s = 0; s < slots; ++s) {   // the activation with its FP8 QDQ, per 32-block
        if (sel[s] < 0) continue;
        for (int b = 0; b < F / 32; ++b) {
            float h[32], amax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                const int k = 32 * b + i;
                const float g = std::fmin(gu[(size_t) s * 2 * F + k], 10.0f);
                const float u = std::fmin(std::fmax(gu[(size_t) s * 2 * F + F + k], -10.0f), 10.0f);
                h[i] = bf16r((g / (1.0f + std::exp(-g))) * u * w[s]);
                amax = std::fmax(amax, std::fabs(h[i]));
            }
            const float scale = round_pow2(std::fmax(amax, 1e-4f) * (1.0f / 448.0f));
            for (int i = 0; i < 32; ++i) {
                const __nv_fp8_e4m3 q(std::fmin(std::fmax(h[i] / scale, -448.0f), 448.0f));
                hidden[(size_t) s * F + 32 * b + i] = bf16r(float(q) * scale);
            }
        }
    }
    std::vector<std::vector<float>> hs(uniq.size());
    for (size_t u = 0; u < uniq.size(); ++u) {
        hs[u].resize(users[u].size() * (size_t) F);
        for (size_t i = 0; i < users[u].size(); ++i)
            std::memcpy(hs[u].data() + i * F, hidden.data() + (size_t) users[u][i] * F, F * 4);
    }
    const int d_rows = (int) uniq.size() * H;
    tp.run(parts, [&](size_t p) {
        float acc[kMaxRows];
        for (int r = (int) (d_rows * p / parts); r < (int) (d_rows * (p + 1) / parts); ++r) {
            const int u = r / H, row = r % H;
            const Mxfp4CpuProj& pr = l.experts[uniq[u]].w2;
            row_dot(pr.codes + (size_t) row * (F / 2), pr.scales + (size_t) row * (F / 32), F, hs[u].data(), F,
                    (int) users[u].size(), acc);
            for (size_t i = 0; i < users[u].size(); ++i) y[(size_t) users[u][i] * H + row] = bf16r(acc[i]);
        }
    });
    for (int t = 0; t < rows; ++t)
        for (int j = 0; j < topk; ++j) {
            const int s = t * topk + j;
            if (sel[s] < 0) continue;
            for (int r = 0; r < H; ++r) out[(size_t) t * H + r] += y[(size_t) s * H + r];
        }
}

}  // namespace strata::ds41
