// CPU semantic checks only. This does not execute or time the CUDA kernels.
// g++ -std=c++17 -O3 -march=native host_semantics.cpp -o /tmp/k8_semantics
#include "math.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace kd = strata::ds41::kernels::k8_detail;
constexpr int N = 384, D = 5120, K = 6, T = 128;
struct Result { std::array<int, K> ids; std::array<float, K> weights; };
struct Candidate { double value; int id; };
static int cases = 0, logits_checked = 0;
static double max_relative_error = 0;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
float bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}
std::vector<float> random_values(int n, float scale, unsigned seed, bool quantize) {
    std::mt19937 generator(seed);
    std::normal_distribution<float> distribution(0.0f, scale);
    std::vector<float> v(n);
    for (float& x : v) { x = distribution(generator); if (quantize) x = bf16(x); }
    return v;
}
float reference_dot(const float* x, const float* w) {
    std::array<float, 32> accum{};
    for (int lane = 0; lane < 32; ++lane)
        for (int d = lane; d < D; d += 32) accum[lane] = std::fma(x[d], w[d], accum[lane]);
    for (int offset = 16; offset > 0; offset >>= 1) {
        const auto old = accum;
        for (int lane = 0; lane + offset < 32; ++lane) accum[lane] = old[lane] + old[lane + offset];
    }
    return accum[0];
}
std::vector<float> tile_logits(const std::vector<float>& x, const std::vector<float>& w, int m) {
    std::vector<float> logits(m * N);
    std::vector<int> written(m * N);
    long weight_pairs = 0;
    // Two width-16 expert groups per 32-thread block, across 192 CTAs.
    for (int block = 0; block < N / 2; ++block) {
        for (int group = 0; group < 2; ++group) {
            const int e = block * 2 + group;
            std::array<std::array<float, 16>, 8> even{}, odd{};
            for (int base = 0; base < D / 2; base += 16) {
                for (int lane = 0; lane < 16; ++lane) {
                    const int pair = base + lane;
                    const float we = w[e * D + 2 * pair], wo = w[e * D + 2 * pair + 1];
                    ++weight_pairs; // Reused in registers across every token.
                    for (int t = 0; t < m; ++t) {
                        even[t][lane] = std::fma(x[t * D + 2 * pair], we, even[t][lane]);
                        odd[t][lane] = std::fma(x[t * D + 2 * pair + 1], wo, odd[t][lane]);
                    }
                }
            }
            for (int t = 0; t < m; ++t) {
                for (int offset = 8; offset > 0; offset >>= 1) {
                    const auto pe = even[t], po = odd[t];
                    for (int lane = 0; lane < 16; ++lane) {
                        // CUDA width-16 shuffles return the caller's own value
                        // when the source crosses its subgroup. Mirror those
                        // otherwise unused lanes too, not just the valid root.
                        const int src = lane + offset < 16 ? lane + offset : lane;
                        even[t][lane] = pe[lane] + pe[src];
                        odd[t][lane] = po[lane] + po[src];
                    }
                }
                // Lane t receives subgroup lane zero's logit and owns score t.
                for (int lane = 0; lane < 16; ++lane) if (lane == t) {
                    logits[t * N + e] = even[t][0] + odd[t][0];
                    ++written[t * N + e];
                }
            }
        }
    }
    require(weight_pairs == N * D / 2, "weight pair reread across tokens");
    for (int n : written) require(n == 1, "score output has missing or duplicate producer");
    return logits;
}

void prove_mapping() {
    // Track expression trees, including operand order, rather than relying on
    // a numerically lucky input. Each FMA leaf is a complete reference lane.
    std::array<std::string,32> reference;
    std::array<std::string,16> even, odd;
    for (int lane=0;lane<32;++lane) reference[lane]="r"+std::to_string(lane);
    for (int lane=0;lane<16;++lane) { even[lane]=reference[2*lane]; odd[lane]=reference[2*lane+1]; }
    for(int off=16;off;off>>=1) {
        auto old=reference;
        for(int lane=0;lane<32;++lane) reference[lane]="("+old[lane]+"+"+old[lane+off<32?lane+off:lane]+")";
    }
    for(int off=8;off;off>>=1) {
        auto pe=even,po=odd;
        for(int lane=0;lane<16;++lane) {
            const int src=lane+off<16?lane+off:lane;
            even[lane]="("+pe[lane]+"+"+pe[src]+")";
            odd[lane]="("+po[lane]+"+"+po[src]+")";
        }
    }
    require(reference[0]=="("+even[0]+"+"+odd[0]+")", "reduction expression tree differs");
    std::array<int,D> visits{};
    for(int lane=0;lane<16;++lane) for(int j=0;j<D/32;++j) {
        const int pair=lane+16*j;
        require(2*pair==2*lane+32*j && 2*pair+1==2*lane+1+32*j,"FMA lane order differs");
        ++visits[2*pair]; ++visits[2*pair+1];
    }
    for(int n:visits) require(n==1,"packed input mapping is not bijective");

    // Four possible base alignments: both aligned, x shifted by one BF16,
    // w shifted by one BF16, both shifted. All 5120-element rows preserve it.
    for(int xs=0;xs<2;++xs) for(int ws=0;ws<2;++ws) {
        std::vector<uint32_t> xb(8*D/2+1), wb(N*D/2+1);
        auto* xp=reinterpret_cast<uint16_t*>(xb.data())+xs;
        auto* wp=reinterpret_cast<uint16_t*>(wb.data())+ws;
        for(int i=0;i<8*D;++i) xp[i]=uint16_t(i*7919+31);
        for(int i=0;i<N*D;++i) wp[i]=uint16_t(i*6271+7);
        const bool aligned=((uintptr_t(xp)|uintptr_t(wp))&3u)==0;
        require(aligned==(xs==0&&ws==0),"alignment dispatch differs");
        for(int source=0;source<2;++source) {
            auto* p=source?wp:xp; const int count=source?N*D:8*D;
            for(int pair=0;pair<count/2;++pair) {
                uint32_t packed;
                if(aligned) std::memcpy(&packed,p+2*pair,4);
                else packed=uint32_t(p[2*pair])|(uint32_t(p[2*pair+1])<<16);
                require(uint16_t(packed)==p[2*pair]&&uint16_t(packed>>16)==p[2*pair+1],"packed/fallback pair mismatch");
            }
        }
    }
}

Result oracle(const std::vector<float>& logits, const std::vector<float>& bias) {
    std::array<double, N> raw{}, values{};
    std::array<int, N> order{};
    for (int e = 0; e < N; ++e) {
        const double z = logits[e];
        raw[e] = std::sqrt(z > 20 ? z : std::log1p(std::exp(z)));
        values[e] = raw[e] + bias[e];
    }
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + K, order.end(), [&](int a, int b) {
        return values[a] > values[b] || (values[a] == values[b] && a < b);
    });
    Result result;
    double sum = 0;
    for (int i = 0; i < K; ++i) sum += raw[order[i]];
    for (int i = 0; i < K; ++i) {
        result.ids[i] = order[i];
        result.weights[i] = float(raw[order[i]] / (sum + 1e-20) * 1.5);
    }
    return result;
}
Candidate warp_reduce(std::array<Candidate, 32> values) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const auto old = values;
        for (int lane = 0; lane + offset < 32; ++lane) {
            const auto a = old[lane + offset], b = old[lane];
            if (kd::better(a.value, a.id, b.value, b.id)) values[lane] = a;
        }
    }
    return values[0];
}
Result simulated_select(const std::vector<float>& logits, const std::vector<float>& bias) {
    std::array<double, N> raw{};
    std::array<Candidate, N> values{};
    for (int e = 0; e < N; ++e) {
        raw[e] = kd::score(logits[e]);
        values[e] = {raw[e] + double(bias[e]), e};
    }
    Result result;
    std::array<double, K> selected{};
    for (int i = 0; i < K; ++i) {
        std::array<Candidate, T> local;
        for (int thread = 0; thread < T; ++thread) {
            Candidate best{-INFINITY, N};
            for (int j = 0; j < N / T; ++j) {
                const Candidate c = values[thread + j * T];
                if (kd::better(c.value, c.id, best.value, best.id)) best = c;
            }
            local[thread] = best;
        }
        std::array<Candidate, 32> warp_winners;
        warp_winners.fill({-INFINITY, N});
        for (int warp = 0; warp < T / 32; ++warp) {
            std::array<Candidate, 32> lanes;
            std::copy_n(local.begin() + warp * 32, 32, lanes.begin());
            warp_winners[warp] = warp_reduce(lanes);
        }
        const int winner = warp_reduce(warp_winners).id;
        require(winner < N, "selection produced sentinel");
        result.ids[i] = winner;
        selected[i] = raw[winner];
        values[winner] = {-INFINITY, N};
    }
    double sum = 0;
    for (double s : selected) sum += s;
    for (int i = 0; i < K; ++i) result.weights[i] = float(selected[i] / (sum + 1e-20) * 1.5);
    return result;
}
Result compare(const std::vector<float>& logits, const std::vector<float>& bias) {
    const auto ref = oracle(logits, bias), got = simulated_select(logits, bias);
    require(ref.ids == got.ids, "expert IDs/order differ");
    for (int i = 0; i < K; ++i) {
        require(std::isfinite(got.weights[i]), "nonfinite output weight");
        const double error = std::abs(double(got.weights[i]) - ref.weights[i]);
        require(error <= 1e-5 * std::abs(double(ref.weights[i])), "weight tolerance exceeded");
        if (ref.weights[i] != 0) max_relative_error = std::max(max_relative_error, error / std::abs(double(ref.weights[i])));
    }
    ++cases;
    return got;
}
#ifndef K8_SEMANTICS_LIBRARY
int main() {
    prove_mapping();
    const auto w = random_values(N * D, 0.02f, 1, true);
    const auto bias = random_values(N, 0.1f, 2, false);
    for (int m = 1; m <= 8; ++m) {
        const auto x = random_values(m * D, 1.0f, 10 + m, true);
        const auto tiled = tile_logits(x, w, m);
        for (int t = 0; t < m; ++t) {
            std::vector<float> reference(N);
            for (int e = 0; e < N; ++e) {
                reference[e] = reference_dot(x.data() + t * D, w.data() + e * D);
                require(std::memcmp(&reference[e], &tiled[t * N + e], sizeof(float)) == 0, "FP32 logit bits differ");
                ++logits_checked;
            }
            compare(reference, bias);
        }
    }
    // High dynamic range and cancellation while retaining finite FP32 sums.
    auto extreme_x = random_values(8 * D, 1.0f, 29093, true);
    auto extreme_w = random_values(N * D, 1.0f, 39107, true);
    for(int i=0;i<8*D;++i) extreme_x[i]=std::ldexp(extreme_x[i],(i%101)-50);
    for(int i=0;i<N*D;++i) extreme_w[i]=std::ldexp(extreme_w[i],((i*37)%101)-50);
    const auto extreme = tile_logits(extreme_x,extreme_w,8);
    for(int t=0;t<8;++t) for(int e=0;e<N;++e) {
        const float ref=reference_dot(extreme_x.data()+t*D,extreme_w.data()+e*D);
        require(std::isfinite(ref),"extreme test unexpectedly overflowed");
        require(std::memcmp(&ref,&extreme[t*N+e],4)==0,"high-dynamic-range FP32 logit differs");
        ++logits_checked;
    }
    std::vector<float> logits(N, 0), b(N, 0);
    auto equal = compare(logits, b);
    for (int i = 0; i < K; ++i) require(equal.ids[i] == i, "all-equal tie did not select lower IDs");
    // A bias step erased by FP32 score+bias, but retained by the fixed oracle.
    b[383] = std::ldexp(1.0f, -26);
    require(float(kd::score(0) + b[383]) == float(kd::score(0)), "near-tie fixture not rounded away in FP32");
    require(compare(logits, b).ids[0] == 383, "near-tie incorrectly collapsed");
    b.assign(N, 0);
    std::fill(logits.begin(), logits.end(), -1000);
    auto zero = compare(logits, b);
    for (int i = 0; i < K; ++i) require(zero.ids[i] == i && zero.weights[i] == 0, "all-zero score behavior differs");
    const std::array<float, 16> edges = {
        -1000, -746, -745, -700, -100, -20, -1, -0.0f, 0, 1,
        std::nextafter(20.0f, 0.0f), 20, std::nextafter(20.0f, INFINITY),
        100, 1e20f, std::numeric_limits<float>::max()};
    for (float z : edges) { logits.assign(N, z); compare(logits, b); }
    for (int e = 0; e < N; ++e) { logits[e] = edges[e % edges.size()]; b[e] = float((e % 11) - 5) * 0.125f; }
    compare(logits, b);
    // Huge finite biases dominate every score even in double; ties must still
    // use IDs instead of depending on the reduction traversal order.
    for(float bias_value:{std::numeric_limits<float>::max(),-std::numeric_limits<float>::max()}) {
        logits.assign(N,0); b.assign(N,bias_value);
        for(int e=0;e<N;++e) logits[e]=float(e%23)-11;
        const auto huge_bias=compare(logits,b);
        for(int i=0;i<K;++i) require(huge_bias.ids[i]==i,"huge-bias tie differs");
    }
    // Repaired K8-03 underflow regression: BF16 impulse-realizable logits.
    // Test the affected pair at ranks 1/2 and 5/6 across selection warps.
    int underflow_cases=0;
    for(int step=0;step<=80;++step) {
        const float z=-80.0f-.5f*step;
        require(bf16(z)==z,"underflow fixture is not BF16-realizable");
        logits.assign(N,-200); b.assign(N,-1);
        logits[0]=z; b[0]=0; b[1]=z==-104?1e-24f:float(kd::score(z)*.75);
        const auto first=compare(logits,b); ++underflow_cases;
        require(first.ids==std::array<int,K>{0,1,2,3,4,5},"underflow ordering at rank 1/2");
        logits.assign(N,-200); b.assign(N,-1);
        for(int e:{14,45,127,380}) { logits[e]=0; b[e]=0; }
        logits[3]=z;b[3]=0;b[8]=z==-104?1e-24f:float(kd::score(z)*.75);
        const auto boundary=compare(logits,b); ++underflow_cases;
        require(boundary.ids==std::array<int,K>{14,45,127,380,3,8},"underflow ordering at rank 5/6");
    }
    // Mixed underflow, large positive scores, negative bias and repeated ties.
    for (unsigned seed = 100; seed < 1100; ++seed) {
        logits = random_values(N, seed % 2 ? 120.0f : 3.0f, seed, false);
        b = random_values(N, 2.0f, seed + 1000, false);
        for (int e = 0; e < N; e += 17) { logits[e] = 0; b[e] = 0.5f; }
        compare(logits, b);
    }
    std::printf("PASS CPU semantics: %d routing cases, %d bit-exact FP32 logits, max relative weight error %.3g\n", cases, logits_checked, max_relative_error);
    std::printf("PASS %d underflow regressions; symbolic FMA/reduction mapping; 4 alignment combinations; unique score writers and single weight-pair reads\n",underflow_cases);
    std::puts("Covers all m=1..8, exact ties, FP32-collapsed near-ties, softplus threshold neighbors, cancellation, large positive and underflow logits.");
    std::puts("GPU execution, CUDA libm parity, graph capture/replay, and timings remain untested.");
}

#endif
