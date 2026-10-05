// CPU-only arithmetic and ownership model. This does not run the CUDA kernel.
// g++ -O3 -std=c++17 -ffp-contract=off test_fused_router_model.cpp -o /tmp/k8_model
#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

constexpr int K = 5120, E = 384, TOP = 6;
using Row = std::array<float, E>;
uint64_t comparisons = 0;
struct Result { std::array<int, TOP> ids; std::array<float, TOP> weights; };
uint32_t bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }
float value(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
float bf16(float f) { uint32_t b = bits(f); b += 0x7fff + ((b >> 16) & 1); return value(b & 0xffff0000u); }
double raw(float z) { return std::sqrt(z > 20 ? double(z) : std::log1p(std::exp(double(z)))); }
float quick(float z) { return std::sqrt(z > 20 ? z : std::log1p(std::exp(z))); }

bool better(int a, int b, const std::array<double,E>& biased) {
    if (a < 0) return false;
    if (b < 0) return true;
    ++comparisons;
    return biased[a] > biased[b] || (biased[a] == biased[b] && a < b);
}

Result selection(const Row& z, const Row& bias) {
    Result result{};
    std::array<double,E> s, biased;
    for (int i = 0; i < E; ++i) { s[i] = raw(z[i]); biased[i] = s[i] + double(bias[i]); }
    std::array<int, E> want;
    std::iota(want.begin(), want.end(), 0);
    std::partial_sort(want.begin(), want.begin() + TOP, want.end(), [&](int a, int b) {
        double aa = raw(z[a]) + double(bias[a]), ab = raw(z[b]) + double(bias[b]);
        return aa > ab || (aa == ab && a < b);
    });
    std::array<unsigned, 32> removed{};
    std::array<double, TOP> selected{};
    for (int rank = 0; rank < TOP; ++rank) {
        std::array<int, 32> best;
        best.fill(-1);
        for (int lane = 0; lane < 32; ++lane) {
            for (int i = 0; i < E / 32; ++i) {
                int e = lane + 32*i;
                if (!(removed[lane] & (1u << i)) && better(e, best[lane], biased)) best[lane] = e;
            }
        }
        for (int off = 16; off; off >>= 1) {
            auto prev = best;
            for (int lane = 0; lane + off < 32; ++lane)
                if (better(prev[lane + off], prev[lane], biased)) best[lane] = prev[lane + off];
        }
        int win = best[0];
        if (win != want[rank]) {
            std::fprintf(stderr, "mismatch rank=%d got=%d wanted=%d logits=(%.9g,%.9g) bias=(%.9g,%.9g)\n",
                         rank, win, want[rank], z[win], z[want[rank]], bias[win], bias[want[rank]]);
            std::abort();
        }
        removed[win & 31] |= 1u << (win >> 5);
        selected[rank] = s[win];
        result.ids[rank] = win;
    }
    double sum = 0;
    for (double v : selected) sum += v;
    if (std::isfinite(sum)) for (int i = 0; i < TOP; ++i) {
        double target = selected[i] / (sum + 1e-20) * 1.5;
        float got = float(target);
        result.weights[i] = got;
        assert(std::fabs(double(got) - target) <= 1e-5 * std::fabs(target) + FLT_TRUE_MIN);
    }
    return result;
}

float ref_dot(const float* x, const float* w) {
    std::array<float,32> acc{};
    for (int l = 0; l < 32; ++l)
        for (int i = l; i < K; i += 32) acc[l] = std::fma(x[i], w[i], acc[l]);
    for (int off = 16; off; off >>= 1) {
        auto prev = acc;
        for (int l = 0; l + off < 32; ++l) acc[l] += prev[l + off];
    }
    return acc[0];
}
float fused_dot(const float* x, const float* w) {
    std::array<float,16> even{}, odd{};
    for (int l = 0; l < 16; ++l)
        for (int j = l; j < K/2; j += 16) {
            even[l] = std::fma(x[2*j], w[2*j], even[l]);
            odd[l] = std::fma(x[2*j+1], w[2*j+1], odd[l]);
        }
    for (int off = 8; off; off >>= 1) {
        auto pe = even, po = odd;
        for (int l = 0; l + off < 16; ++l) { even[l] += pe[l + off]; odd[l] += po[l + off]; }
    }
    return even[0] + odd[0];
}

int main() {
    std::mt19937 rng(38003);
    std::normal_distribution<float> normal;
    std::uniform_real_distribution<float> unit(-1,1);
    Row z{}, bias{};
    int cases = 0;
    auto check = [&] { selection(z, bias); ++cases; };
    check();
    for (int e = 0; e < E; ++e) bias[e] = float((E-e)/7); check();
    bias.fill(0);
    for (int e = 0; e < E; ++e) z[e] = float(-e)*10; check();
    for (int e = 0; e < E; ++e) z[e] = -800.0f-float(e); check();
    for (int e = 0; e < E; ++e) z[e] = value(1 + e%7); check();
    z.fill(0); for (int e = 0; e < E; ++e) bias[e] = std::ldexp(float(e), -30); check();
    z.fill(1e30f); for (int e = 0; e < E; ++e) bias[e] = std::ldexp(float(e), -30); check();
    z.fill(std::numeric_limits<float>::infinity()); check();
    // Regression for the reviewed legal BF16 input: x[0]=1, other x=0;
    // the first expert weights are -104 and -200. Ranking must use the
    // transform before its small nonzero score can underflow in FP32 exp.
    int underflow_regressions = 0;
    for (int step = 0; step <= 80; ++step) {
        const float az = -80.0f - .5f * step;
        assert(bf16(az) == az && bf16(-200) == -200);
        std::array<float,K> impulse{}, row{};
        impulse[0]=1; row[0]=bf16(az);
        assert(bits(ref_dot(impulse.data(),row.data()))==bits(az));
        assert(bits(fused_dot(impulse.data(),row.data()))==bits(az));
        row[0]=bf16(-200);
        assert(bits(ref_dot(impulse.data(),row.data()))==bits(-200));
        assert(bits(fused_dot(impulse.data(),row.data()))==bits(-200));
        z.fill(-200); bias.fill(-1);
        z[0] = az; bias[0] = 0; bias[1] = float(raw(az) * .75);
        if (az == -104) bias[1] = 1e-24f; // the exact independent-review case
        auto result = selection(z, bias); ++cases; ++underflow_regressions;
        assert((result.ids == std::array<int,TOP>{0,1,2,3,4,5}));
        const double a = raw(az), b = raw(-200), denominator = a + 5*b + 1e-20;
        for (int rank=0;rank<TOP;++rank) {
            const double expected = (rank == 0 ? a : b) / denominator * 1.5;
            assert(std::fabs(double(result.weights[rank])-expected) <= 1e-5*std::fabs(expected));
        }
        // Move the same pair across the final top-6 boundary and across warp
        // lanes, with four stronger equal experts. Check all six coefficients.
        z.fill(-200); bias.fill(-1);
        for (int id : {14,45,127,380}) { z[id]=0; bias[id]=0; }
        z[3]=az; bias[3]=0; bias[8]=float(a*.75);
        if (az == -104) bias[8] = 1e-24f;
        result=selection(z,bias); ++cases; ++underflow_regressions;
        assert((result.ids == std::array<int,TOP>{14,45,127,380,3,8}));
        const double strong = raw(0), sum = ((strong+strong)+strong)+strong+a+b;
        for (int rank=0;rank<TOP;++rank) {
            const double numerator = rank<4 ? strong : rank==4 ? a : b;
            const double expected = numerator / (sum+1e-20) * 1.5;
            // The last score can round to an FP32 subnormal, so its unavoidable
            // final-cast rounding is checked exactly as well as by magnitude.
            assert(bits(result.weights[rank]) == bits(float(expected)));
        }
    }
    // Establish the original failure independently of the fixed comparator.
    assert(quick(-104)==0 && quick(-200)==0);
    assert(quick(-104)+0 < quick(-200)+1e-24f);
    assert(raw(-104) > raw(-200)+double(1e-24f));
    for (int trial = 0; trial < 4000; ++trial) {
        for (int e = 0; e < E; ++e) {
            switch (trial % 8) {
            case 0: z[e] = normal(rng)*2; bias[e] = normal(rng)*.1f; break;
            case 1: z[e] = unit(rng)*1000; bias[e] = 0; break;
            case 2: z[e] = float(rng()%7); bias[e] = float(rng()%4); break;
            case 3: z[e] = unit(rng)*100; bias[e] = -quick(z[e]) + std::ldexp(unit(rng), -22); break;
            case 4: z[e] = std::nextafter(20.0f, e%2 ? INFINITY : -INFINITY); bias[e] = std::ldexp(unit(rng), -18); break;
            case 5: z[e] = -90-unit(rng)*15; bias[e] = std::ldexp(unit(rng), -70); break;
            case 6: z[e] = unit(rng)*1000; bias[e] = unit(rng)*1000; break;
            default: z[e] = std::ldexp(unit(rng), -120); bias[e] = std::ldexp(unit(rng), -60); break;
            }
        }
        check();
    }
    std::vector<float> w(E*K), x(K);
    for (float& v:w) v=bf16(normal(rng)*.02f);
    uint64_t dots=0;
    for (int m=1; m<=8; ++m) for (int t=0; t<m; ++t) {
        for (float& v:x) v=bf16(normal(rng));
        for (int e=0; e<E; ++e) {
            float a=ref_dot(x.data(),w.data()+e*K), b=fused_dot(x.data(),w.data()+e*K);
            assert(bits(a)==bits(b)); z[e]=b; bias[e]=normal(rng)*.1f; ++dots;
        }
        check();
    }
    // All 64 half-warp owners cover every expert once, and each vector load
    // covers exactly the scalar positions of the fixed reference lane pair.
    std::array<int,E> writes{};
    for(int group=0;group<64;++group) for(int e=group;e<E;e+=64) {
        ++writes[e]; std::array<int,K> reads{};
        for(int l=0;l<16;++l) for(int j=l;j<K/2;j+=16) { ++reads[j*2]; ++reads[j*2+1]; }
        for(int n:reads) assert(n==1);
    }
    for(int n:writes) assert(n==1);
    std::printf("PASS CPU model: %d selection/weight cases, %llu bitwise GEMV dots, all m=1..8, ownership bounds\n",cases,(unsigned long long)dots);
    std::printf("PASS underflow regression: %d full top-6 and normalization cases across logits -80..-120\n", underflow_regressions);
    std::printf("double-score comparison calls=%llu\n",(unsigned long long)comparisons);
    std::puts("CUDA execution, capture/replay, memory races and performance are NOT tested by this model.");
}
