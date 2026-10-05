// CPU-only checks for K2-13. CUDA host FP8/BF16 conversions; no GPU calls.
// This is a numerical/layout model, not a replacement for the fixed GPU test.
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using bf16 = __nv_bfloat16;
static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static bool same(float a, float b) {
    return bits(a) == bits(b) || (std::isnan(a) && std::isnan(b));
}
static float fp8(uint8_t b) { __nv_fp8_e4m3 q; q.__x = b; return float(q); }
static float stored(uint8_t b) { return __bfloat162float(__float2bfloat16_rn(fp8(b))); }
static float round_pow2(float a) {
    int e = 0; const float m = std::frexp(a, &e);
    return std::ldexp(1.0f, m == 0.5f ? e - 1 : e);
}
static bool safe(int a, int b) {
    return a >= -117 && a <= 118 && b >= -117 && b <= 118 &&
           a + b >= -108 && a + b <= 104;
}
static void quantize(const std::vector<float>& x, std::vector<uint8_t>& q,
                     std::vector<float>& scales) {
    q.resize(x.size()); scales.resize(x.size() / 32);
    for (size_t k = 0; k < x.size(); k += 32) {
        float amax = 0;
        for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[k + i]));
        const float s = round_pow2(std::fmax(amax, 1e-4f) * (1.0f / 448.0f));
        scales[k / 32] = s;
        for (int i = 0; i < 32; ++i) {
            __nv_fp8_e4m3 v(std::fmin(std::fmax(x[k + i] / s, -448.0f), 448.0f));
            q[k + i] = v.__x;
            assert(same(fp8(v.__x) * s, stored(v.__x) * s));
        }
    }
}
static float reference(const std::vector<uint8_t>& a, const std::vector<float>& as,
                       const std::vector<uint8_t>& w, const std::vector<uint8_t>& ws,
                       bool stored_a) {
    std::array<float, 32> lanes{};
    for (size_t k = 0; k < a.size(); ++k) {
        const float av = (stored_a ? stored(a[k]) : fp8(a[k])) * as[k / 32];
        const float bv = fp8(w[k]) * std::ldexp(1.0f, int(ws[k / 32]) - 127);
        lanes[k % 32] = std::fma(av, bv, lanes[k % 32]);
    }
    for (int off = 16; off > 0; off >>= 1) {
        auto prior = lanes;
        for (int lane = 0; lane + off < 32; ++lane) lanes[lane] += prior[lane + off];
    }
    return lanes[0];
}
static float candidate(const std::vector<uint8_t>& a, const std::vector<float>& as,
                       const std::vector<uint8_t>& w, const std::vector<uint8_t>& ws,
                       bool* fell_back = nullptr) {
    float total = 0; bool fallback = false;
    for (size_t k = 0; k < a.size(); k += 32) {
        int ae = int((bits(as[k / 32]) >> 23) & 255) - 127;
        int be = int(ws[k / 32]) - 127;
        if (!safe(ae, be)) { fallback = true; break; }
        float part = 0;
        // Serial fma within each K32 is one allowed host ordering model.
        // Hardware BF16 MMA internal ordering still needs the GPU acceptance.
        for (int i = 0; i < 32; ++i) part = std::fma(stored(a[k+i]), stored(w[k+i]), part);
        total = std::fma(part, std::ldexp(1.0f, ae + be), total);
    }
    fallback |= !std::isfinite(total);
    if (fell_back) *fell_back = fallback;
    return fallback ? reference(a, as, w, ws, true) : total;
}
static float bfround(float f) { return __bfloat162float(__float2bfloat16_rn(f)); }

int main() {
    // All FP8 encodings and E8M0 bytes, including signed zero, NaN, subnormals,
    // scales that underflow individual products, and overflow/255 scales.
    for (unsigned b = 0; b < 256; ++b) {
        assert(same(fp8(b), stored(b)));
        for (int e = -127; e <= 128; ++e)
            assert(same(fp8(b) * std::ldexp(1.f, e), stored(b) * std::ldexp(1.f, e)));
    }
    // Every finite BF16 encoding, in mixed-sign/range quantization blocks.
    std::vector<float> x(32);
    std::vector<uint8_t> a, w(32), ws(1);
    std::vector<float> as;
    unsigned finite_inputs = 0;
    for (unsigned b = 0; b < 65536; ++b) {
        uint32_t u = b << 16; float f; std::memcpy(&f, &u, 4);
        if (!std::isfinite(f)) continue;
        ++finite_inputs;
        for (int i = 0; i < 32; ++i) x[i] = i % 3 == 0 ? f : (i % 3 == 1 ? -f : 0.f);
        quantize(x, a, as);
    }
    // Full exponent grid, both boundaries and just-outside boundaries. Extreme
    // cases must take the conversion-identical fallback, not move the scale.
    unsigned range_cases = 0, fallback_cases = 0;
    std::mt19937 gen(13);
    for (int ae = -22; ae <= 120; ++ae) for (int be = -127; be <= 128; ++be) {
        a.resize(96); w.resize(96); ws.assign(3, uint8_t(be + 127)); as.assign(3, std::ldexp(1.f, ae));
        for (int k = 0; k < 96; ++k) { a[k] = uint8_t(gen() % 255); w[k] = uint8_t(gen() % 255); }
        bool fallback;
        float got = candidate(a, as, w, ws, &fallback);
        float ref = reference(a, as, w, ws, false);
        if (fallback) { assert(same(got, ref)); ++fallback_cases; }
        if (!safe(ae, be)) assert(fallback);
        ++range_cases;
    }
    // Shape-like K values and independent per-K32 scales. Test all five K
    // regimes including K32/K96 tails, sparse values, alternating signs and
    // near cancellation. Keep L2 checks row-based, as in the fixed test.
    double worst = 0;
    for (int K : {32, 96, 1280, 5120, 8192}) for (int row = 0; row < 16; ++row) {
        x.resize(K); a.resize(K); w.resize(K); ws.resize(K / 32);
        std::normal_distribution<float> d(0.f, 1.f), wd(0.f, 40.f);
        for (int k = 0; k < K; ++k) x[k] = bfround(d(gen));
        quantize(x, a, as);
        double err = 0, norm = 0;
        for (int col = 0; col < 128; ++col) {
            for (int k = 0; k < K; ++k) {
                __nv_fp8_e4m3 q(std::fmin(std::fmax(wd(gen), -448.f), 448.f));
                w[k] = q.__x;
                if (row % 4 == 0 && k % 7 == 0) w[k] = 0;
                if (row % 4 == 1 && k % 2 == 1) { a[k] = a[k-1]; w[k] = w[k-1] ^ 128; }
            }
            for (auto& e : ws) e = uint8_t(118 + gen() % 5);
            const float got = bfround(candidate(a, as, w, ws));
            const float ref = bfround(reference(a, as, w, ws, false));
            err += double(got-ref)*(got-ref); norm += double(ref)*ref;
        }
        const double rel = std::sqrt(err / std::max(norm, 1e-30));
        worst = std::max(worst, rel);
        assert(rel <= 0.002);
    }
    // Demonstrate why scaling must precede cross-K32 accumulation.
    a.assign(64, 0); w.assign(64, 0); a[0] = a[32] = 0x38;
    w[0] = 0x38; w[32] = 0xb8; as = {1.f, 2.f}; ws = {127,127};
    assert(candidate(a, as, w, ws) == -1.f);
    // Enumerate output ownership, M/N tails, and K-independent workspace bound.
    a.clear(); w.clear(); as.clear(); ws.clear();
    assert(candidate(a, as, w, ws) == 0.0f);
    for (int rows : {16,32,64,128}) {
        std::vector<int> visits(rows*32);
        for (int r=0;r<rows;++r) for(int k=0;k<32;++k) {
            int index=(r*32+k)^((r&7)*8);
            assert(index>=0 && index<rows*32); ++visits[index];
            if(k%8==0) assert(index%8==0);
        }
        for(int v:visits) assert(v==1);
    }
    unsigned layouts = 0;
    for (int M : {1, 15, 16, 17, 32, 63, 64, 77, 129})
    for (int N : {1, 7, 31, 32, 63, 64, 65, 127, 128, 129})
    for (auto tile : {std::array<int,4>{16,64,16,128}, {32,64,32,128}, {64,128,64,256}}) {
        auto [TM,TN,WN,T] = tile;
        std::vector<int> visits(M*N);
        for (int rb=0;rb<M;rb+=TM) for(int cb=0;cb<N;cb+=TN)
        for(int tid=0;tid<T;++tid) for(int j=0;j<WN/8;++j) for(int i=0;i<4;++i) {
            int warp=tid/32,lane=tid%32;
            int r=rb+(warp/(TN/WN))*16+lane/4+(i/2)*8;
            int c=cb+(warp%(TN/WN))*WN+j*8+(lane%4)*2+i%2;
            if(r<M && c<N) ++visits[r*N+c];
        }
        for(int v:visits) assert(v==1);
        ++layouts;
    }
    for (int M : {1,77,512,2048,16384}) for (int K : {0,32,96,1280,5120,8192}) {
        int64_t mk=int64_t(M)*K;
        assert(2*mk+(mk/32)*4==17*mk/8 && 17*mk/8<=4*mk && (2*mk)%4==0);
    }
    std::printf("PASS: 65536 FP8/E8M0 conversions; %u finite BF16 inputs; %u exponent-grid dots (%u exact fallbacks); worst modeled row L2 %.9g; %u tail ownership layouts; workspace 17*M*K/8 <= 4*M*K\n", finite_inputs,range_cases,fallback_cases,worst,layouts);
}
