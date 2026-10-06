// Host-only validation of the production pair decoder and unchanged lane reduction.
// Build with nvcc -std=c++17 -O3 -I include; no GPU is needed to execute this file.
#include "strata/ds41/fp8_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>
namespace strata::ds41 {
namespace {
#include "pair_decode.cuh"
}
}
namespace sd = strata::ds41;
namespace d = sd::detail;
static float2 checked_pair(uint16_t q, uint8_t s) {
    return s <= 246 ? sd::decode_small_pair<true>(q, s) : sd::decode_small_pair<false>(q, s);
}
using strata::kernels::bf16_from_f32;
static void require(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }
static bool equal(float a, float b) {
    return std::isnan(a) ? std::isnan(b) : d::float_bits(a) == d::float_bits(b);
}
static void conversion_test() {
    unsigned long long pairs = 0;
    for (unsigned s = 0; s < 256; ++s) {
        const float sw = d::decode_e8m0(uint8_t(s));
        for (unsigned q = 0; q < 65536; ++q) {
            const float2 got = checked_pair(uint16_t(q), uint8_t(s));
            require(equal(got.x, d::decode_e4m3(uint8_t(q)) * sw), "pair low byte/scale");
            require(equal(got.y, d::decode_e4m3(uint8_t(q >> 8)) * sw), "pair high byte/scale");
            // CUDA's host fallback models the native SM89/120 conversion result.
            const auto raw = __nv_cvt_fp8x2_to_halfraw2(uint16_t(q), __NV_E4M3);
            const float2 native = __half22float2(__half2(raw));
            require(equal(native.x * sw, got.x) && equal(native.y * sw, got.y), "native pair parity");
            ++pairs;
        }
    }
    // Negative controls ensure the signed, lane-order and rare-scale probes are meaningful.
    const auto a = checked_pair(0xb801, 127);
    const auto b = checked_pair(0x01b8, 127);
    require(!equal(a.x, b.x), "byte-swap negative control");
    require(!equal(a.y, std::fabs(a.y)), "sign negative control");
    require(!equal(checked_pair(0x0101, 0).x,
                   checked_pair(0x0101, 1).x), "subnormal scale negative control");
    require(std::isfinite(checked_pair(0x0101, 247).x), "extreme scale fallback control");
    require(std::isnan(checked_pair(0x007f, 127).x), "NaN negative control");
    std::printf("pair conversions: %llu pairs x 2 values, all 256 scale codes, software/native-host/reference agree; negative controls PASS\n", pairs);
}
static void layout_test() {
    unsigned cases = 0;
    for (int n : {1, 2, 31, 32, 33, 511, 512, 1023, 1024, 1025, 1280, 2048,
                  2049, 2304, 4096, 5120, 8191, 8192, 8193, 524289}) {
        const int split = d::gemv_split_warps(n), rows = d::gemv_rows_per_group(n);
        const int groups = 4 / split, per = groups * rows;
        const int grid = std::min((n + per - 1) / per, 65535);
        std::vector<int> writes(n);
        for (int block = 0; block < grid; ++block)
            for (int base = block * per; base < n; base += grid * per)
                for (int g = 0; g < groups; ++g)
                    for (int r = 0; r < rows && base + g * rows + r < n; ++r) {
                        const int row = base + g * rows + r;
                        require(row / 32 == (base + g * rows) / 32, "scale reuse crossed row");
                        ++writes[row];
                    }
        require(std::all_of(writes.begin(), writes.end(), [](int x) { return x == 1; }), "output ownership");
        for (int k : {32, 64, 96, 512, 544, 1024, 1280, 2048, 2304, 4096, 5120, 6144, 8192, 8224}) {
            std::vector<int> reads(k);
            for (int l = 0; l < split * 32; ++l)
                for (int col = l * 16; col < k; col += split * 512)
                    for (int j = 0; j < 16; ++j) {
                        require(col + j < k && col / 32 == (col + j) / 32, "packet tail/scale ownership");
                        ++reads[col + j];
                    }
            require(std::all_of(reads.begin(), reads.end(), [](int x) { return x == 1; }), "weight ownership");
            ++cases;
        }
    }
    std::printf("layout: %u geometries, unique reads/writes, tails and capped grid PASS\n", cases);
}
static void accumulation_test() {
    std::mt19937 rng(410109);
    unsigned cases = 0;
    for (int n : {1, 33, 512, 1023, 1024, 1025, 1280, 2048, 2049, 2304, 4096, 5120, 8192})
        for (int k : {32, 96, 544, 1280, 2304, 5120, 6144, 8192})
            for (int trial = 0; trial < 24; ++trial) {
                const int split = d::gemv_split_warps(n);
                std::vector<uint8_t> w(k), s(k / 32);
                std::vector<float> x(k);
                for (int i = 0; i < k; ++i) {
                    w[i] = uint8_t(rng() % 127) | uint8_t((rng() & 1) << 7);
                    x[i] = d::decode_e4m3(uint8_t(rng() % 127) | uint8_t((rng() & 1) << 7)) * 0x1p-8f;
                }
                for (auto& scale : s) scale = uint8_t(110 + rng() % 35);
                float old_acc[128] = {}, new_acc[128] = {};
                for (int l = 0; l < split * 32; ++l)
                    for (int col = l * 16; col < k; col += split * 512)
                        for (int j = 0; j < 16; j += 2) {
                            const int i = col + j;
                            const uint8_t scale = s[col / 32];
                            const auto v = checked_pair(uint16_t(w[i]) | (uint16_t(w[i + 1]) << 8), scale);
                            old_acc[l] = std::fma(x[i], d::decode_e4m3(w[i]) * d::decode_e8m0(scale), old_acc[l]);
                            old_acc[l] = std::fma(x[i + 1], d::decode_e4m3(w[i + 1]) * d::decode_e8m0(scale), old_acc[l]);
                            new_acc[l] = std::fma(x[i], v.x, new_acc[l]);
                            new_acc[l] = std::fma(x[i + 1], v.y, new_acc[l]);
                        }
                for (int l = 0; l < split * 32; ++l) require(equal(old_acc[l], new_acc[l]), "lane FP32 equality");
                for (int z = 0; z < split; ++z)
                    for (int step = 16; step; step /= 2)
                        for (int l = 0; l < step; ++l) {
                            old_acc[z * 32 + l] += old_acc[z * 32 + l + step];
                            new_acc[z * 32 + l] += new_acc[z * 32 + l + step];
                        }
                for (int z = 1; z < split; ++z) { old_acc[0] += old_acc[z * 32]; new_acc[0] += new_acc[z * 32]; }
                require(bf16_from_f32(old_acc[0]) == bf16_from_f32(new_acc[0]), "final BF16 equality");
                ++cases;
            }
    std::printf("accumulation: %u sampled output rows, lane FP32 and final BF16 bit-exact to control PASS\n", cases);
}
int main() {
    try { conversion_test(); layout_test(); accumulation_test(); }
    catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
    std::puts("HOST PASS; GPU numerics, graph replay and speed are not established");
}
