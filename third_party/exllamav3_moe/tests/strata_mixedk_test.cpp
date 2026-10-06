// Supplemental integer-reference check of K1..K6 CPU band dispatch.
// Linux: c++ -O2 -std=c++17 -pthread strata_mixedk_test.cpp -o /tmp/mixedk-bands
// The macOS host audit supplies an extracted band-only translation unit.
#ifdef MIXEDK_BANDS_ONLY
#include "mixedk_bands.inc"
#else
#include "../moe_mul1.cpp"
#endif
#include <random>

namespace {
int failures = 0;
size_t output_checks = 0;

// Integer reference matches AVX2's quantized activations; it does not compare
// against the different floating-point/activation math in scalar_tiles.
template<int bits>
void check_matrix(int k, int n, int m, int pattern, std::mt19937& rng)
{
    std::vector<uint16_t> packed(static_cast<size_t>(k) * n * bits / 16);
    for (auto& p : packed) p = pattern == 0 ? 0 : pattern == 1 ? 0xffff : uint16_t(rng());
    MoeCpuMatrix mat{}; mat.trellis = packed.data(); mat.k = k; mat.n = n; mat.bits = bits;
    std::vector<int32_t> dup(static_cast<size_t>(m) * k), splat(dup.size());
    std::vector<int> xs(dup.size());
    PreparedIn in{}; in.splat_dup = dup.data(); in.splat32 = splat.data();
    for (int i = 0; i < m; ++i) {
        in.q[i] = (i + 1) * 0.00390625f;
        for (int j = 0; j < k; ++j) {
            const int v = pattern == 0 ? -127 : pattern == 1 ? 127 : int(rng() % 255) - 127;
            xs[i * k + j] = v;
            const uint32_t d = uint16_t(v) | (uint32_t(uint16_t(v)) << 16);
            std::memcpy(&dup[i * k + j], &d, sizeof d);
            const uint32_t repeated = uint32_t(uint8_t(v)) * 0x01010101u;
            std::memcpy(&splat[i * k + j], &repeated, sizeof repeated);
            in.sum_x8[i] += v;
        }
    }
    std::vector<float> got(static_cast<size_t>(m) * n, std::nanf("")), integer_ref(got.size());
    // Split the output range to test nonzero tn0 and adjacent worker partitions.
    const int split = (n / 16) / 2;
    run_tiles(mat, in, got.data(), m, 0, split);
    run_tiles(mat, in, got.data(), m, split, n / 16);
    constexpr auto inv = make_tc_perm_inv();
    for (int i = 0; i < m; ++i) for (int col = 0; col < n; ++col) {
        int64_t acc = 0;
        for (int r = 0; r < k; ++r) {
            const uint16_t* tile = packed.data() + (static_cast<size_t>(r / 16) * (n / 16) + col / 16) * (16 * bits);
            const uint16_t state = decode_state_scalar<bits, false>(tile, inv[(r % 16) * 16 + col % 16]);
            const uint32_t prod = uint32_t(state) * MUL1_MULT;
            const int sum = (prod & 255) + ((prod >> 8) & 255) + ((prod >> 16) & 255) + (prod >> 24);
            acc += static_cast<int64_t>(sum) * xs[i * k + r];
        }
        if (acc < INT32_MIN || acc > INT32_MAX) ++failures;
        const float scale = mul1_k_inv() * in.q[i];
        const float corr = -510.0f * static_cast<float>(in.sum_x8[i]) * scale;
        integer_ref[i * n + col] = std::fma(static_cast<float>(acc), scale, corr);
    }
    for (size_t i = 0; i < got.size(); ++i) {
        ++output_checks;
        if (std::memcmp(&integer_ref[i], &got[i], sizeof(float)))
            ++failures;
    }
}
}

int main() {
    if (g_isa == Isa::Scalar) { std::puts("SKIP: AVX2/FMA unavailable"); return 77; }
    std::mt19937 rng(410159);
    for (int k : {16, 128, 2304, 5120})
        for (int n : {32, 128}) for (int m = 1; m <= 4; ++m) for (int pattern = 0; pattern < 3; ++pattern) {
            check_matrix<1>(k, n, m, pattern, rng);
            check_matrix<2>(k, n, m, pattern, rng);
            check_matrix<3>(k, n, m, pattern, rng);
            check_matrix<4>(k, n, m, pattern, rng);
            check_matrix<5>(k, n, m, pattern, rng);
            check_matrix<6>(k, n, m, pattern, rng);
        }
    std::printf("RESULT %s K1..K6 integer_reference bit_exact=1 output_checks=%zu failures=%d isa=%d synthetic=1\n",
                failures ? "fail" : "pass", output_checks, failures, int(g_isa));
    return failures ? 1 : 0;
}
