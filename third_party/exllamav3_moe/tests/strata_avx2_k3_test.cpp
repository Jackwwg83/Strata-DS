// Supplemental synthetic checks for Strata-DS K11-02; not the fixed K11 golden test.
// Build: g++ -O3 -std=c++17 -pthread strata_avx2_k3_test.cpp -o /tmp/k11-k3-test
#include "../moe_mul1.cpp"
#include <random>

namespace {
int failures = 0;
size_t state_checks = 0, output_checks = 0;

template<int row = 0>
M1_TARGET_AVX2 void check_states(const uint16_t* packed, const __m256i (&preg)[3])
{
    if constexpr (row < 16) {
        alignas(32) uint32_t codes[16];
        _mm256_store_si256(reinterpret_cast<__m256i*>(codes), strata_avx2_k3_codes<row, 0>(preg));
        _mm256_store_si256(reinterpret_cast<__m256i*>(codes + 8), strata_avx2_k3_codes<row, 1>(preg));
        constexpr auto inv = make_tc_perm_inv();
        for (int c = 0; c < 16; ++c) {
            ++state_checks;
            if (codes[c] != decode_state_scalar<3, false>(packed, inv[row * 16 + c])) ++failures;
        }
        check_states<row + 1>(packed, preg);
    }
}

M1_TARGET_AVX2 void check_tile(const uint16_t* packed)
{
    __m256i preg[3];
    for (int i = 0; i < 3; ++i)
        preg[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + i * 16));
    check_states(packed, preg);
}

void candidate(const MoeCpuMatrix& mat, const PreparedIn& in, float* out, int m, int tn0, int tn1)
{
    switch (m) {
        case 1: strata_avx2_k3_tiles<1>(mat, in, out, tn0, tn1); return;
        case 2: strata_avx2_k3_tiles<2>(mat, in, out, tn0, tn1); return;
        case 3: strata_avx2_k3_tiles<3>(mat, in, out, tn0, tn1); return;
        case 4: strata_avx2_k3_tiles<4>(mat, in, out, tn0, tn1); return;
    }
}

// Integer reference matches AVX2's quantized activations; it does not compare
// against the different floating-point/activation math in scalar_tiles.
M1_TARGET_AVX2 void check_matrix(int k, int n, int m, int pattern, std::mt19937& rng)
{
    std::vector<uint16_t> packed(static_cast<size_t>(k) * n * 3 / 16);
    for (auto& p : packed) p = pattern == 0 ? 0 : pattern == 1 ? 0xffff : uint16_t(rng());
    MoeCpuMatrix mat{}; mat.trellis = packed.data(); mat.k = k; mat.n = n; mat.bits = 3;
    std::vector<int32_t> dup(static_cast<size_t>(m) * k);
    std::vector<int> xs(dup.size());
    PreparedIn in{}; in.splat_dup = dup.data();
    for (int i = 0; i < m; ++i) {
        in.q[i] = (i + 1) * 0.00390625f;
        for (int j = 0; j < k; ++j) {
            const int v = pattern == 0 ? -127 : pattern == 1 ? 127 : int(rng() % 255) - 127;
            xs[i * k + j] = v;
            const uint32_t d = uint16_t(v) | (uint32_t(uint16_t(v)) << 16);
            std::memcpy(&dup[i * k + j], &d, sizeof d);
            in.sum_x8[i] += v;
        }
    }
    std::vector<float> want(static_cast<size_t>(m) * n), got(want.size()), integer_ref(want.size());
    avx2_tiles<3, false>(mat, in, want.data(), m, 0, n / 16);
    // Split the output range to test nonzero tn0 and adjacent worker partitions.
    const int split = (n / 16) / 2;
    candidate(mat, in, got.data(), m, 0, split);
    candidate(mat, in, got.data(), m, split, n / 16);
    constexpr auto inv = make_tc_perm_inv();
    for (int i = 0; i < m; ++i) for (int col = 0; col < n; ++col) {
        int64_t acc = 0;
        for (int r = 0; r < k; ++r) {
            const uint16_t* tile = packed.data() + (static_cast<size_t>(r / 16) * (n / 16) + col / 16) * 48;
            const uint16_t state = decode_state_scalar<3, false>(tile, inv[(r % 16) * 16 + col % 16]);
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
        if (std::memcmp(&want[i], &got[i], sizeof(float)) || std::memcmp(&integer_ref[i], &got[i], sizeof(float)))
            ++failures;
    }
}
}

int main()
{
    if (!exl3_moe_cpu_has_avx2()) { std::puts("SKIP: AVX2/FMA unavailable"); return 77; }
    std::mt19937 rng(0x4b313102);
    uint16_t packed[48]{};
    check_tile(packed);
    std::fill_n(packed, 48, uint16_t(0xffff)); check_tile(packed);
    // Every single bit in the 768-bit ring, both set-only and cleared-only.
    for (int bit = 0; bit < 768; ++bit) {
        std::fill_n(packed, 48, uint16_t(0)); packed[bit / 16] = uint16_t(1u << (bit % 16)); check_tile(packed);
        std::fill_n(packed, 48, uint16_t(0xffff)); packed[bit / 16] ^= uint16_t(1u << (bit % 16)); check_tile(packed);
    }
    for (int r = 0; r < 4096; ++r) { for (auto& p : packed) p = uint16_t(rng()); check_tile(packed); }
    for (int k : {16, 32, 128, 2304, 5120, 8192})
        for (int n : {16, 48}) for (int m = 1; m <= 4; ++m) for (int pattern = 0; pattern < 3; ++pattern)
            check_matrix(k, n, m, pattern, rng);
    std::printf("RESULT %s state_checks=%zu output_checks=%zu failures=%d synthetic=1\n",
        failures ? "fail" : "pass", state_checks, output_checks, failures);
    return failures ? 1 : 0;
}
