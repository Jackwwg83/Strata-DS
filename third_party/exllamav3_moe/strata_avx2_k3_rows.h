// Strata-DS K11-02 patch: plain AVX2 K3 with a compile-time token-row count.
// Included inside moe_mul1.cpp's anonymous namespace, after the vendor AVX2 helpers.
// Other bitrates, activation preparation and all AVX-512 tiers stay on vendor code.

template <int row, int half>
M1_TARGET_AVX2
M1_ALWAYS_INLINE __m256i strata_avx2_k3_codes(const __m256i (&preg)[3])
{
    constexpr int shift = row_shift<3, false, row>(half * 8);
    const __m256i b = avx2_gather_half<3, false, row, true, half>(preg);
    const __m256i mask16 = _mm256_set1_epi32(0xffff);
    if constexpr (shift <= 16)
    {
        // The 16-bit window is wholly in b. Bits from a << (32 - shift)
        // cannot survive mask16, including shift == 0 (a << 32 is zero).
        return _mm256_and_si256(_mm256_srli_epi32(b, shift), mask16);
    }
    else
    {
        const __m256i a = avx2_gather_half<3, false, row, false, half>(preg);
        return _mm256_and_si256(_mm256_or_si256(
            _mm256_srli_epi32(b, shift), _mm256_slli_epi32(a, 32 - shift)), mask16);
    }
}

template <int rows, int row = 0>
M1_TARGET_AVX2
M1_ALWAYS_INLINE void strata_avx2_k3_accum(
    const __m256i (&preg)[3], const int32_t* splat_dup, int k,
    __m256i (&acc)[rows][2], const __m256i& mult, const __m256i& ones32)
{
    if constexpr (row < 16)
    {
        const __m256i lo = strata_avx2_k3_codes<row, 0>(preg);
        const __m256i hi = strata_avx2_k3_codes<row, 1>(preg);
        // Same nonsaturating byte-pair sums and signed-16 multiply as vendor AVX2.
        // Keeping rows constant removes per-row branches and memory-resident acc.
        const __m256i p_lo = _mm256_maddubs_epi16(_mm256_mullo_epi32(lo, mult), ones32);
        const __m256i p_hi = _mm256_maddubs_epi16(_mm256_mullo_epi32(hi, mult), ones32);
        #define STRATA_K3_ACC(i) \
            if constexpr ((i) < rows) { \
                const __m256i xs = _mm256_set1_epi32(splat_dup[static_cast<size_t>(i) * k + row]); \
                acc[i][0] = _mm256_add_epi32(acc[i][0], _mm256_madd_epi16(p_lo, xs)); \
                acc[i][1] = _mm256_add_epi32(acc[i][1], _mm256_madd_epi16(p_hi, xs)); \
            }
        STRATA_K3_ACC(0) STRATA_K3_ACC(1) STRATA_K3_ACC(2) STRATA_K3_ACC(3)
        #undef STRATA_K3_ACC
        strata_avx2_k3_accum<rows, row + 1>(preg, splat_dup, k, acc, mult, ones32);
    }
}

template <int rows>
M1_TARGET_AVX2
void strata_avx2_k3_tiles(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int tn0, int tn1)
{
    static_assert(rows >= 1 && rows <= MAX_M, "one specialization per prepared row count");
    constexpr int packed_size = tile_u16(3, false);
    const int tiles_k = mat.k / 16;
    const size_t row_stride = static_cast<size_t>(mat.n / 16) * packed_size;
    const __m256i mult = _mm256_set1_epi32(static_cast<int32_t>(MUL1_MULT));
    const __m256i ones32 = _mm256_set1_epi32(0x01010101);
    for (int tile_n = tn0; tile_n < tn1; ++tile_n)
    {
        __m256i acc[rows][2];
        for (int i = 0; i < rows; ++i)
        {
            acc[i][0] = _mm256_setzero_si256();
            acc[i][1] = _mm256_setzero_si256();
        }
        const uint16_t* packed = mat.trellis + static_cast<size_t>(tile_n) * packed_size;
        for (int tile_k = 0; tile_k < tiles_k; ++tile_k, packed += row_stride)
        {
            // Retain vendor AVX2's four-row lookahead and two cache-line touches.
            const uint16_t* pf = packed + row_stride * 4;
            _mm_prefetch(reinterpret_cast<const char*>(pf), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(pf) + 64, _MM_HINT_T0);
            __m256i preg[3];
            for (int i = 0; i < 3; ++i)
                preg[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + i * 16));
            strata_avx2_k3_accum<rows>(preg, in.splat_dup + tile_k * 16, mat.k, acc, mult, ones32);
        }
        for (int i = 0; i < rows; ++i)
        {
            const float scale = mul1_k_inv() * in.q[i];
            const __m256 corr = _mm256_set1_ps(-510.0f * static_cast<float>(in.sum_x8[i]) * scale);
            float* out = tout + static_cast<size_t>(i) * mat.n + tile_n * 16;
            _mm256_storeu_ps(out, _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[i][0]), _mm256_set1_ps(scale), corr));
            _mm256_storeu_ps(out + 8, _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc[i][1]), _mm256_set1_ps(scale), corr));
        }
    }
}
