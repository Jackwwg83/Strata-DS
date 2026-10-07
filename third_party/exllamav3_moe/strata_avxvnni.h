// Strata-DS patch: exact 3-bit mul1 kernel for AVX-VNNI (VEX.256, no AVX-512).
// Included inside moe_mul1.cpp's anonymous namespace after the vendor AVX2 helpers.
// The existing AVX2/AVX-512 kernels and public interfaces remain unchanged.
#pragma once

#if defined(__GNUC__) && defined(__linux__) && __GNUC__ >= 11
#define STRATA_M1_AVXVNNI __attribute__((target("avx2,avxvnni,fma,f16c")))

inline bool strata_has_avx_vnni()
{
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return false;
    constexpr unsigned required = (1u << 27) | (1u << 28) | (1u << 12) | (1u << 29);
    if ((c & required) != required) return false; // OSXSAVE, AVX, FMA, F16C
    unsigned xlo, xhi;
    __asm__ volatile ("xgetbv" : "=a"(xlo), "=d"(xhi) : "c"(0));
    if ((xlo & 6u) != 6u) return false; // XMM and YMM context saving
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d) || a < 1 || !(b & (1u << 5)))
        return false;
    __cpuid_count(7, 1, a, b, c, d);
    return (a & (1u << 4)) != 0; // AVX-VNNI, independent of AVX512-VNNI
}

template <int row, int half>
STRATA_M1_AVXVNNI M1_ALWAYS_INLINE
__m256i strata_vnni_code(const __m256i (&packed)[3])
{
    constexpr int shift = row_shift<3, false, row>(half * 8);
    const __m256i b = avx2_gather_half<3, false, row, true, half>(packed);
    if constexpr (shift == 16)
        return _mm256_srli_epi32(b, 16);
    else if constexpr (shift < 16)
    {
        // The low 16 bits do not cross a packed-word boundary. The other word's
        // left-shifted contribution would be discarded by the final 0xffff mask.
        return _mm256_and_si256(_mm256_srli_epi32(b, shift), _mm256_set1_epi32(0xffff));
    }
    else
    {
        const __m256i a = avx2_gather_half<3, false, row, false, half>(packed);
        return _mm256_and_si256(_mm256_or_si256(_mm256_srli_epi32(b, shift),
            _mm256_slli_epi32(a, 32 - shift)), _mm256_set1_epi32(0xffff));
    }
}

template <int rows, int row = 0>
STRATA_M1_AVXVNNI M1_ALWAYS_INLINE
void strata_vnni_rows(const __m256i (&packed)[3], const int32_t* splat, int k,
                     __m256i (&acc)[rows][2], const __m256i mult)
{
    if constexpr (row < 16)
    {
        const __m256i p0 = _mm256_mullo_epi32(strata_vnni_code<row, 0>(packed), mult);
        const __m256i p1 = _mm256_mullo_epi32(strata_vnni_code<row, 1>(packed), mult);
        // Unsigned product bytes times the signed activation replicated four times.
        // Non-saturating vpdpbusd has exactly the vendor bytesum/maddwd result.
        // Compile-time row count keeps unused accumulators/branches out of the loop.
#define STRATA_VNNI_ACC(i) \
        if constexpr ((i) < rows) { \
            const __m256i x = _mm256_set1_epi32(splat[static_cast<size_t>(i) * k + row]); \
            acc[i][0] = _mm256_dpbusd_avx_epi32(acc[i][0], p0, x); \
            acc[i][1] = _mm256_dpbusd_avx_epi32(acc[i][1], p1, x); \
        }
        STRATA_VNNI_ACC(0) STRATA_VNNI_ACC(1) STRATA_VNNI_ACC(2) STRATA_VNNI_ACC(3)
#undef STRATA_VNNI_ACC
        strata_vnni_rows<rows, row + 1>(packed, splat, k, acc, mult);
    }
}

template <int rows>
STRATA_M1_AVXVNNI
void strata_vnni_tiles(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int tn0, int tn1)
{
    const int tiles_k = mat.k / 16;
    const int tiles_n = mat.n / 16;
    constexpr int packed_size = 48;
    const size_t row_stride = static_cast<size_t>(tiles_n) * packed_size;
    const __m256i mult = _mm256_set1_epi32(static_cast<int32_t>(MUL1_MULT));
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
            // Match the existing AVX2 traversal/prefetch policy; no weight repacking.
            const uint16_t* pf = packed + row_stride * 4;
            _mm_prefetch(reinterpret_cast<const char*>(pf), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(pf) + 64, _MM_HINT_T0);
            const __m256i p[3] = {
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed)),
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + 16)),
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + 32))
            };
            strata_vnni_rows<rows>(p, in.splat32 + tile_k * 16, mat.k, acc, mult);
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

inline bool strata_run_avx_vnni(const MoeCpuMatrix& mat, const PreparedIn& in,
                              float* tout, int m, int tn0, int tn1)
{
    if (mat.bits != 3 || mat.hb || mat.swz) return false;
    switch (m)
    {
        case 1: strata_vnni_tiles<1>(mat, in, tout, tn0, tn1); return true;
        case 2: strata_vnni_tiles<2>(mat, in, tout, tn0, tn1); return true;
        case 3: strata_vnni_tiles<3>(mat, in, tout, tn0, tn1); return true;
        case 4: strata_vnni_tiles<4>(mat, in, tout, tn0, tn1); return true;
        default: return false;
    }
}
#undef STRATA_M1_AVXVNNI
#else
// Preserve vendor fallback on toolchains without a separately targetable AVX-VNNI intrinsic.
inline bool strata_has_avx_vnni() { return false; }
inline bool strata_run_avx_vnni(const MoeCpuMatrix&, const PreparedIn&, float*, int, int, int)
{ return false; }
#endif
