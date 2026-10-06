// One token's original reference chains, shared by CUDA and the CPU model.
#pragma once
#ifdef __CUDACC__
#define K15_INLINE __device__ __forceinline__
#else
#define K15_INLINE inline
#endif

namespace strata::ds41::kernels::k15_detail {
constexpr int kColumns = 20480;
constexpr int kRows = 24;
constexpr int kThreads = 256;

// Each physical lane owns four original norm lanes. Keeping the four phases
// separate preserves all 20 stride-1024 FMA terms of each norm accumulator.
// Dot accumulators consume every phase in order: the reference's complete
// 80-term stride-256 FMA chain is never split into independent tile sums.
template <typename Load, typename Fma>
K15_INLINE void accumulate(Load load, Fma fma, float (&dots)[kRows], float (&squares)[4]) {
#pragma unroll 1
    for (int group = 0; group < kColumns / 1024; ++group) {
#pragma unroll
        for (int phase = 0; phase < 4; ++phase) {
            const int step = group * 4 + phase;
            const float value = load.value(step);
            squares[phase] = fma(value, value, squares[phase]);
#pragma unroll
            for (int row = 0; row < kRows; ++row)
                dots[row] = fma(value, load.weight(row, step), dots[row]);
        }
    }
}
}  // namespace strata::ds41::kernels::k15_detail
#undef K15_INLINE
