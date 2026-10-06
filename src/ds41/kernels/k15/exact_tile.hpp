// Shared arithmetic schedule for K15-02's CUDA tile and its supplemental CPU model.
#pragma once
#include <cstddef>

#ifdef __CUDACC__
#define K15_INLINE __host__ __device__ __forceinline__
#else
#define K15_INLINE inline
#endif

namespace strata::ds41::kernels::k15_detail {
constexpr int kWidth = 20480;
constexpr int kRows = 24;
constexpr int kDotLanes = 256;
constexpr int kDotWarps = 8;
constexpr int kSteps = kWidth / kDotLanes;
constexpr int kMaxTokens = 16384;
constexpr int kTileTokens = 16;
constexpr int kTileRows = 3;
constexpr int kTileWarps = 4;

K15_INLINE size_t partial_index(int token, int row, int warp) {
    return (static_cast<size_t>(token) * kRows + row) * kDotWarps + warp;
}

K15_INLINE size_t workspace_size(int m) {
    return m >= 1 && m <= kMaxTokens
        ? static_cast<size_t>(m) * kRows * kDotWarps * sizeof(float) : 0;
}

// A tile owns complete reference lanes, not contiguous K slices. Every lane
// consumes i, i+256, ..., i+79*256 in that order. The order of independent
// accumulators may change, but no FP32 dot chain is split or reassociated.
template <int Tokens, int Rows, typename Load, typename Fma>
K15_INLINE void accumulate_tile(Load load, Fma fma, float (&dots)[Rows][Tokens]) {
#pragma unroll 1
    for (int step = 0; step < kSteps; ++step) {
        float weights[Rows];
        float values[Tokens];
#pragma unroll
        for (int row = 0; row < Rows; ++row) weights[row] = load.weight(row, step);
#pragma unroll
        for (int token = 0; token < Tokens; ++token) values[token] = load.value(token, step);
#pragma unroll
        for (int row = 0; row < Rows; ++row)
#pragma unroll
            for (int token = 0; token < Tokens; ++token)
                dots[row][token] = fma(values[token], weights[row], dots[row][token]);
    }
}
}  // namespace strata::ds41::kernels::k15_detail
#undef K15_INLINE
