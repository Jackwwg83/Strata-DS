// Scalar load-ahead shared by the CUDA scorer and its CPU schedule model.
// Only load timing changes: one FP32 FMA chain consumes steps 0..Steps-1.
#pragma once

#ifdef __CUDACC__
#define K8_PIPELINE_INLINE __device__ __forceinline__
#else
#define K8_PIPELINE_INLINE inline
#endif

namespace strata::ds41::kernels::k8_detail {

struct Pair { float input; float weight; };

template <int Steps, typename Load, typename Fma>
K8_PIPELINE_INLINE float register_prefetch(Load load, Fma fma) {
    static_assert(Steps >= 2);
    Pair current = load(0);
    float acc = 0.0f;
#pragma unroll 8
    for (int step = 0; step < Steps - 1; ++step) {
        const Pair next = load(step + 1);
        acc = fma(current.input, current.weight, acc);
        current = next;
    }
    // Drain the last legal pair exactly once; never load step Steps.
    return fma(current.input, current.weight, acc);
}

}  // namespace strata::ds41::kernels::k8_detail
#undef K8_PIPELINE_INLINE
