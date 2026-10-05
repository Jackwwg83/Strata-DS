#pragma once
#include <cmath>

#ifdef __CUDACC__
#define K8_SPLIT_HD __host__ __device__ __forceinline__
#else
#define K8_SPLIT_HD inline
#endif

namespace strata::ds41::kernels::k8_split {
// The fixed oracle uses double after the FP32 dot. Preserve that precision
// through bias comparisons and normalization, including sub-FP32 near-ties.
K8_SPLIT_HD double score(float logit) {
    const double value = logit;
    return sqrt(value > 20.0 ? value : log1p(exp(value)));
}
K8_SPLIT_HD bool better(double value, int id, double other, int other_id) {
    return value > other || (value == other && id < other_id);
}
}  // namespace strata::ds41::kernels::k8_split
#undef K8_SPLIT_HD
