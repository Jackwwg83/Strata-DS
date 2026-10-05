#pragma once
namespace strata::ds41::kernels::k8_detail {
// Fixed interface dimensions, also consumed by the CPU ownership/phase model.
constexpr int kExpertsPerBlock = 4;
constexpr int kWeightTile = 2048;
constexpr int kDimension = 5120;
#if defined(__CUDACC__)
__host__ __device__
#endif
constexpr int shared_bytes(int tokens) {
    return 2 * (tokens * kDimension + kExpertsPerBlock * kWeightTile);
}
}  // namespace strata::ds41::kernels::k8_detail
