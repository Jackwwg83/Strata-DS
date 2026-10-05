#pragma once
namespace strata::ds41::kernels::k8_detail {
// Fixed dimensional tiling, shared by the kernel and its CPU index-order model.
// These resource choices cover every legal m, not just acceptance-test shapes.
constexpr int kTile = 1024;
constexpr int kExpertsPerBlock = 2;
constexpr int kCopyValues = 8;  // One aligned 16-byte cp.async transaction.
}  // namespace strata::ds41::kernels::k8_detail
