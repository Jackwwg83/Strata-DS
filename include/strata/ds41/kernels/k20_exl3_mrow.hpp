// include/strata/ds41/kernels/k20_exl3_mrow.hpp - task K20: dense EXL3 linears at decode sizes (1..16 rows), several per
// launch, with sudoingX's exl3_mrow kernel (third_party/exl3_smoe): a producer warp streams each k-row of a 128-column
// block with one TMA bulk copy into a shared-memory ring; eight warps decode the trellis into tensor-core fragments.
// Implementation in src/ds41/kernels/k20_exl3_mrow.cu.
#pragma once

#include "strata/ds41/kernels/k10_exl3_moe.hpp"

#include <cuda_bf16.h>

#include <vector>

namespace strata::ds41::kernels {

/// One linear of a group: y[:, out_col : out_col + n) = W x[:, x_col : x_col + k) (exllamav3's LinearEXL3)
struct MrowLinear {
    Exl3Proj w;
    int x_col = 0, out_col = 0;
};

/// A group of up to 8 linears of the same bitrate (2..6), run in one launch: registered once (its k-split plan and
/// scratch), then run per call. Rows 1..16; x and y bf16 with row strides that are multiples of 4. Deterministic (the
/// k-slices are summed in order), CUDA-graph capturable (no allocation or host synchronization in run()).
class Exl3Mrow {
public:
    explicit Exl3Mrow(const std::vector<MrowLinear>& linears);
    ~Exl3Mrow();
    Exl3Mrow(const Exl3Mrow&) = delete;
    Exl3Mrow& operator=(const Exl3Mrow&) = delete;
    void run(const __nv_bfloat16* x, int ldx, __nv_bfloat16* y, int ldy, int m, cudaStream_t stream) const;
    /// the same, y in FP32 (logits): the sums before the bf16 rounding
    void run_f32(const __nv_bfloat16* x, int ldx, float* y, int ldy, int m, cudaStream_t stream) const;
    static constexpr int kMaxRows = 16;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace strata::ds41::kernels
