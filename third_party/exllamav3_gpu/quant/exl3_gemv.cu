// Strata raw-pointer adapter for the upstream small-row GEMV kernel.
#include "exl3_gemv.cuh"
#include "../util.h"
#include "../util.cuh"
#include "exl3_gemv_kernel.cuh"

namespace strata_exl3 {

void gemv_mul1_3bit(const GemvJob* jobs, int count, int max_n, cudaStream_t stream)
{
    // CFG=0: 16 k-split warps, two n-tiles per warp, four-step FP16 fold.
    // Ordinary launches let all tokens/slots run without a residency restriction.
    exl3_gemv_kernel<3, true, 2, 0, 0, false>
        <<<dim3(max_n / 32, count), 512, 0, stream>>>(jobs);
}

}  // namespace strata_exl3
