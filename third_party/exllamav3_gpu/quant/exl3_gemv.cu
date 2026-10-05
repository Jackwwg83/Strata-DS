// Strata raw-pointer adapter for the upstream small-row GEMV kernel.
#include "exl3_gemv.cuh"
#include "../util.h"
#include "../util.cuh"
#include "exl3_gemv_kernel.cuh"

namespace strata_exl3 {

void gemv_mul1_3bit(const GemvJob* jobs, int count, int max_n, cudaStream_t stream)
{
    // K10 gate/up jobs have K=5120, N=2304 (checked by input_had). Eight
    // k-splits keep the original four-slice FP16 fold boundaries. Down jobs
    // have K=2304 and retain 16 splits because their chunk tails differ.
    // Both paths keep two n-tiles/warp and the four-entry prefetch ring.
    if (max_n == 2304)
        exl3_gemv_kernel<3, true, 2, 0, 0, false, false, true>
            <<<dim3(max_n / 32, count), 256, 0, stream>>>(jobs);
    else
        exl3_gemv_kernel<3, true, 2, 0, 0, false>
            <<<dim3(max_n / 32, count), 512, 0, stream>>>(jobs);
}

}  // namespace strata_exl3
