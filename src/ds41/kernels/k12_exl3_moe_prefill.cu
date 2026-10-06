// K12 placeholder: computes nothing, so the acceptance test fails. Spec: ds41/tasks/K12.md
#include "strata/ds41/kernels/k12_exl3_moe_prefill.hpp"

namespace strata::ds41::kernels {

size_t exl3_moe_prefill_workspace_bytes(int max_rows, int max_groups) {
    (void) max_rows;
    (void) max_groups;
    return 256;
}

void exl3_moe_prefill(const __half* x, const int32_t* tok, const float* w, const int32_t* off, int n_groups,
                      const Exl3Expert* experts, float* out, void* workspace, size_t workspace_bytes,
                      cudaStream_t stream) {
    (void) x, (void) tok, (void) w, (void) off, (void) n_groups, (void) experts, (void) out;
    (void) workspace, (void) workspace_bytes, (void) stream;
}

}  // namespace strata::ds41::kernels
