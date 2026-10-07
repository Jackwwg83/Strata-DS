// include/strata/ds41/skip_misses.hpp - DS41_SKIP_MISS: leave light routed experts out when they are not in VRAM.
//
// A decode step copies every routed expert that misses the VRAM tier over PCIe (or computes it on the CPU). That
// copy is most of the step. An expert with a small routing weight adds little to the output, so a step may leave it
// out: a miss whose weight is below tau times the token's weight sum is skipped. The heaviest expert always stays.
// With renorm, the kept weights are scaled back to the old sum. This changes the model's output (opt-in); measure it
// with a teacher-forced nll (ds41_generate --force-ids).
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define DS41_HOST_DEVICE __host__ __device__
#else
#define DS41_HOST_DEVICE
#endif

namespace strata::ds41 {

/// One token's top-k: ids[topk] (negative: no expert) with weights w[topk] (changed in place with renorm), res the
/// layer's VRAM slot table (res[id] >= 0: in VRAM; null: no tier, every expert is a miss). Writes the ids the step
/// computes to out[topk] (-1: left out) and returns how many it left out.
DS41_HOST_DEVICE inline int skip_misses_row(const int32_t* ids, float* w, int topk, const int32_t* res, float tau,
                                            bool renorm, int32_t* out) {
    float sum = 0.0f, best = -1.0f;
    int top = -1;
    for (int j = 0; j < topk; ++j) {
        if (ids[j] < 0) continue;
        sum += w[j];
        if (w[j] > best) {
            best = w[j];
            top = j;
        }
    }
    float kept = 0.0f;
    int skipped = 0;
    for (int j = 0; j < topk; ++j) {
        const int32_t id = ids[j];
        const bool miss = id >= 0 && !(res && res[id] >= 0);
        const bool skip = miss && j != top && w[j] < tau * sum;
        out[j] = skip ? -1 : id;
        if (skip) ++skipped;
        else if (id >= 0) kept += w[j];
    }
    if (renorm && skipped && kept > 0.0f) {
        const float scale = sum / kept;
        for (int j = 0; j < topk; ++j)
            if (out[j] >= 0) w[j] *= scale;
    }
    return skipped;
}

}  // namespace strata::ds41
