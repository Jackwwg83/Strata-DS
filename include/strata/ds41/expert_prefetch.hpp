// include/strata/ds41/expert_prefetch.hpp - decode prefetch of the next layer's experts.
//
// While layer l computes, layer l+1's router applied to layer l's expert input guesses layer l+1's experts (upstream's
// router lookahead, docs/DETAILS.md, here on the GPU). The guesses outside VRAM that the RAM tier holds are copied to
// a VRAM buffer, one per layer parity. The guesses and the copy run on a stream of their own, beside layer l's shared
// expert, its experts and the wait for the CPU: on the main stream they cost 32 + 10 us per layer and the copy waited
// for layer l's experts (RTX 5090 Laptop, nsys, 2026-10-10). When layer l+1 routes, a miss that was guessed is
// computed by the GPU from the buffer (the doorbell's publish gets the guessed ids and their descriptors), so its
// copy is off the critical path.
// Measured on SAGE 1.59bpw (RTX 3090 box, DS41_PREDICT_STATS): 9 guesses hold 77-79% of the misses, 6 hold 66%.
// Everything is enqueued on the device (capturable): no host wait.
#pragma once
#include "strata/ds41/expert_staging.hpp"
#include "strata/ds41/kernels/k10_exl3_moe.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strata::ds41 {

class ExpertPrefetch {
public:
    static constexpr int kMaxGuesses = 16;

    /// `guesses` per layer (1..kMaxGuesses), a buffer of `buffer_bytes` per layer parity, routers of n_experts x dim
    ExpertPrefetch(int guesses, size_t buffer_bytes, int n_experts, int dim);
    ~ExpertPrefetch();
    ExpertPrefetch(const ExpertPrefetch&) = delete;
    ExpertPrefetch& operator=(const ExpertPrefetch&) = delete;

    int guesses() const { return guesses_; }

    /// On the prefetch stream, after the work enqueued on `main` so far: layer `layer`'s guesses from x (bf16 [dim],
    /// the previous layer's expert input) and its router (w [n_experts][dim] bf16, bias [n_experts]); the copy plan: in
    /// rank order, each guess outside VRAM (res[id] < 0) that the RAM tier holds (ram[id].w1.trellis set) while the
    /// buffer has room (res, ram, blobs: the layer's rows); then the copy. The buffer of this parity must be free.
    void plan(int layer, const __nv_bfloat16* x, const __nv_bfloat16* w, const float* bias, const int32_t* res,
              const kernels::Exl3Expert* ram, const ExpertBlob* blobs, cudaStream_t main);
    /// `main` waits for the guesses of `layer` (ids() and descs()); x may be written after this.
    void ready(int layer, cudaStream_t main);
    /// `main` waits for the copy of `layer` (before the GPU computes the layer's experts).
    void join(int layer, cudaStream_t main);

    /// The layer's copied guesses [guesses] (-1 after the last) and their descriptors in the buffer (device).
    /// Valid for the layer planned last with this parity.
    const int32_t* ids(int layer) const { return ids_ + (size_t) (layer & 1) * kMaxGuesses; }
    const kernels::Exl3Expert* descs(int layer) const { return descs_ + (size_t) (layer & 1) * kMaxGuesses; }
    /// The guesses in rank order, before the residency filter (device; tests)
    const int32_t* ranked(int layer) const { return ranked_ + (size_t) (layer & 1) * kMaxGuesses; }

private:
    int guesses_, n_experts_, dim_;
    size_t buffer_bytes_;
    uint8_t* buffer_ = nullptr;                 ///< [2][buffer_bytes]
    float* logits_ = nullptr;                   ///< [n_experts]
    int32_t* ids_ = nullptr;                    ///< [2][kMaxGuesses]
    int32_t* ranked_ = nullptr;                 ///< [2][kMaxGuesses]
    kernels::Exl3Expert* descs_ = nullptr;      ///< [2][kMaxGuesses]
    ExpertCopy* jobs_ = nullptr;                ///< [2][kMaxGuesses]
    int* count_ = nullptr;                      ///< [2]
    cudaStream_t copy_ = nullptr;
    cudaEvent_t ready_[2] = {}, planned_[2] = {}, done_[2] = {};
};

}  // namespace strata::ds41
