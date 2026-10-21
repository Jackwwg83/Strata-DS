// include/strata/ds41/kernels/k19_exl3_smoe.hpp - task K19: routed experts from EXL3 (mul1) weights at decode sizes,
// with the streaming kernels of sudoingX's exl3_smoe (third_party/exl3_smoe), which read the weights on GB10 (DGX
// Spark) at 207-221 GB/s where exllamav3's small-row GEMV (K10) reads 46-96. K10's interface; implementation in
// src/ds41/kernels/k19_exl3_smoe.cu.
#pragma once

#include "strata/ds41/kernels/k10_exl3_moe.hpp"

namespace strata::ds41::kernels {

/// The bitrates K19 decodes: integer K 1..6 per projection, gate and up the same or one apart (SAGE's pairs)
constexpr bool k19_takes(int k_gate, int k_up, int k_down) {
    return k_gate >= 1 && k_gate <= 6 && k_up >= 1 && k_up <= 6 && k_down >= 1 && k_down <= 6 &&
           k_gate - k_up <= 1 && k_up - k_gate <= 1;
}

/// Workspace bytes for exl3_smoe_decode with m tokens of topk picks
size_t exl3_smoe_workspace_bytes(int m, int topk);

/// K10's call: m tokens (1..8), topk picks each, experts[sel[t][j]] (sel < 0: none), x [m][5120] fp16, w [m][topk],
/// out [m][5120] FP32 accumulated into. Each distinct expert of the call is read once for all its rows. The math is
/// exllamav3's (and exl3_smoe's), which K10's differs from in two places: the routing weight scales the down output (K10:
/// the hidden activation), and the hidden activation is not FP8 quantize-dequantized. Per token the picks are summed in
/// pick order, then added to out. Gate and up of an expert must have the same bitrate (2..6). No allocation, no host
/// synchronization (CUDA-graph capturable).
void exl3_smoe_decode(const __half* x, int m, const int32_t* sel, const float* w, int topk, const Exl3Expert* experts,
                      float* out, void* workspace, size_t workspace_bytes, cudaStream_t stream);

/// K21 (prompts): K12's grouped call with K19's kernels, for groups with few rows (K12 dequantizes a whole expert
/// even for one row). Group g is experts[g] with rows off[g]..off[g + 1] (off on the host, tok and w on the device, rows
/// of a group with distinct tokens); out[tok[r]] += the row's weighted output, in row order. At most
/// kExl3SmoePrefillRows rows a call. The workspace is exl3_smoe_workspace_bytes'.
constexpr int kExl3SmoePrefillRows = 512;
void exl3_smoe_prefill(const __half* x, const int32_t* tok, const float* w, const int32_t* off, int n_groups,
                       const Exl3Expert* experts, float* out, void* workspace, size_t workspace_bytes,
                       cudaStream_t stream);

}  // namespace strata::ds41::kernels
