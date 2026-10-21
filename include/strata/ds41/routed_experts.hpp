// include/strata/ds41/routed_experts.hpp - the routed experts' kernels chosen by the pack's expert format: on the GPU
// K10 (exl3) or K16 (mxfp4), on the CPU exllamav3's moe_mul1 (exl3) or mxfp4_cpu (mxfp4). The engine calls these, so
// one engine serves both kinds of pack. The CPU side also builds each expert's descriptors from its ExpertSlot, in
// one place for the engine, DSpark and the RAM tier.
#pragma once

#include "strata/ds41/kernels/k10_exl3_moe.hpp"
#include "strata/ds41/pack.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strata::ds41 {

/// The GPU's routed experts for m tokens (decode, verify windows, batch steps): K10's contract, computed by K10 or K16
void routed_moe_decode(ExpertFormat f, const __half* x, int m, const int32_t* sel, const float* w, int topk,
                       const kernels::Exl3Expert* experts, float* out, void* workspace, size_t workspace_bytes,
                       cudaStream_t stream);

/// Throws when the GPU's kernel for these experts cannot decode one of them (K19: bitrates outside k19_takes); what
/// names them in the message ("layer", "DSpark stage"). A pack the kernels cannot run is refused at load.
void check_routed_experts(ExpertFormat f, const ExpertSlot* const* slots, int n, int group, const char* what);

/// The prompt's routed experts, rows grouped by expert: K12's contract, computed by K12 (exl3) or K17 (mxfp4)
size_t routed_moe_prefill_workspace_bytes(ExpertFormat f, int max_rows, int max_groups);
void routed_moe_prefill(ExpertFormat f, const __half* x, const int32_t* tok, const float* w, const int32_t* off,
                        int n_groups, const kernels::Exl3Expert* experts, float* out, void* workspace,
                        size_t workspace_bytes, cudaStream_t stream);

/// The CPU's routed experts. A layer registers n experts: expert i's layout is slots[i], its bytes start at bytes[i]
/// (the caller keeps them alive). swiglu_limit: the clamp of DeepSeek's SwiGLU (exl3; mxfp4 has it built in).
int64_t cpu_make_layer(ExpertFormat f, const ExpertSlot* const* slots, const uint8_t* const* bytes, int n,
                       float swiglu_limit);
void cpu_free_layer(ExpertFormat f, int64_t handle);
/// Point expert e of a layer at other bytes with the layout `slot`. Only between forwards of that layer.
void cpu_set_expert(ExpertFormat f, int64_t handle, int e, const ExpertSlot& slot, const uint8_t* bytes);
/// out[t] = the token's routed experts: x [rows][5120] fp16 bits, ids [rows][topk] (-1: none), w [rows][topk],
/// out [rows][5120] FP32 (overwritten). exl3 rounds the weights to fp16 first, as before.
void cpu_forward(ExpertFormat f, int64_t handle, const uint16_t* x, const int32_t* ids, const float* w, float* out,
                 int rows, int topk, int threads);
/// Wake the CPU kernel's helpers before work arrives (exl3; nothing for mxfp4)
void cpu_pool_prime(ExpertFormat f, int threads);

}  // namespace strata::ds41
