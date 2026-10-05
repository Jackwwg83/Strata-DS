// include/strata/ds41/ops.hpp - the M1 GPU operations for DeepSeek V4.1 Flash decode (one token, batch 1).
//
// M1 is correctness first: every operation restates one line of DeepSeek's model.py (as the prototype runs it
// with proto/torch_kernels.py), including where values are rounded to bf16. Speed comes in M2.
// All pointers are device pointers; all calls are ordered on the default stream.
#pragma once

#include <cuda_bf16.h>
#include <cstdint>

namespace strata::ds41::ops {

using bf16 = __nv_bfloat16;

/// h[c][d] = embed[token][d] for the 4 hyper-connection copies
void embed(const bf16* table, int token, bf16* h);
/// RMSNorm: y = bf16(w * x * rsqrt(mean(x^2) + eps)), statistics in fp32
void rmsnorm(const bf16* x, const bf16* w, bf16* y, int n, float eps);
/// Hyper-connection coefficients from the stream x [4*5120]: pre[4], post[4], comb[4][4] (Sinkhorn)
void hc_mixes(const bf16* x, const float* fn, const float* scale, const float* base,
              float* pre, float* post, float* comb, float* scratch /* >= 25 floats */);
/// y[d] = bf16(sum_j pre[j] * x[j][d])
void hc_pre(const bf16* x, const float* pre, bf16* y);
/// y[k][d] = bf16(post[k] * out[d] + sum_j comb[j][k] * res[j][d]); y must not alias res
void hc_post(const bf16* out, const bf16* res, const float* post, const float* comb, bf16* y);

/// FP8 linear as model.py linear(): activation FP8 block quantized (32, power-of-two scale), weight FP8 E4M3
/// with E8M0 32x32 block scales, FP32 accumulation, BF16 output. `act` is scratch for K floats.
void fp8_linear(const bf16* x, int64_t k, const uint8_t* w, const uint8_t* w_scale, int64_t n, bf16* y,
                float* act);
/// y = x @ W^T with a BF16 weight, FP32 accumulation; output BF16 (yb) or FP32 (yf), whichever is non-null.
/// x_f32 non-null means the input is FP32 (the compressor and the router take x.float()).
void bf16_linear(const bf16* x, const float* x_f32, const bf16* w, int64_t k, int64_t n, bf16* yb, float* yf);

/// RoPE on the last 64 values of each of `n_vec` vectors spaced `stride` apart. cs holds 32 (cos, sin) pairs.
void rope(bf16* v, int n_vec, int stride, const float* cs, bool inverse);
/// FP8 quantize-dequantize in place, blocks of 32, power-of-two scale (act_quant(..., inplace=True))
void act_quant_inplace(bf16* v, int n);
/// FP4 E2M1 quantize-dequantize in place (fp4_act_quant): E4M3 scales (block 16) or E8M0 scales (block 32)
void fp4_quant_inplace(bf16* v, int n, int block, bool e4m3_scale);

/// Decode attention for one query: q [64][512]; kv rows idx < 128 from `window`, others from compressed[idx-128];
/// idx -1 is empty. Softmax with a per-head sink in the denominator; output o [64][512].
void sparse_attn(const bf16* q, const bf16* window, const bf16* compressed, const int32_t* idx, int n_idx,
                 const float* sink, float scale, bf16* o);
/// Grouped low-rank output projection: o [8][4096] x wo_a [8][1024][4096] -> y [8*1024] (BF16, FP32 accumulate)
void wo_a_grouped(const bf16* o, const bf16* wo_a, bf16* y);

/// Indexer scores for one query against t compressed keys: score[t] = bf16(sum_h bf16(relu(bf16(q_h.k_t)) * w_h))
/// q [32][128], keys [t][128], w [32] (already scaled, bf16). Output FP32 holding the bf16 values.
void indexer_scores(const bf16* q, const bf16* keys, int64_t t, const bf16* w, float* score);
/// w[h] = bf16(float(wp[h]) * scale)
void scale_bf16(const bf16* wp, float scale, bf16* w, int n);

/// Shared expert / SwiGLU core: h = bf16(silu(min(g, lim)) * clamp(u, -lim, lim)) from bf16 inputs
void swiglu(const bf16* g, const bf16* u, float lim, bf16* h, int n);
/// y = bf16(a_f32 + float(b)) (MoE output: routed fp32 sum plus the shared expert)
void add_f32_bf16(const float* a, const bf16* b, bf16* y, int n);
/// x_half = fp16(x) after the FP8 activation quantization (the routed-expert input in the prototype)
void to_half_fp8q(const bf16* x, uint16_t* x_half, int n);

/// Compressor pooling of one finished group: out[d] = bf16(sum_r kv[r][d] * softmax_r(score[r][d]))
void compress_pool(const float* kv_state, const float* score_state, int ratio, bf16* out);

/// Engram gate (model.py Engram.forward): h[c] += gate_c * value, gate from the normalized dot of h[c] and key[c]
void engram_apply(bf16* h, const bf16* kv /* [5*5120]: 4 keys then value */, const bf16* qw, const bf16* kw, float eps);
/// Engram rows: FP8 E4M3 [rows][256] times E8M0 [rows][8] -> BF16 [rows][256]
void engram_dequant(const uint8_t* w, const uint8_t* s, int rows, bf16* out);

}  // namespace strata::ds41::ops
