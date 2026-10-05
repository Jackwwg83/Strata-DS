// src/ds41/engine.cu - DeepSeek V4.1 Flash decode, one token at a time. See engine.hpp.
//
// Each block of code below restates one method of DeepSeek's model.py (Block, Attention, Compressor, Indexer,
// MoE, Engram, ParallelHead) for one token. Host-side pieces (routing top-k, indexer top-k, n-gram hashes) are
// small and stay on the CPU in M1; the GPU work is ordered on the default stream.
#include "strata/ds41/engine.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/ops.hpp"

#include "moe_mul1.h"   // third_party/exllamav3_moe

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace strata::ds41 {

using ops::bf16;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("ds41 engine: ") + what + ": " + cudaGetErrorString(e));
}

template <typename T>
T* dalloc(size_t n) {
    void* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    ck(cudaMemset(p, 0, n * sizeof(T)), "cudaMemset");
    return (T*) p;
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/// precompute_freqs_cis in fp32, as model.py computes it: [seqlen][32] (cos, sin) pairs
std::vector<float> rope_table(int seqlen, bool yarn) {
    const int dim = kRopeDim;
    const double base = yarn ? kCompressRopeTheta : kRopeTheta;
    std::vector<float> freqs(dim / 2);
    for (int i = 0; i < dim / 2; ++i) freqs[i] = 1.0f / std::pow((float) base, (float) (2 * i) / (float) dim);
    if (yarn) {
        auto corrected = [&](double rot) {
            return dim * std::log(kOriginalSeqLen / (rot * 2 * M_PI)) / (2 * std::log(base));
        };
        const int low = std::max((int) std::floor(corrected(kBetaFast)), 0);
        const int high = std::min((int) std::ceil(corrected(kBetaSlow)), dim - 1);
        for (int i = 0; i < dim / 2; ++i) {
            float ramp = ((float) i - (float) low) / std::max((float) (high - low), 1e-3f);
            ramp = std::min(std::max(ramp, 0.0f), 1.0f);
            const float smooth = 1.0f - ramp;
            freqs[i] = freqs[i] / (float) kRopeFactor * (1.0f - smooth) + freqs[i] * smooth;
        }
    }
    std::vector<float> t((size_t) seqlen * dim);
    for (int p = 0; p < seqlen; ++p)
        for (int i = 0; i < dim / 2; ++i) {
            const float a = (float) p * freqs[i];
            t[(size_t) p * dim + 2 * i] = std::cos(a);
            t[(size_t) p * dim + 2 * i + 1] = std::sin(a);
        }
    return t;
}

float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

}  // namespace

struct Engine::Impl {
    Pack pack;
    int max_seq;
    int cpu_threads;

    struct Fp8 { const uint8_t* w; const uint8_t* s; int64_t n, k; };
    struct Layer {
        Fp8 wq_a, wq_b, wkv, wo_b, sh_w1, sh_w2, sh_w3, idx_wq_b, eng_wkv;
        const bf16 *q_norm, *kv_norm, *wo_a, *attn_norm, *ffn_norm, *gate_w;
        const float *sink, *gate_bias, *hc_attn_fn, *hc_attn_base, *hc_attn_scale, *hc_ffn_fn, *hc_ffn_base,
            *hc_ffn_scale;
        const bf16 *c_wkv = nullptr, *c_wgate = nullptr, *c_norm = nullptr;       // compressor (kv sources)
        const bf16 *idx_wp = nullptr, *idx_wk = nullptr, *idx_knorm = nullptr;     // indexer
        const bf16 *eng_qw = nullptr, *eng_kw = nullptr;                          // engram
        int ratio = 0;
        // state
        bf16* window = nullptr;                    // [128][512]
        bf16* comp = nullptr;                      // [max_seq/ratio][512]   (kv sources)
        bf16* idx_keys = nullptr;                  // [max_seq/ratio][128]   (kv sources)
        float *kv_state = nullptr, *score_state = nullptr;   // [ratio][512]   (kv sources, ratio > 1)
        int64_t moe_handle = -1;
        std::vector<float> gate_bias_host;
    };
    std::vector<Layer> L;
    const bf16 *embed, *head, *final_norm;

    // rope tables on the device: [max_seq][64] floats
    float* rope_plain = nullptr;
    float* rope_yarn = nullptr;

    // engram
    std::vector<int> eng_fd;
    std::vector<int32_t> history;      // compressed token ids fed so far

    // scratch
    bf16 *h, *h2, *xa, *xf, *qr, *q, *kvv, *o, *oa, *attn_out, *latent, *ik, *iq, *iw_raw, *iw, *g, *u, *sh_h,
        *sh_out, *ffn_out, *eng_vals, *eng_kv, *final_x;
    float *act, *pre_mix, *pre, *post, *comb, *mix_scratch, *ffn_pre, *attn_pre, *attn_post, *attn_comb, *ffn_post,
        *ffn_comb, *ckv, *cscore, *scores, *router, *routed, *logits;
    uint16_t* x_half_dev;
    int32_t* idx_dev;
    uint8_t *eng_w_dev, *eng_s_dev;

    // DS41_DEBUG=<file>: per step, the intermediates of layers 1 and 2 (bf16 bits), for bisecting a mismatch
    std::FILE* dbg = nullptr;
    void dbg_write(const bf16* dev, int n) {
        if (!dbg) return;
        std::vector<uint16_t> b(n);
        ck(cudaMemcpy(b.data(), dev, (size_t) n * 2, cudaMemcpyDeviceToHost), "debug dump");
        std::fwrite(b.data(), 2, n, dbg);
    }

    // shared attention state for the current token (SharedAttentionRuntime)
    const bf16* cur_comp = nullptr;
    const bf16* cur_index_k = nullptr;
    std::vector<int32_t> cur_topk;     // compressed part, already offset by kWindow
    std::vector<uint8_t> candidates;   // from the candidate source layer, per compressed position
    std::vector<float> lg;             // logits of the last step

    Impl(const std::string& dir, int max_seq_, int threads) : pack(dir), max_seq(max_seq_), cpu_threads(threads) {}

    Fp8 fp8(const std::string& name) {
        const auto& w = pack.dense(name + ".weight");
        const auto& s = pack.dense(name + ".scale");
        if (w.dtype != DType::F8E4M3 || s.dtype != DType::E8M0) throw std::runtime_error(name + " is not FP8 + E8M0");
        return {(const uint8_t*) w.device, (const uint8_t*) s.device, w.shape[0], w.shape[1]};
    }
    const bf16* bf(const std::string& name) {
        const auto& t = pack.dense(name);
        if (t.dtype != DType::BF16) throw std::runtime_error(name + " is not BF16");
        return (const bf16*) t.device;
    }
    const float* f32(const std::string& name) {
        const auto& t = pack.dense(name);
        if (t.dtype != DType::F32) throw std::runtime_error(name + " is not F32");
        return (const float*) t.device;
    }

    void init() {
        pack.upload_dense();
        pack.map_experts();
        embed = bf("embed.weight");
        head = bf("head.weight");
        final_norm = bf("norm.weight");
        L.resize(kLayers);
        for (int l = 0; l < kLayers; ++l) {
            auto& y = L[l];
            const std::string p = "layers." + std::to_string(l) + ".";
            y.ratio = kCompressRatio[l];
            y.wq_a = fp8(p + "attn.wq_a");
            y.wq_b = fp8(p + "attn.wq_b");
            y.wkv = fp8(p + "attn.wkv");
            y.wo_b = fp8(p + "attn.wo_b");
            y.q_norm = bf(p + "attn.q_norm.weight");
            y.kv_norm = bf(p + "attn.kv_norm.weight");
            y.wo_a = bf(p + "attn.wo_a.weight");
            y.sink = f32(p + "attn.attn_sink");
            y.attn_norm = bf(p + "attn_norm.weight");
            y.ffn_norm = bf(p + "ffn_norm.weight");
            y.hc_attn_fn = f32(p + "hc_attn_fn");
            y.hc_attn_base = f32(p + "hc_attn_base");
            y.hc_attn_scale = f32(p + "hc_attn_scale");
            y.hc_ffn_fn = f32(p + "hc_ffn_fn");
            y.hc_ffn_base = f32(p + "hc_ffn_base");
            y.hc_ffn_scale = f32(p + "hc_ffn_scale");
            y.gate_w = bf(p + "ffn.gate.weight");
            y.gate_bias = f32(p + "ffn.gate.bias");
            y.gate_bias_host.resize(kExperts);
            ck(cudaMemcpy(y.gate_bias_host.data(), y.gate_bias, kExperts * 4, cudaMemcpyDeviceToHost), "bias");
            y.sh_w1 = fp8(p + "ffn.shared_experts.w1");
            y.sh_w2 = fp8(p + "ffn.shared_experts.w2");
            y.sh_w3 = fp8(p + "ffn.shared_experts.w3");
            if (is_kv_source(l)) {
                y.c_wkv = bf(p + "attn.compressor.wkv.weight");
                y.c_norm = bf(p + "attn.compressor.norm.weight");
                if (y.ratio > 1) y.c_wgate = bf(p + "attn.compressor.wgate.weight");
                y.comp = dalloc<bf16>((size_t) (max_seq / y.ratio + 1) * kHeadDim);
                y.idx_keys = dalloc<bf16>((size_t) (max_seq / y.ratio + 1) * kIndexDim);
                if (y.ratio > 1) {
                    y.kv_state = dalloc<float>((size_t) y.ratio * kHeadDim);
                    y.score_state = dalloc<float>((size_t) y.ratio * kHeadDim);
                }
                y.idx_wk = bf(p + "attn.indexer.wk.weight");
                y.idx_knorm = bf(p + "attn.indexer.k_norm.weight");
            }
            if (is_index_source(l)) {
                y.idx_wq_b = fp8(p + "attn.indexer.wq_b");
                y.idx_wp = bf(p + "attn.indexer.weights_proj.weight");
            }
            if (is_engram_layer(l)) {
                y.eng_wkv = fp8(p + "engram.wkv");
                y.eng_qw = bf(p + "engram.q_weight");
                y.eng_kw = bf(p + "engram.k_weight");
            }
            y.window = dalloc<bf16>((size_t) kWindow * kHeadDim);
            register_cpu_experts(l);
        }
        // rope tables
        auto plain = rope_table(max_seq, false), yarn = rope_table(max_seq, true);
        rope_plain = dalloc<float>(plain.size());
        rope_yarn = dalloc<float>(yarn.size());
        ck(cudaMemcpy(rope_plain, plain.data(), plain.size() * 4, cudaMemcpyHostToDevice), "rope");
        ck(cudaMemcpy(rope_yarn, yarn.data(), yarn.size() * 4, cudaMemcpyHostToDevice), "rope");
        // engram files
        for (const auto& t : pack.engram_tables()) {
            const int fd = open(t.path.c_str(), O_RDONLY);
            if (fd < 0) throw std::runtime_error("cannot open engram table " + t.path);
            eng_fd.push_back(fd);
        }
        // scratch
        h = dalloc<bf16>(kHc * kDim);
        h2 = dalloc<bf16>(kHc * kDim);
        xa = dalloc<bf16>(kDim);
        xf = dalloc<bf16>(kDim);
        qr = dalloc<bf16>(kQLora);
        q = dalloc<bf16>(kHeads * kHeadDim);
        kvv = dalloc<bf16>(kHeadDim);
        o = dalloc<bf16>(kHeads * kHeadDim);
        oa = dalloc<bf16>(kOGroups * kOLora);
        attn_out = dalloc<bf16>(kDim);
        latent = dalloc<bf16>(kHeadDim);
        ik = dalloc<bf16>(kIndexDim);
        iq = dalloc<bf16>(kIndexHeads * kIndexDim);
        iw_raw = dalloc<bf16>(kIndexHeads);
        iw = dalloc<bf16>(kIndexHeads);
        g = dalloc<bf16>(kMoeInter);
        u = dalloc<bf16>(kMoeInter);
        sh_h = dalloc<bf16>(kMoeInter);
        sh_out = dalloc<bf16>(kDim);
        ffn_out = dalloc<bf16>(kDim);
        eng_vals = dalloc<bf16>(24 * 256);
        eng_kv = dalloc<bf16>((kHc + 1) * kDim);
        final_x = dalloc<bf16>(kDim);
        act = dalloc<float>(8192);
        pre_mix = dalloc<float>(kHc);
        pre = dalloc<float>(kHc);
        post = dalloc<float>(kHc);
        comb = dalloc<float>(kHc * kHc);
        mix_scratch = dalloc<float>(32);
        attn_pre = dalloc<float>(kHc);
        attn_post = dalloc<float>(kHc);
        attn_comb = dalloc<float>(kHc * kHc);
        ffn_pre = dalloc<float>(kHc);
        ffn_post = dalloc<float>(kHc);
        ffn_comb = dalloc<float>(kHc * kHc);
        ckv = dalloc<float>(kHeadDim);
        cscore = dalloc<float>(kHeadDim);
        scores = dalloc<float>(max_seq + 1);
        router = dalloc<float>(kExperts);
        routed = dalloc<float>(kDim);
        logits = dalloc<float>(kVocab);
        x_half_dev = dalloc<uint16_t>(kDim);
        idx_dev = dalloc<int32_t>(1024);
        eng_w_dev = dalloc<uint8_t>(24 * 256);
        eng_s_dev = dalloc<uint8_t>(24 * 8);
        history.reserve(max_seq);
        if (const char* p = std::getenv("DS41_DEBUG")) dbg = std::fopen(p, "wb");
    }

    void register_cpu_experts(int l) {
        std::vector<MoeCpuMatrixDesc> gate(kExperts), up(kExperts), down(kExperts);
        const uint8_t* base = pack.expert_base();
        for (int e = 0; e < kExperts; ++e) {
            const auto& s = pack.expert(l, e);
            auto desc = [&](int c0, int k_tiles, int n_tiles) {
                MoeCpuMatrixDesc d;
                d.trellis = (const uint16_t*) (base + s.offset + s.comp_off[c0]);
                d.suh = (const at::Half*) (base + s.offset + s.comp_off[c0 + 1]);
                d.svh = (const at::Half*) (base + s.offset + s.comp_off[c0 + 2]);
                d.k_tiles = k_tiles;
                d.n_tiles = n_tiles;
                d.tile_w = (int) (s.comp_bytes[c0] / ((uint64_t) k_tiles * n_tiles * 2));
                return d;
            };
            gate[e] = desc(0, kDim / 16, kMoeInter / 16);   // w1: 5120 -> 2304
            up[e] = desc(4, kDim / 16, kMoeInter / 16);     // w3
            down[e] = desc(8, kMoeInter / 16, kDim / 16);   // w2: 2304 -> 5120
        }
        L[l].moe_handle = exl3_moe_cpu_make_layer_raw(gate.data(), up.data(), down.data(), kExperts, 0, kSwigluLimit, 0);
    }

    const float* rope_at(bool yarn, int pos) const {
        return (yarn ? rope_yarn : rope_plain) + (size_t) pos * kRopeDim;
    }

    void fp8_linear(const bf16* x, const Fp8& w, bf16* y) { ops::fp8_linear(x, w.k, w.w, w.s, w.n, y, act); }

    // ------------------------------------------------------------------------------------- engram
    void engram(int l, int li, int pos) {
        const auto& hs = pack.engram_hash();
        const int n = hs.max_ngram, nh = hs.n_heads, cols = (n - 1) * nh;
        std::vector<int64_t> toks(n);
        for (int s = 0; s < n; ++s) toks[s] = pos - s >= 0 ? history[pos - s] : hs.pad;
        std::vector<int64_t> ids(cols);
        const auto& m = hs.multipliers[li];
        int64_t rolling = toks[0] * m[0];
        for (int i = 1; i < n; ++i) {
            rolling ^= toks[i] * m[i];
            for (int hh = 0; hh < nh; ++hh) {
                const int c = (i - 1) * nh + hh;
                ids[c] = rolling % hs.primes[li][c] + hs.offsets[li][c];
            }
        }
        const auto& t = pack.engram_tables()[li];
        if (t.layer != l) throw std::runtime_error("engram table order does not match engram_hash.txt");
        uint8_t w[24 * 256], s[24 * 8];
        for (int c = 0; c < cols; ++c) {
            if (pread(eng_fd[li], w + c * 256, 256, (off_t) (t.weight_offset + (uint64_t) ids[c] * 256)) != 256 ||
                pread(eng_fd[li], s + c * 8, 8, (off_t) (t.scale_offset + (uint64_t) ids[c] * 8)) != 8)
                throw std::runtime_error("engram row read failed");
        }
        ck(cudaMemcpy(eng_w_dev, w, sizeof w, cudaMemcpyHostToDevice), "engram rows");
        ck(cudaMemcpy(eng_s_dev, s, sizeof s, cudaMemcpyHostToDevice), "engram scales");
        ops::engram_dequant(eng_w_dev, eng_s_dev, cols, eng_vals);
        fp8_linear(eng_vals, L[l].eng_wkv, eng_kv);
        ops::engram_apply(h, eng_kv, L[l].eng_qw, L[l].eng_kw, kNormEps);
    }

    // ------------------------------------------------------------------------------------- indexer
    /// Top-k compressed positions for this layer (Indexer.forward, decode with one query). Returns them offset.
    std::vector<int32_t> indexer(int l, int pos, bool have_latent) {
        auto& y = L[l];
        const int ratio = y.ratio;
        const int t = (pos + 1) / ratio;
        if (is_kv_source(l) && have_latent) {
            ops::bf16_linear(latent, nullptr, y.idx_wk, kHeadDim, kIndexDim, ik, nullptr);
            ops::rmsnorm(ik, y.idx_knorm, ik, kIndexDim, kNormEps);
            ops::rope(ik, 1, kIndexDim, rope_at(true, pos + 1 - ratio), false);
            ops::fp4_quant_inplace(ik, kIndexDim, 32, false);
            ck(cudaMemcpy(y.idx_keys + (size_t) (pos / ratio) * kIndexDim, ik, kIndexDim * 2, cudaMemcpyDeviceToDevice),
               "index key");
        }
        if (is_kv_source(l)) cur_index_k = y.idx_keys;
        fp8_linear(qr, y.idx_wq_b, iq);
        ops::rope(iq, kIndexHeads, kIndexDim, rope_at(true, pos), false);
        ops::fp4_quant_inplace(iq, kIndexHeads * kIndexDim, 32, false);
        ops::bf16_linear(xa, nullptr, y.idx_wp, kDim, kIndexHeads, iw_raw, nullptr);
        ops::scale_bf16(iw_raw, (float) (std::pow(kIndexDim, -0.5) * std::pow(kIndexHeads, -0.5)), iw, kIndexHeads);
        ops::indexer_scores(iq, cur_index_k, t, iw, scores);
        std::vector<float> s(t);
        ck(cudaMemcpy(s.data(), scores, (size_t) t * 4, cudaMemcpyDeviceToHost), "scores");
        if (l == kCandidateLayer) {
            // select_candidate_blocks: best position per block of 8; the newest block is always kept
            const int nb = (t + kCandidateBlock - 1) / kCandidateBlock;
            std::vector<float> bs(nb, -INFINITY);
            for (int i = 0; i < t; ++i) bs[i / kCandidateBlock] = std::max(bs[i / kCandidateBlock], s[i]);
            bs[(t - 1) / kCandidateBlock] = INFINITY;
            std::vector<int> order(nb);
            std::iota(order.begin(), order.end(), 0);
            const int keep = std::min(kCandidateBlocks, nb);
            std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                              [&](int a, int b) { return bs[a] > bs[b] || (bs[a] == bs[b] && a < b); });
            candidates.assign(t, 0);
            for (int i = 0; i < keep; ++i) {
                if (bs[order[i]] == -INFINITY) continue;
                for (int j = order[i] * kCandidateBlock; j < std::min(t, (order[i] + 1) * kCandidateBlock); ++j)
                    candidates[j] = 1;
            }
        } else if (l > kCandidateLayer) {
            for (int i = 0; i < t; ++i)
                if (!candidates[i]) s[i] = -INFINITY;
        }
        const int k = std::min(kIndexTopK, t);
        std::vector<int> order(t);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + k, order.end(),
                          [&](int a, int b) { return s[a] > s[b] || (s[a] == s[b] && a < b); });
        std::vector<int32_t> out(order.begin(), order.begin() + k);
        std::sort(out.begin(), out.end());
        for (auto& v : out) v += kWindow;
        return out;
    }

    // ------------------------------------------------------------------------------------- attention
    void attention(int l, int pos) {
        auto& y = L[l];
        const bool yarn = y.ratio > 0;
        fp8_linear(xa, y.wq_a, qr);
        ops::rmsnorm(qr, y.q_norm, qr, kQLora, kNormEps);
        fp8_linear(qr, y.wq_b, q);
        ops::rope(q, kHeads, kHeadDim, rope_at(yarn, pos), false);
        // sliding window
        fp8_linear(xa, y.wkv, kvv);
        ops::rmsnorm(kvv, y.kv_norm, kvv, kHeadDim, kNormEps);
        ops::rope(kvv, 1, kHeadDim, rope_at(yarn, pos), false);
        ops::act_quant_inplace(kvv, kHeadDim);
        ck(cudaMemcpy(y.window + (size_t) (pos % kWindow) * kHeadDim, kvv, kHeadDim * 2, cudaMemcpyDeviceToDevice),
           "window");
        std::vector<int32_t> idx;
        const int oldest = pos % kWindow + 1;
        for (int i = oldest; i < kWindow; ++i) idx.push_back(i > pos ? -1 : i);
        for (int i = 0; i < oldest; ++i) idx.push_back(i > pos ? -1 : i);
        const bf16* comp = nullptr;
        if (y.ratio > 0) {
            const int ratio = y.ratio;
            const int compress_len = (pos + 1) / ratio;
            bool have_latent = false;
            if (is_kv_source(l)) {
                if (ratio == 1) {
                    ops::bf16_linear(xa, nullptr, y.c_wkv, kDim, kHeadDim, latent, nullptr);
                    ops::rmsnorm(latent, y.c_norm, latent, kHeadDim, kNormEps);
                    have_latent = true;
                } else {
                    const int slot = pos % ratio;
                    ops::bf16_linear(xa, nullptr, y.c_wkv, kDim, kHeadDim, nullptr, y.kv_state + slot * kHeadDim);
                    ops::bf16_linear(xa, nullptr, y.c_wgate, kDim, kHeadDim, nullptr, y.score_state + slot * kHeadDim);
                    if ((pos + 1) % ratio == 0) {
                        ops::compress_pool(y.kv_state, y.score_state, ratio, latent);
                        ops::rmsnorm(latent, y.c_norm, latent, kHeadDim, kNormEps);
                        have_latent = true;
                    }
                }
                cur_comp = y.comp;
            }
            if (is_index_source(l)) {
                cur_topk = compress_len == 0 ? std::vector<int32_t>() : indexer(l, pos, have_latent);
            }
            if (have_latent) {
                ops::rope(latent, 1, kHeadDim, rope_at(true, pos + 1 - ratio), false);
                ops::fp4_quant_inplace(latent, kHeadDim, 16, true);
                ck(cudaMemcpy(y.comp + (size_t) (pos / ratio) * kHeadDim, latent, kHeadDim * 2,
                              cudaMemcpyDeviceToDevice), "compressed kv");
            }
            idx.insert(idx.end(), cur_topk.begin(), cur_topk.end());
            comp = cur_comp;
        }
        ck(cudaMemcpy(idx_dev, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice), "attention index");
        ops::sparse_attn(q, y.window, comp, idx_dev, (int) idx.size(), y.sink, (float) std::pow(kHeadDim, -0.5), o);
        ops::rope(o, kHeads, kHeadDim, rope_at(yarn, pos), true);
        ops::wo_a_grouped(o, y.wo_a, oa);
        fp8_linear(oa, y.wo_b, attn_out);
    }

    // ------------------------------------------------------------------------------------- moe
    void moe(int l, StepDump* dump, Timing& tm) {
        auto& y = L[l];
        ops::bf16_linear(xf, nullptr, y.gate_w, kDim, kExperts, nullptr, router);
        float sc[kExperts];
        ck(cudaMemcpy(sc, router, sizeof sc, cudaMemcpyDeviceToHost), "router");
        float score[kExperts], biased[kExperts];
        for (int e = 0; e < kExperts; ++e) {
            score[e] = std::sqrt(softplus(sc[e]));
            biased[e] = score[e] + y.gate_bias_host[e];
        }
        int sel[kTopK];
        {
            int order[kExperts];
            std::iota(order, order + kExperts, 0);
            std::partial_sort(order, order + kTopK, order + kExperts,
                              [&](int a, int b) { return biased[a] > biased[b] || (biased[a] == biased[b] && a < b); });
            std::copy(order, order + kTopK, sel);
        }
        float w[kTopK], sum = 0.0f;
        for (int i = 0; i < kTopK; ++i) sum += (w[i] = score[sel[i]]);
        for (int i = 0; i < kTopK; ++i) w[i] = w[i] / (sum + 1e-20f) * kRouteScale;
        if (dump) {
            for (int i = 0; i < kTopK; ++i) { dump->routes[l][i] = sel[i]; dump->weights[l][i] = w[i]; }
        }
        // routed experts on the CPU, from the mmap'ed pack
        ops::to_half_fp8q(xf, x_half_dev, kDim);
        uint16_t xh[kDim];
        ck(cudaMemcpy(xh, x_half_dev, sizeof xh, cudaMemcpyDeviceToHost), "expert input");
        const double t0 = now_ms();
        c10::Half wh[kTopK];
        for (int i = 0; i < kTopK; ++i) {
            const __half hv = __float2half_rn(w[i]);
            wh[i] = c10::Half(__half_as_ushort(hv), c10::Half::from_bits());
        }
        float out[kDim];
        exl3_moe_cpu_forward_raw(y.moe_handle, (const at::Half*) xh, sel, wh, out, 1, kTopK, cpu_threads);
        tm.cpu_experts_ms += now_ms() - t0;
        ck(cudaMemcpy(routed, out, sizeof out, cudaMemcpyHostToDevice), "expert output");
        // shared expert
        fp8_linear(xf, y.sh_w1, g);
        fp8_linear(xf, y.sh_w3, u);
        ops::swiglu(g, u, kSwigluLimit, sh_h, kMoeInter);
        fp8_linear(sh_h, y.sh_w2, sh_out);
        ops::add_f32_bf16(routed, sh_out, ffn_out, kDim);
    }

    // ------------------------------------------------------------------------------------- step
    int step(int token, int pos, StepDump* dump, Timing& tm) {
        if (pos != (int) history.size()) throw std::runtime_error("tokens must be fed in order from position 0");
        if (pos >= max_seq) throw std::runtime_error("position past max_seq");
        tm = Timing{};
        const double t_start = now_ms();
        history.push_back(pack.engram_hash().token_map[token]);
        if (dump) {
            dump->hidden.assign((size_t) kLayers * kHc * kDim, 0);
            dump->routes.assign(kLayers, {});
            dump->weights.assign(kLayers, {});
        }
        ops::embed(embed, token, h);
        const float one_hot[kHc] = {1, 0, 0, 0};
        ck(cudaMemcpy(pre_mix, one_hot, sizeof one_hot, cudaMemcpyHostToDevice), "pre_mix");
        int eng_i = 0;
        for (int l = 0; l < kLayers; ++l) {
            auto& y = L[l];
            if (is_engram_layer(l)) {
                const double t0 = now_ms();
                engram(l, eng_i++, pos);
                tm.engram_ms += now_ms() - t0;
            }
            const bool dbg_layer = dbg && (l == 1 || l == 2);
            if (dbg_layer) dbg_write(h, kHc * kDim);                    // block input (after engram)
            // attention sub-block: h -> h2
            ops::hc_mixes(h, y.hc_attn_fn, y.hc_attn_scale, y.hc_attn_base, attn_pre, attn_post, attn_comb, mix_scratch);
            ops::hc_pre(h, pre_mix, xa);
            ops::rmsnorm(xa, y.attn_norm, xa, kDim, kNormEps);
            if (dbg_layer) dbg_write(xa, kDim);                         // attention input
            attention(l, pos);
            if (dbg_layer) dbg_write(attn_out, kDim);                   // attention output
            ops::hc_post(attn_out, h, attn_post, attn_comb, h2);
            // ffn sub-block: h2 -> h
            ops::hc_mixes(h2, y.hc_ffn_fn, y.hc_ffn_scale, y.hc_ffn_base, ffn_pre, ffn_post, ffn_comb, mix_scratch);
            ops::hc_pre(h2, attn_pre, xf);
            ops::rmsnorm(xf, y.ffn_norm, xf, kDim, kNormEps);
            if (dbg_layer) dbg_write(xf, kDim);                         // ffn input
            moe(l, dump, tm);
            if (dbg_layer) dbg_write(ffn_out, kDim);                    // ffn output
            ops::hc_post(ffn_out, h2, ffn_post, ffn_comb, h);
            ck(cudaMemcpy(pre_mix, ffn_pre, kHc * 4, cudaMemcpyDeviceToDevice), "pre_mix");
            if (dump)
                ck(cudaMemcpy(dump->hidden.data() + (size_t) l * kHc * kDim, h, kHc * kDim * 2, cudaMemcpyDeviceToHost),
                   "dump hidden");
        }
        ops::hc_pre(h, pre_mix, final_x);
        ops::rmsnorm(final_x, final_norm, final_x, kDim, kNormEps);
        ops::bf16_linear(final_x, nullptr, head, kDim, kVocab, nullptr, logits);
        lg.resize(kVocab);
        ck(cudaMemcpy(lg.data(), logits, kVocab * 4, cudaMemcpyDeviceToHost), "logits");
        const int best = (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
        if (dump) {
            std::vector<int> order(kVocab);
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + 8, order.end(), [&](int a, int b) { return lg[a] > lg[b]; });
            dump->top_logits.clear();
            for (int i = 0; i < 8; ++i) dump->top_logits.push_back({order[i], lg[order[i]]});
        }
        tm.total_ms = now_ms() - t_start;
        tm.gpu_ms = tm.total_ms - tm.cpu_experts_ms - tm.engram_ms;
        return best;
    }
};

Engine::Engine(const std::string& pack_dir, int max_seq, int cpu_threads)
    : impl_(new Impl(pack_dir, max_seq, cpu_threads)) {
    impl_->init();
}

Engine::~Engine() = default;

int Engine::step(int token, int pos, StepDump* dump) { return impl_->step(token, pos, dump, timing_); }

const std::vector<float>& Engine::last_logits() const { return impl_->lg; }

}  // namespace strata::ds41
