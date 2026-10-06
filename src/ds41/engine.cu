// src/ds41/engine.cu - DeepSeek V4.1 Flash decode, one token at a time. See engine.hpp.
//
// Each block of code below restates one method of DeepSeek's model.py (Block, Attention, Compressor, Indexer,
// MoE, Engram, ParallelHead) for one token. The GPU work is ordered on the default stream. The dense FP8 GEMVs
// (K1), the hyper-connection mixes (K7), the sparse attention (K3), routing (K8) and the indexer top-k (K5) go
// through the task kernels. The routed experts run on a CPU thread that meets the
// stream once per layer through an ExpertDoorbell (upstream's doorbell), so the host thread only enqueues work
// and waits once, for the logits. The engram rows of both engram layers are read at the start of the step.
#include "strata/ds41/engine.hpp"

#include "strata/ds41/config.hpp"
#include "strata/ds41/doorbell.hpp"
#include "strata/ds41/engram_rows.hpp"
#include "strata/ds41/expert_stream.hpp"
#include "strata/ds41/host_experts.hpp"
#include "strata/ds41/lookahead.hpp"
#include "strata/ds41/fp8_gemv.hpp"
#include "strata/ds41/kernels/k12_exl3_moe_prefill.hpp"
#include "strata/ds41/kernels/k13_sparse_attn_prefill.hpp"
#include "strata/ds41/kernels/k14_indexer_prefill.hpp"
#include "strata/ds41/kernels/k2_fp8_gemm.hpp"
#include "strata/ds41/kernels/k3_sparse_attn.hpp"
#include "strata/ds41/kernels/k5_indexer.hpp"
#include "strata/ds41/kernels/k7_hc.hpp"
#include "strata/ds41/kernels/k8_router.hpp"
#include "strata/ds41/ops.hpp"
#include "strata/ds41/prefill_ops.hpp"
#include "strata/ds41/vram_experts.hpp"

#include "moe_mul1.h"   // third_party/exllamav3_moe

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <future>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

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

}  // namespace

struct Engine::Impl {
    Pack pack;
    EngineOptions opt;
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
    };
    std::vector<Layer> L;
    const bf16 *embed, *head, *final_norm;

    // rope tables on the device: [max_seq][64] floats
    float* rope_plain = nullptr;
    float* rope_yarn = nullptr;

    // engram: rows of both engram layers for the current step, read before the step's GPU work
    std::unique_ptr<EngramRows> eng_rows;   // O_DIRECT reads (upstream DirectFile): the rows bypass the file cache
    int n_eng = 0;                     // engram tables
    std::vector<std::vector<int64_t>> eng_ids;   // [table][kEngRows] the rows of the current step
    std::vector<int32_t> history;      // compressed token ids fed so far
    static constexpr int kEngRows = 24;
    uint8_t* eng_host = nullptr;       // pinned: per engram layer, kEngRows*256 weight bytes then kEngRows*8 scales
    uint8_t* eng_dev = nullptr;        // the same on the device

    // routed experts: the CPU thread and its doorbell (round l+1 = layer l)
    std::unique_ptr<ExpertDoorbell> db;
    std::thread worker;
    std::mutex mu;
    std::condition_variable cv;
    uint64_t go = 0;                   // steps released to the worker
    std::atomic<bool> stop{false};
    std::atomic<int64_t> worker_us{0}; // CPU expert time of the current step
    std::atomic<int> worker_misses{0}; // routed experts the CPU computed in the current step
    std::unique_ptr<VramExperts> vram;  // the VRAM tier (null: none)
    std::unique_ptr<HostExperts> host;  // the RAM tier (null: none); the rest is read from the mapped file
    std::atomic<int> worker_ram{0}, worker_file{0}, worker_ssd{0};   // CPU experts of the step by tier
    std::vector<unsigned char> mincore_buf;
    std::unique_ptr<RouterLookahead> lookahead;   // warms the next layer's file-tier experts (DS41_LOOKAHEAD=0: off)
    bool fetch_now = true;             // ask for a layer's missing file pages before computing (DS41_FETCH_NOW=0: off)
    int32_t* gpu_sel = nullptr;        // [6] this layer's resident slots (-1: CPU)
    int32_t* routes_dev = nullptr;     // [40][6]
    float* weights_dev = nullptr;      // [40][6]
    uint8_t* cand_dev = nullptr;       // [max_seq] candidate mask of the candidate layer

    // scratch
    bf16 *h, *h2, *xa, *xf, *qr, *q, *kvv, *o, *oa, *attn_out, *latent, *ik, *iq, *iw_raw, *iw, *g, *u, *sh_h,
        *sh_out, *ffn_out, *eng_vals, *eng_kv, *final_x;
    float *act, *pre_mix, *pre, *post, *comb, *ffn_pre, *attn_pre, *attn_post, *attn_comb, *ffn_post,
        *ffn_comb, *ckv, *cscore, *scores, *routed, *logits;
    uint16_t* x_half_dev;
    int32_t* idx_dev;                  // attention index list: [0, 128) window, then the compressed top-k

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
    std::vector<float> lg;             // logits of the last step

    Impl(const std::string& dir, const EngineOptions& o)
        : pack(dir), opt(o), max_seq(o.max_seq), cpu_threads(o.cpu_threads) {}
    ~Impl() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv.notify_all();
        if (worker.joinable()) worker.join();
        // the lookahead calls into both tiers, and the VRAM tier's copy thread writes into RAM slots: stop them first
        lookahead.reset();
        vram.reset();
        host.reset();
        if (eng_host) cudaFreeHost(eng_host);
        if (pfh.base) cudaFreeHost(pfh.base);
    }

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
        // engram tables
        {
            std::vector<EngramRows::Table> tabs;
            for (const auto& t : pack.engram_tables()) tabs.push_back({t.path, t.weight_offset, t.scale_offset});
            n_eng = (int) tabs.size();
            if (n_eng) eng_rows = std::make_unique<EngramRows>(tabs, kEngRows);
            eng_ids.assign(n_eng, std::vector<int64_t>(kEngRows, 0));
        }
        const size_t eng_bytes = (size_t) n_eng * kEngRows * (256 + 8);
        ck(cudaHostAlloc((void**) &eng_host, std::max<size_t>(eng_bytes, 1), cudaHostAllocDefault), "engram pinned");
        eng_dev = dalloc<uint8_t>(std::max<size_t>(eng_bytes, 1));
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
        attn_pre = dalloc<float>(kHc);
        attn_post = dalloc<float>(kHc);
        attn_comb = dalloc<float>(kHc * kHc);
        ffn_pre = dalloc<float>(kHc);
        ffn_post = dalloc<float>(kHc);
        ffn_comb = dalloc<float>(kHc * kHc);
        ckv = dalloc<float>(kHeadDim);
        cscore = dalloc<float>(kHeadDim);
        scores = dalloc<float>(max_seq + 1);
        routed = dalloc<float>(kDim);
        logits = dalloc<float>(kVocab);
        x_half_dev = dalloc<uint16_t>(kDim);
        idx_dev = dalloc<int32_t>(kWindow + kIndexTopK);
        routes_dev = dalloc<int32_t>(kLayers * kTopK);
        weights_dev = dalloc<float>(kLayers * kTopK);
        cand_dev = dalloc<uint8_t>(max_seq + 1);
        history.reserve(max_seq);
        db = std::make_unique<ExpertDoorbell>(1, kTopK, kDim);
        gpu_sel = dalloc<int32_t>(kTopK);
        // the VRAM expert tier last: an automatic slot count takes what the rest left free
        if (!opt.expert_profile.empty() && opt.vram_expert_slots != 0) {
            VramExperts::Adapt ad;
            ad.every = opt.adapt_every;
            ad.decay = opt.adapt_decay;
            ad.max_swaps = opt.adapt_swaps;
            vram = std::make_unique<VramExperts>(pack, opt.expert_profile, opt.vram_expert_slots,
                                                 opt.vram_reserve_bytes, ad);
            std::fprintf(stderr, "ds41: %d VRAM expert slots (%.2f GiB)\n", vram->slots(),
                         vram->slots() * (double) vram->slot_bytes() / (1ull << 30));
        }
        // the RAM tier after it: the hottest experts the VRAM tier does not hold (upstream's resident budget)
        if (!opt.expert_profile.empty() && opt.ram_budget_gib != 0) {
            const size_t budget = opt.ram_budget_gib < 0 ? auto_ram_budget(4ull << 30)
                                                         : (size_t) (opt.ram_budget_gib * (double) (1ull << 30));
            std::vector<int64_t> handles;
            for (const auto& y : L) handles.push_back(y.moe_handle);
            const double t0 = now_ms();
            host = std::make_unique<HostExperts>(
                pack, read_expert_profile(opt.expert_profile, kLayers, kExperts),
                vram ? vram->res_host() : std::vector<int32_t>((size_t) kLayers * kExperts, -1), budget, handles, 8);
            if (vram) vram->set_host(host.get());
            std::fprintf(stderr, "ds41: RAM tier %d experts (%.1f GiB, %s), filled in %.1f s\n", host->slots(),
                         host->slots() * (double) host->slot_bytes() / (1ull << 30),
                         host->locked() ? "locked" : "not locked", (now_ms() - t0) / 1000.0);
        }
        // the router lookahead: every expert outside the VRAM and RAM tiers is read from the file (upstream turns it
        // on with a RAM budget; here the file tier exists whenever the experts do not all fit in RAM)
        if (const char* f = std::getenv("DS41_FETCH_NOW")) fetch_now = f[0] != '0';
        const char* la_env = std::getenv("DS41_LOOKAHEAD");
        if (!(la_env && la_env[0] == '0')) {
            std::vector<std::vector<uint16_t>> rw(kLayers, std::vector<uint16_t>((size_t) kExperts * kDim));
            std::vector<std::vector<float>> rb(kLayers, std::vector<float>(kExperts));
            for (int l = 0; l < kLayers; ++l) {
                ck(cudaMemcpy(rw[l].data(), L[l].gate_w, rw[l].size() * 2, cudaMemcpyDeviceToHost), "router weights");
                ck(cudaMemcpy(rb[l].data(), L[l].gate_bias, rb[l].size() * 4, cudaMemcpyDeviceToHost), "router bias");
            }
            // The VRAM and RAM tables change only between steps; a prediction racing that change warms one expert
            // more or less, never changes what is computed.
            lookahead = std::make_unique<RouterLookahead>(std::move(rw), std::move(rb), kExperts, kDim, kTopK,
                                                          [this](int l, int e) { return warm_file_expert(l, e); });
        }
        worker = std::thread([this] { worker_loop(); });
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

    /// model.py linear() for one token: K1's activation quantizer, then K1's GEMV (act holds up to 8192 floats)
    void fp8_linear(const bf16* x, const Fp8& w, bf16* y) {
        fp8_quantize_activation_f32((const uint16_t*) x, 1, w.k, act, nullptr);
        fp8_block_gemv_q(act, 1, w.k, w.w, w.s, w.n, (uint16_t*) y, nullptr);
    }

    // ------------------------------------------------------------------------------------- cpu experts
    /// The CPU expert thread: per released step, layers 0..39 in order, each when the GPU publishes it.
    void worker_loop() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop.load() || go != seen; });
                if (stop) return;
                seen = go;
            }
            for (int l = 0; l < kLayers; ++l) {
                if (!db->wait_published(l + 1, stop)) return;
                if (lookahead) lookahead->post(l, db->x());   // predict layer l+1 while this layer computes
                c10::Half wh[kTopK];
                for (int i = 0; i < kTopK; ++i) {
                    const __half hv = __float2half_rn(db->w()[i]);
                    wh[i] = c10::Half(__half_as_ushort(hv), c10::Half::from_bits());
                }
                int misses = 0, n_file = 0;
                int32_t file_ids[kTopK];
                for (int i = 0; i < kTopK; ++i) {
                    const int32_t e = db->ids()[i];
                    if (e < 0) continue;
                    ++misses;
                    if (host && host->slot_of(l, e) >= 0) { ++worker_ram; continue; }
                    ++worker_file;
                    file_ids[n_file++] = e;
                    if (file_pages_missing(l, e)) {
                        ++worker_ssd;
                        // upstream fetches a layer's missing experts in one batch before computing: ask for the whole
                        // range now, so the reads run in parallel instead of page fault by page fault
                        if (fetch_now) warm_file_expert(l, e);
                    }
                }
                if (lookahead) lookahead->observe(l, file_ids, n_file);
                worker_misses += misses;
                const double t0 = now_ms();
                exl3_moe_cpu_forward_raw(L[l].moe_handle, (const at::Half*) db->x(), db->ids(), wh, db->y(), 1, kTopK,
                                         cpu_threads);
                worker_us += (int64_t) ((now_ms() - t0) * 1000.0);
                db->mark_done(l + 1);
            }
        }
    }

    /// Lookahead callback: ask the OS for the pages of (layer, expert) when it is in neither the VRAM nor the RAM tier.
    bool warm_file_expert(int l, int e) {
        if (vram && vram->res_host()[(size_t) l * kExperts + e] >= 0) return false;
        if (host && host->slot_of(l, e) >= 0) return false;
        const ExpertSlot& x = pack.expert(l, e);
        const uintptr_t a = (uintptr_t) (pack.expert_base() + x.offset) & ~(uintptr_t) 4095;
        madvise((void*) a, (uintptr_t) (pack.expert_base() + x.offset + x.bytes) - a, MADV_WILLNEED);
        return true;
    }

    /// True when some page of (layer, expert) in the mapped file is not in RAM: computing it reads the SSD.
    bool file_pages_missing(int l, int e) {
        const ExpertSlot& x = pack.expert(l, e);
        const uintptr_t a = (uintptr_t) (pack.expert_base() + x.offset) & ~(uintptr_t) 4095;
        const size_t len = (uintptr_t) (pack.expert_base() + x.offset + x.bytes) - a;
        mincore_buf.resize((len + 4095) / 4096);
        if (mincore((void*) a, len, mincore_buf.data()) != 0) return false;
        for (unsigned char c : mincore_buf)
            if (!(c & 1)) return true;
        return false;
    }

    // ------------------------------------------------------------------------------------- engram
    /// Hash the n-grams ending at pos for engram layer li: its table rows go to eng_ids[li] (host only).
    void engram_ids(int l, int li, int pos) {
        const auto& hs = pack.engram_hash();
        const int n = hs.max_ngram, nh = hs.n_heads, cols = (n - 1) * nh;
        std::vector<int64_t> toks(n);
        for (int s = 0; s < n; ++s) toks[s] = pos - s >= 0 ? history[pos - s] : hs.pad;
        if (cols > kEngRows) throw std::runtime_error("engram: more n-gram heads than the row buffer holds");
        int64_t* ids = eng_ids[li].data();
        const auto& m = hs.multipliers[li];
        int64_t rolling = toks[0] * m[0];
        for (int i = 1; i < n; ++i) {
            rolling ^= toks[i] * m[i];
            for (int hh = 0; hh < nh; ++hh) {
                const int c = (i - 1) * nh + hh;
                ids[c] = rolling % hs.primes[li][c] + hs.offsets[li][c];
            }
        }
        if (pack.engram_tables()[li].layer != l)
            throw std::runtime_error("engram table order does not match engram_hash.txt");
    }

    /// The rows of every engram table for this step, into the pinned buffer, all reads in flight together.
    void engram_read_all() {
        if (!n_eng) return;
        const auto& hs = pack.engram_hash();
        const int cols = (hs.max_ngram - 1) * hs.n_heads;
        std::vector<const int64_t*> ids;
        std::vector<uint8_t*> w, s;
        for (int li = 0; li < n_eng; ++li) {
            uint8_t* base = eng_host + (size_t) li * kEngRows * (256 + 8);
            ids.push_back(eng_ids[li].data());
            w.push_back(base);
            s.push_back(base + kEngRows * 256);
        }
        eng_rows->read(ids, cols, w, s);
    }

    /// Engram.forward for layer l (engram layer li) from the rows engram_read put on the device.
    void engram(int l, int li) {
        const auto& hs = pack.engram_hash();
        const int cols = (hs.max_ngram - 1) * hs.n_heads;
        const uint8_t* w = eng_dev + (size_t) li * kEngRows * (256 + 8);
        ops::engram_dequant(w, w + kEngRows * 256, cols, eng_vals);
        fp8_linear(eng_vals, L[l].eng_wkv, eng_kv);
        ops::engram_apply(h, eng_kv, L[l].eng_qw, L[l].eng_kw, kNormEps);
    }

    // ------------------------------------------------------------------------------------- indexer
    /// Top-k compressed positions for this layer (Indexer.forward, decode with one query), offset by kWindow,
    /// written after the window part of idx_dev.
    void indexer(int l, int pos, bool have_latent) {
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
        // the candidate layer selects blocks from its own unmasked scores; the layers after it mask with them
        const uint8_t* cand = l > kCandidateLayer ? cand_dev : nullptr;
        kernels::indexer_topk(iq, cur_index_k, t, iw, cand, std::min(kIndexTopK, t), kWindow, scores, idx_dev + kWindow,
                              0);
        if (l == kCandidateLayer) kernels::candidate_blocks(scores, t, kCandidateBlocks, kCandidateBlock, cand_dev, 0);
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
        int n_idx = kWindow;   // idx_dev holds the window part for the whole step (window_index at step start)
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
            if (is_index_source(l) && compress_len > 0) indexer(l, pos, have_latent);
            if (have_latent) {
                ops::rope(latent, 1, kHeadDim, rope_at(true, pos + 1 - ratio), false);
                ops::fp4_quant_inplace(latent, kHeadDim, 16, true);
                ck(cudaMemcpy(y.comp + (size_t) (pos / ratio) * kHeadDim, latent, kHeadDim * 2,
                              cudaMemcpyDeviceToDevice), "compressed kv");
            }
            n_idx += std::min(kIndexTopK, compress_len);   // this group's index source wrote them (none yet: 0)
            comp = cur_comp;
        }
        kernels::sparse_attn_decode(q, y.window, comp, idx_dev, 1, n_idx, y.sink, (float) std::pow(kHeadDim, -0.5), o,
                                    0);
        ops::rope(o, kHeads, kHeadDim, rope_at(yarn, pos), true);
        ops::wo_a_grouped(o, y.wo_a, oa);
        fp8_linear(oa, y.wo_b, attn_out);
    }

    // ------------------------------------------------------------------------------------- moe
    void moe(int l) {
        auto& y = L[l];
        int32_t* ids = routes_dev + l * kTopK;
        float* w = weights_dev + l * kTopK;
        kernels::router_topk(xf, 1, y.gate_w, y.gate_bias, ids, w, 0);
        // routed experts: the misses go to the CPU thread (which reads the mmap'ed pack), the hits to K10
        ops::to_half_fp8q(xf, x_half_dev, kDim);
        const bool tier = vram && vram->slots() > 0;
        db->publish(x_half_dev, ids, w, 1, tier ? vram->res_dev() + (size_t) l * kExperts : nullptr, gpu_sel,
                    (uint32_t) (l + 1), 0);
        ck(cudaMemsetAsync(routed, 0, kDim * sizeof(float), 0), "routed");
        if (tier)
            kernels::exl3_moe_decode((const __half*) x_half_dev, 1, gpu_sel, w, kTopK, vram->experts_dev(), routed,
                                     vram->workspace(), VramExperts::kWorkspaceBytes, 0);
        // shared expert, while the CPU works
        fp8_linear(xf, y.sh_w1, g);
        fp8_linear(xf, y.sh_w3, u);
        ops::swiglu(g, u, kSwigluLimit, sh_h, kMoeInter);
        fp8_linear(sh_h, y.sh_w2, sh_out);
        db->wait_add(routed, 1, (uint32_t) (l + 1), 0);
        ops::add_f32_bf16(routed, sh_out, ffn_out, kDim);
    }

    // ------------------------------------------------------------------------------------- prefill (M3)
    // Layer-major prefill: a pass of S tokens (the whole prompt when it fits; positions p0 .. p0 + S - 1) runs through
    // the layers one at a time. Inside a layer the tokens go in sub-batches of B, in order, through the attention
    // part; then the routed experts run once for all S tokens (each expert copied to the GPU once per pass, not once
    // per sub-batch); then each sub-batch finishes the layer. Every block below is the batched form of the decode
    // code above: the same ops per token, with GEMMs (K2, cuBLAS) instead of GEMVs and the prefill task kernels (K12
    // experts, K13 attention, K14 indexer). It is defined to equal decode token by token (ds41/docs/m3-plan.html,
    // section 0): the window, the compressor state and the indexer's causal limit carry over between sub-batches and
    // passes exactly as between decode steps.
    static constexpr int kStreamAll = 1024;   ///< from this many tokens, every expert of every layer is streamed
    static constexpr int kNllRows = 64;       ///< logits rows per head GEMM when measuring nll
    static constexpr int kMinRing = 16;       ///< the ring shrinks to this many slots before the pass shrinks
    static constexpr int kEngBatch = 256;     ///< tokens per engram read (its O_DIRECT buffers: 16 KiB per row)
    /// engram reads in flight per table: the rows are small random reads, an NVMe needs a deep queue. Measured on
    /// the 7950X + 4090 box, 32K prompt: 16 (the decode default) 14.9 s of GPU wait, 64 2.6 s (1,148 tok/s).
    static constexpr int kEngIoThreads = 64;

    /// Device scratch of one prefill call: per-pass arrays ([cap] tokens) and per-sub-batch arrays ([sub] tokens),
    /// carved from lent VRAM tier slots or cudaMalloc; the ring separately
    struct Prefill {
        int cap = 0, sub = 0;
        uint8_t* lent = nullptr;                 // lent VRAM tier slots (scratch, ring or both)
        uint8_t* own_scratch = nullptr;          // or cudaMalloc
        uint8_t* own_ring = nullptr;
        size_t slot_bytes = 0;
        uint8_t* ring = nullptr;
        int ring_slots = 0;
        // per pass: the residual streams, the expert input and output, routing, indexer results
        bf16 *h, *h2, *xf;
        float *pre_mix, *ffn_pre, *ffn_post, *ffn_comb, *routed, *wts, *rows_w, *nll_out;
        uint16_t* x_half;
        int32_t *tok, *ids, *topk, *rows_tok, *targets;
        uint8_t* cand;
        // per sub-batch
        bf16 *xa, *qr, *q, *o, *oa, *attn_out, *kvv, *iq, *iw_raw, *iw, *g, *u, *sh_h, *sh_out, *ffn_out, *eng_vals,
            *eng_kv, *final_x, *latent, *ik, *attn_kv;
        float *attn_pre, *attn_post, *attn_comb, *ftmp, *router_logits, *ckv, *csc, *nll_logits;
        int32_t* idx;
        uint8_t* eng_dev;
        kernels::Exl3Expert* desc;
        void *k2_ws, *k12_ws, *k14_ws;
        size_t k12_bytes = 0, k14_bytes = 0;
    } pf;
    /// Pinned host buffers of prefill, kept between calls (sized for the largest chunk so far)
    struct PrefillHost {
        uint8_t* base = nullptr;
        int cap = 0;
        int32_t *tok, *ids, *rows_tok, *targets;
        float *wts, *rows_w, *pre, *nll;
        kernels::Exl3Expert* desc;
        uint8_t* eng;   // per engram table: cap * kEngRows * 256 weight bytes, then cap * kEngRows * 8 scale bytes
    } pfh;
    std::vector<std::unique_ptr<EngramRows>> eng_rows_pf;   // one reader per table: read one after the other
    int eng_rows_pf_cap = 0;
    std::vector<std::vector<int64_t>> eng_ids_pf;
    std::unique_ptr<ExpertStream> estream;
    std::vector<int64_t> layer_first_job;   ///< stream-all mode: first job of each layer
    PrefillTiming* ptm = nullptr;

    template <typename T>
    static T* carve(uint8_t* base, size_t& used, size_t n) {
        used = (used + 255) & ~(size_t) 255;
        T* p = base ? (T*) (base + used) : nullptr;
        used += n * sizeof(T);
        return p;
    }


    /// Lays the scratch out from `base` (null: only counts) for passes of `cap` tokens and sub-batches of `sub`.
    /// Returns the bytes. The ring is separate.
    size_t layout(Prefill& p, uint8_t* base, int cap, int sub) {
        size_t u = 0;
        const size_t c = (size_t) cap, b = (size_t) sub;
        int64_t max_k = 0;
        for (const auto& y : L)
            for (const Fp8* f : {&y.wq_a, &y.wq_b, &y.wkv, &y.wo_b, &y.sh_w1, &y.sh_w2, &y.sh_w3, &y.idx_wq_b, &y.eng_wkv})
                if (f->w) max_k = std::max(max_k, f->k);
        // per pass
        p.h = carve<bf16>(base, u, c * kHc * kDim);
        p.h2 = carve<bf16>(base, u, c * kHc * kDim);
        p.xf = carve<bf16>(base, u, c * kDim);
        p.x_half = carve<uint16_t>(base, u, c * kDim);
        p.routed = carve<float>(base, u, c * kDim);
        p.pre_mix = carve<float>(base, u, c * kHc);
        p.ffn_pre = carve<float>(base, u, c * kHc);
        p.ffn_post = carve<float>(base, u, c * kHc);
        p.ffn_comb = carve<float>(base, u, c * kHc * kHc);
        p.wts = carve<float>(base, u, c * kTopK);
        p.rows_w = carve<float>(base, u, c * kTopK);
        p.nll_out = carve<float>(base, u, c);
        p.tok = carve<int32_t>(base, u, c);
        p.ids = carve<int32_t>(base, u, c * kTopK);
        p.topk = carve<int32_t>(base, u, c * kIndexTopK);
        p.rows_tok = carve<int32_t>(base, u, c * kTopK);
        p.targets = carve<int32_t>(base, u, c);
        p.cand = carve<uint8_t>(base, u, c * (size_t) max_seq);
        // per sub-batch
        p.xa = carve<bf16>(base, u, b * kDim);
        p.qr = carve<bf16>(base, u, b * kQLora);
        p.q = carve<bf16>(base, u, b * kHeads * kHeadDim);
        p.o = carve<bf16>(base, u, b * kHeads * kHeadDim);
        p.oa = carve<bf16>(base, u, b * kOGroups * kOLora);
        p.attn_out = carve<bf16>(base, u, b * kDim);
        p.kvv = carve<bf16>(base, u, b * kHeadDim);
        p.iq = carve<bf16>(base, u, b * kIndexHeads * kIndexDim);
        p.iw_raw = carve<bf16>(base, u, b * kIndexHeads);
        p.iw = carve<bf16>(base, u, b * kIndexHeads);
        p.g = carve<bf16>(base, u, b * kMoeInter);
        p.u = carve<bf16>(base, u, b * kMoeInter);
        p.sh_h = carve<bf16>(base, u, b * kMoeInter);
        p.sh_out = carve<bf16>(base, u, b * kDim);
        p.ffn_out = carve<bf16>(base, u, b * kDim);
        p.eng_vals = carve<bf16>(base, u, b * kEngRows * 256);
        p.eng_kv = carve<bf16>(base, u, b * (kHc + 1) * kDim);
        p.final_x = carve<bf16>(base, u, b * kDim);
        p.latent = carve<bf16>(base, u, (b + 1) * kHeadDim);
        p.ik = carve<bf16>(base, u, (b + 1) * kIndexDim);
        p.attn_kv = carve<bf16>(base, u, ((size_t) max_seq + kWindow + b) * kHeadDim);
        p.attn_pre = carve<float>(base, u, b * kHc);
        p.attn_post = carve<float>(base, u, b * kHc);
        p.attn_comb = carve<float>(base, u, b * kHc * kHc);
        p.ftmp = carve<float>(base, u, b * kOGroups * kOLora);
        p.router_logits = carve<float>(base, u, b * kExperts);
        p.ckv = carve<float>(base, u, (b + 2) * kHeadDim);
        p.csc = carve<float>(base, u, (b + 2) * kHeadDim);
        p.nll_logits = carve<float>(base, u, (size_t) kNllRows * kVocab);
        p.idx = carve<int32_t>(base, u, b * (kWindow + kIndexTopK));
        p.eng_dev = carve<uint8_t>(base, u, std::max<size_t>((size_t) n_eng * b * kEngRows * (256 + 8), 1));
        p.desc = carve<kernels::Exl3Expert>(base, u, 2 * kExperts);
        p.k2_ws = carve<uint8_t>(base, u, b * (size_t) max_k * 4);
        p.k12_bytes = kernels::exl3_moe_prefill_workspace_bytes(cap * kTopK, kExperts);
        p.k12_ws = carve<uint8_t>(base, u, p.k12_bytes);
        p.k14_bytes = kernels::indexer_topk_prefill_workspace_bytes(sub, max_seq);
        p.k14_ws = carve<uint8_t>(base, u, p.k14_bytes);
        return (u + 255) & ~(size_t) 255;
    }

    /// Chooses the pass and sub-batch sizes and gets the scratch and the ring: lent VRAM tier slots (up to 90% of the
    /// tier, as upstream) and free VRAM, in that order of preference. Larger passes first (each pass copies every
    /// expert to the GPU once, so the pass size decides the prefill speed of long prompts); for a pass, the largest
    /// sub-batch (up to opt.prefill_batch) and ring (opt.prefill_ring, at least 16 slots) that fit.
    void prefill_begin(int n) {
        pf = Prefill{};
        if (vram) pf.slot_bytes = vram->slot_bytes();
        else {
            for (int l = 0; l < kLayers; ++l)
                for (int e = 0; e < kExperts; ++e) pf.slot_bytes = std::max<size_t>(pf.slot_bytes, pack.expert(l, e).bytes);
            pf.slot_bytes = (pf.slot_bytes + 255) & ~(size_t) 255;
        }
        const int lendable = vram ? vram->slots() * 9 / 10 : 0;
        size_t free_b = 0, total_b = 0;
        ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        const size_t spare = free_b > (512ull << 20) ? free_b - (512ull << 20) : 0;   // cuBLAS and K13 scratch
        auto slots_for = [&](size_t b) { return (int) ((b + pf.slot_bytes - 1) / pf.slot_bytes); };
        size_t scratch = 0, ring = 0;
        int plan = -1;   // 0: both lent; 1: scratch lent, ring cudaMalloc; 2: ring lent, scratch cudaMalloc; 3: both cudaMalloc
        int cap = std::max(1, std::min(n, opt.prefill_chunk)), sub = 0;
        for (; plan < 0; cap /= 2) {
            if (cap < 16) throw std::runtime_error("ds41 prefill: not enough VRAM for a 16-token pass");
            for (sub = std::max(1, std::min(cap, opt.prefill_batch)); plan < 0 && sub >= std::min(cap, 512); sub /= 2) {
                scratch = layout(pf, nullptr, cap, sub);
                for (int r = std::max(opt.prefill_ring, kMinRing); plan < 0 && r >= kMinRing; r /= 2) {
                    pf.ring_slots = r;
                    ring = (size_t) r * pf.slot_bytes;
                    if (slots_for(scratch + ring) <= lendable) plan = 0;
                    else if (slots_for(scratch) <= lendable && ring <= spare) plan = 1;
                    else if (slots_for(ring) <= lendable && scratch <= spare) plan = 2;
                    else if (scratch + ring <= spare) plan = 3;
                }
                if (plan >= 0) break;
            }
            if (plan >= 0) break;
        }
        pf.cap = cap;
        pf.sub = sub;
        const size_t lend_bytes = plan == 0 ? scratch + ring : plan == 1 ? scratch : plan == 2 ? ring : 0;
        if (lend_bytes) pf.lent = vram->lend(slots_for(lend_bytes));
        uint8_t* scratch_base = plan <= 1 ? pf.lent : nullptr;
        if (plan >= 2) ck(cudaMalloc((void**) &pf.own_scratch, scratch), "prefill scratch");
        if (plan == 1 || plan == 3) ck(cudaMalloc((void**) &pf.own_ring, ring), "prefill ring");
        if (!scratch_base) scratch_base = pf.own_scratch;
        layout(pf, scratch_base, cap, sub);
        pf.ring = plan == 0 ? pf.lent + scratch : plan == 2 ? pf.lent : pf.own_ring;
        std::fprintf(stderr, "ds41 prefill: pass %d tokens, sub-batch %d; scratch %.2f GiB %s; ring %d slots %s; %d tier "
                     "slots lent\n", cap, sub, scratch / 1073741824.0, plan <= 1 ? "in lent slots" : "cudaMalloc",
                     pf.ring_slots, plan == 0 || plan == 2 ? "in lent slots" : "cudaMalloc",
                     lend_bytes ? slots_for(lend_bytes) : 0);
        // pinned host buffers
        if (pfh.cap < cap) {
            if (pfh.base) cudaFreeHost(pfh.base);
            size_t u = 0;
            auto lay = [&](uint8_t* b) {
                u = 0;
                const size_t c = (size_t) cap;
                pfh.tok = carve<int32_t>(b, u, c);
                pfh.ids = carve<int32_t>(b, u, c * kTopK);
                pfh.rows_tok = carve<int32_t>(b, u, c * kTopK);
                pfh.targets = carve<int32_t>(b, u, c);
                pfh.wts = carve<float>(b, u, c * kTopK);
                pfh.rows_w = carve<float>(b, u, c * kTopK);
                pfh.pre = carve<float>(b, u, c * kHc);
                pfh.nll = carve<float>(b, u, c);
                pfh.desc = carve<kernels::Exl3Expert>(b, u, 2 * kExperts);
                pfh.eng = carve<uint8_t>(b, u, std::max<size_t>((size_t) n_eng * c * kEngRows * (256 + 8), 1));
            };
            lay(nullptr);
            ck(cudaHostAlloc((void**) &pfh.base, u, cudaHostAllocDefault), "prefill pinned buffers");
            lay(pfh.base);
            pfh.cap = cap;
            for (int i = 0; i < cap; ++i)
                for (int j = 0; j < kHc; ++j) pfh.pre[i * kHc + j] = j == 0 ? 1.0f : 0.0f;
        }
        if (n_eng && eng_rows_pf.empty())
            for (const auto& t : pack.engram_tables())
                eng_rows_pf.push_back(std::make_unique<EngramRows>(
                    std::vector<EngramRows::Table>{{t.path, t.weight_offset, t.scale_offset}}, kEngBatch * kEngRows,
                    256, 8, kEngIoThreads));
        if (n_eng && eng_rows_pf_cap < cap) {
            eng_rows_pf_cap = cap;
            eng_ids_pf.assign(n_eng, std::vector<int64_t>((size_t) cap * kEngRows, 0));
        }
        estream = std::make_unique<ExpertStream>(pack, host.get(), pf.ring, pf.ring_slots, pf.slot_bytes,
                                                 std::max(1, opt.prefill_threads), std::max(1, opt.prefill_host_buffers));
    }


    /// Returns the lent slots (or frees the scratch). After an error the stream is stopped without draining (its
    /// unreleased jobs would never complete).
    void prefill_end() {
        estream.reset();
        ck(cudaDeviceSynchronize(), "prefill end");
        if (pf.own_scratch) cudaFree(pf.own_scratch);
        if (pf.own_ring) cudaFree(pf.own_ring);
        if (pf.lent && vram) vram->restore();
        pf.lent = pf.own_scratch = pf.own_ring = nullptr;
    }

    bool resident(int l, int e) const { return vram && vram->res_host()[(size_t) l * kExperts + e] >= 0; }

    /// model.py linear() for T rows (K2)
    void fp8_rows(const bf16* x, int T, const Fp8& w, bf16* y) {
        kernels::fp8_block_gemm(x, T, w.k, w.w, w.s, w.n, y, pf.k2_ws, 0);
    }

    /// K7 for T rows, 8 per call
    void hc_rows(const bf16* x, int T, const float* fn, const float* scale, const float* base, const float* pre_in,
                 bf16* y, float* pre, float* post, float* comb) {
        for (int t = 0; t < T; t += 8)
            kernels::hc_mixes_pre(x + (size_t) t * kHc * kDim, std::min(8, T - t), fn, scale, base, pre_in + t * kHc,
                                  y + (size_t) t * kDim, pre + t * kHc, post + t * kHc, comb + t * kHc * kHc, 0);
    }

    /// The compressed entries of kv source l completed in this chunk (decode: attention(), compressor part), their
    /// indexer keys, and the compressor state left for decode.
    void compress_rows(int l, int T, int p0) {
        auto& y = L[l];
        const int r = y.ratio;
        int G, g0;
        if (r == 1) {
            prefill::bf16_gemm(pf.xa, y.c_wkv, T, kDim, kHeadDim, pf.latent, nullptr, pf.ftmp);
            ops::rmsnorm(pf.latent, y.c_norm, pf.latent, kHeadDim, kNormEps, T);
            G = T;
            g0 = p0;
        } else {
            // the rows of the group in progress (decode's state slots 0 .. carry-1), then this chunk's rows
            const int carry = p0 % r;
            if (carry) {
                ck(cudaMemcpyAsync(pf.ckv, y.kv_state, (size_t) carry * kHeadDim * 4, cudaMemcpyDeviceToDevice, 0), "carry");
                ck(cudaMemcpyAsync(pf.csc, y.score_state, (size_t) carry * kHeadDim * 4, cudaMemcpyDeviceToDevice, 0),
                   "carry");
            }
            prefill::bf16_gemm(pf.xa, y.c_wkv, T, kDim, kHeadDim, nullptr, pf.ckv + (size_t) carry * kHeadDim);
            prefill::bf16_gemm(pf.xa, y.c_wgate, T, kDim, kHeadDim, nullptr, pf.csc + (size_t) carry * kHeadDim);
            G = (carry + T) / r;
            g0 = (p0 - carry) / r;
            ops::compress_pool(pf.ckv, pf.csc, r, pf.latent, G);
            ops::rmsnorm(pf.latent, y.c_norm, pf.latent, kHeadDim, kNormEps, G);
            const int rem = (carry + T) % r;   // the next group's first rows: decode's state slots 0 .. rem-1
            if (rem) {
                ck(cudaMemcpyAsync(y.kv_state, pf.ckv + (size_t) G * r * kHeadDim, (size_t) rem * kHeadDim * 4,
                                   cudaMemcpyDeviceToDevice, 0), "state");
                ck(cudaMemcpyAsync(y.score_state, pf.csc + (size_t) G * r * kHeadDim, (size_t) rem * kHeadDim * 4,
                                   cudaMemcpyDeviceToDevice, 0), "state");
            }
        }
        if (G > 0) {
            // indexer keys (decode: indexer(), have_latent), from the latent before its RoPE
            prefill::bf16_gemm(pf.latent, y.idx_wk, G, kHeadDim, kIndexDim, pf.ik, nullptr, pf.ftmp);
            ops::rmsnorm(pf.ik, y.idx_knorm, pf.ik, kIndexDim, kNormEps, G);
            prefill::rope_rows(pf.ik, G, 1, kIndexDim, rope_yarn, g0 * r, r, false);
            ops::fp4_quant_inplace(pf.ik, G * kIndexDim, 32, false);
            ck(cudaMemcpyAsync(y.idx_keys + (size_t) g0 * kIndexDim, pf.ik, (size_t) G * kIndexDim * 2,
                               cudaMemcpyDeviceToDevice, 0), "index keys");
            prefill::rope_rows(pf.latent, G, 1, kHeadDim, rope_yarn, g0 * r, r, false);
            ops::fp4_quant_inplace(pf.latent, G * kHeadDim, 16, true);
            ck(cudaMemcpyAsync(y.comp + (size_t) g0 * kHeadDim, pf.latent, (size_t) G * kHeadDim * 2,
                               cudaMemcpyDeviceToDevice, 0), "compressed kv");
        }
        cur_comp = y.comp;
        cur_index_k = y.idx_keys;
    }

    /// Indexer of index source l for the T queries of the sub-batch at pass row b0 (decode: indexer()): their top-512
    /// compressed rows into pf.topk; the candidate layer writes each query's candidate mask, the later layers mask
    /// with it (both per pass, at row b0).
    void indexer_rows(int l, int T, int p0, int b0) {
        auto& y = L[l];
        fp8_rows(pf.qr, T, y.idx_wq_b, pf.iq);
        prefill::rope_rows(pf.iq, T, kIndexHeads, kIndexDim, rope_yarn, p0, 1, false);
        ops::fp4_quant_inplace(pf.iq, T * kIndexHeads * kIndexDim, 32, false);
        prefill::bf16_gemm(pf.xa, y.idx_wp, T, kDim, kIndexHeads, pf.iw_raw, nullptr, pf.ftmp);
        ops::scale_bf16(pf.iw_raw, (float) (std::pow(kIndexDim, -0.5) * std::pow(kIndexHeads, -0.5)), pf.iw,
                        T * kIndexHeads);
        uint8_t* cand = pf.cand + (size_t) b0 * max_seq;
        kernels::indexer_topk_prefill(pf.iq, cur_index_k, pf.iw, T, p0, y.ratio, l > kCandidateLayer ? cand : nullptr,
                                      l == kCandidateLayer ? cand : nullptr, max_seq, kIndexTopK, 0, kCandidateBlocks,
                                      kCandidateBlock, pf.topk + (size_t) b0 * kIndexTopK, pf.k14_ws, pf.k14_bytes, 0);
    }

    /// decode: attention(), for the T rows of the sub-batch at pass row b0 (positions p0 ..). One KV buffer per layer
    /// and sub-batch: [compressed rows][window rows for positions p0 - 127 .. p0 + T - 1] (K13's layout).
    void attention_rows(int l, int T, int p0, int b0) {
        auto& y = L[l];
        const bool yarn = y.ratio > 0;
        const float* table = yarn ? rope_yarn : rope_plain;
        fp8_rows(pf.xa, T, y.wq_a, pf.qr);
        ops::rmsnorm(pf.qr, y.q_norm, pf.qr, kQLora, kNormEps, T);
        fp8_rows(pf.qr, T, y.wq_b, pf.q);
        prefill::rope_rows(pf.q, T, kHeads, kHeadDim, table, p0, 1, false);
        fp8_rows(pf.xa, T, y.wkv, pf.kvv);
        ops::rmsnorm(pf.kvv, y.kv_norm, pf.kvv, kHeadDim, kNormEps, T);
        prefill::rope_rows(pf.kvv, T, 1, kHeadDim, table, p0, 1, false);
        ops::act_quant_inplace(pf.kvv, T * kHeadDim);
        int win_base = 0, n_idx = kWindow;
        const int32_t* topk = nullptr;
        if (y.ratio > 0) {
            if (is_kv_source(l)) compress_rows(l, T, p0);
            if (is_index_source(l)) indexer_rows(l, T, p0, b0);
            const int c_end = (p0 + T) / y.ratio;   // compressed rows any query of the chunk may see
            if (c_end)
                ck(cudaMemcpyAsync(pf.attn_kv, cur_comp, (size_t) c_end * kHeadDim * 2, cudaMemcpyDeviceToDevice, 0),
                   "attention kv: compressed rows");
            win_base = c_end;
            n_idx = kWindow + kIndexTopK;
            topk = pf.topk + (size_t) b0 * kIndexTopK;
        }
        // window rows: the ring's positions before the chunk, then the chunk's own; then the ring for decode
        const int prev = std::min(p0, kWindow - 1);
        prefill::window_gather(y.window, p0, prev, pf.attn_kv + (size_t) (win_base + kWindow - 1 - prev) * kHeadDim);
        ck(cudaMemcpyAsync(pf.attn_kv + (size_t) (win_base + kWindow - 1) * kHeadDim, pf.kvv, (size_t) T * kHeadDim * 2,
                           cudaMemcpyDeviceToDevice, 0), "attention kv: window rows");
        prefill::window_scatter(y.window, pf.kvv, p0, T);
        prefill::attn_index_rows(T, p0, win_base, topk, kIndexTopK, pf.idx, n_idx);
        kernels::sparse_attn_prefill(pf.q, pf.attn_kv, pf.idx, T, n_idx, y.sink, (float) std::pow(kHeadDim, -0.5), pf.o,
                                     0);
        prefill::rope_rows(pf.o, T, kHeads, kHeadDim, table, p0, 1, true);
        prefill::wo_a_grouped_rows(pf.o, y.wo_a, T, pf.oa, pf.ftmp);
        fp8_rows(pf.oa, T, y.wo_b, pf.attn_out);
    }


    /// decode: moe(), routing part, for the sub-batch at pass row b0: router (GPU logits), the routes into the pass
    /// arrays, and the experts' input (FP8-quantized, fp16).
    void moe_route(int l, int T, int b0) {
        auto& y = L[l];
        const bf16* xf = pf.xf + (size_t) b0 * kDim;
        prefill::bf16_gemm(xf, y.gate_w, T, kDim, kExperts, nullptr, pf.router_logits);
        prefill::route_rows(pf.router_logits, y.gate_bias, T, pf.ids + (size_t) b0 * kTopK, pf.wts + (size_t) b0 * kTopK);
        ops::to_half_fp8q(xf, pf.x_half + (size_t) b0 * kDim, T * kDim);
    }

    /// decode: moe(), routed experts, for all S tokens of the pass at once: the host sorts the (token, expert) rows
    /// by expert, VRAM tier experts first (one K12 call from their slots), then the streamed experts in job order
    /// (one K12 call per group of ring slots, each released when its call has run). Each expert is copied once.
    void moe_experts(int l, int S) {
        ck(cudaMemcpyAsync(pfh.ids, pf.ids, (size_t) S * kTopK * 4, cudaMemcpyDeviceToHost, 0), "routes down");
        ck(cudaMemcpyAsync(pfh.wts, pf.wts, (size_t) S * kTopK * 4, cudaMemcpyDeviceToHost, 0), "weights down");
        ck(cudaMemsetAsync(pf.routed, 0, (size_t) S * kDim * 4, 0), "routed");
        ck(cudaStreamSynchronize(0), "routes");
        std::vector<int> count(kExperts, 0);
        for (int i = 0; i < S * kTopK; ++i) count[pfh.ids[i]]++;
        std::vector<int> order;
        for (int e = 0; e < kExperts; ++e)
            if (count[e] && resident(l, e)) order.push_back(e);
        const int n_res = (int) order.size();
        int64_t first_job;
        int n_jobs = 0;
        if (S >= kStreamAll) {
            first_job = layer_first_job[l];
            n_jobs = (int) (layer_first_job[l + 1] - first_job);
            for (int e = 0; e < kExperts; ++e)
                if (!resident(l, e)) order.push_back(e);
        } else {
            std::vector<std::pair<int, int>> jobs;
            for (int e = 0; e < kExperts; ++e)
                if (count[e] && !resident(l, e)) {
                    jobs.push_back({l, e});
                    order.push_back(e);
                }
            n_jobs = (int) jobs.size();
            first_job = n_jobs ? estream->push(jobs) : 0;
        }
        std::vector<int> start(kExperts, 0), fill(kExperts, 0);
        std::vector<int32_t> off(order.size() + 1, 0);
        for (size_t i = 0; i < order.size(); ++i) {
            start[order[i]] = off[i];
            off[i + 1] = off[i] + count[order[i]];
        }
        for (int t = 0; t < S; ++t)
            for (int j = 0; j < kTopK; ++j) {
                const int e = pfh.ids[t * kTopK + j];
                const int r = start[e] + fill[e]++;
                pfh.rows_tok[r] = t;
                pfh.rows_w[r] = pfh.wts[t * kTopK + j];
            }
        ck(cudaMemcpyAsync(pf.rows_tok, pfh.rows_tok, (size_t) S * kTopK * 4, cudaMemcpyHostToDevice, 0), "rows up");
        ck(cudaMemcpyAsync(pf.rows_w, pfh.rows_w, (size_t) S * kTopK * 4, cudaMemcpyHostToDevice, 0), "rows up");
        const auto* xh = (const __half*) pf.x_half;
        if (n_res) {
            for (int i = 0; i < n_res; ++i) pfh.desc[i] = vram->desc(vram->res_host()[(size_t) l * kExperts + order[i]]);
            ck(cudaMemcpyAsync(pf.desc, pfh.desc, n_res * sizeof(kernels::Exl3Expert), cudaMemcpyHostToDevice, 0),
               "descriptors up");
            kernels::exl3_moe_prefill(xh, pf.rows_tok, pf.rows_w, off.data(), n_res, pf.desc, pf.routed, pf.k12_ws,
                                      pf.k12_bytes, 0);
            ptm->vram_experts += n_res;
        }
        const int group = std::max(1, pf.ring_slots / 4);
        for (int g0 = 0; g0 < n_jobs; g0 += group) {
            const int n = std::min(group, n_jobs - g0);
            for (int i = 0; i < n; ++i) {
                uint8_t* slot = estream->wait(first_job + g0 + i, 0);   // stream 0 waits for the copy on the GPU
                pfh.desc[kExperts + g0 + i] = VramExperts::describe_at(pack, l, order[n_res + g0 + i], slot);
            }
            ck(cudaMemcpyAsync(pf.desc + kExperts + g0, pfh.desc + kExperts + g0, n * sizeof(kernels::Exl3Expert),
                               cudaMemcpyHostToDevice, 0), "descriptors up");
            kernels::exl3_moe_prefill(xh, pf.rows_tok, pf.rows_w, off.data() + n_res + g0, n, pf.desc + kExperts + g0,
                                      pf.routed, pf.k12_ws, pf.k12_bytes, 0);
            for (int i = 0; i < n; ++i) estream->release(first_job + g0 + i, 0);
        }
        ptm->streamed += n_jobs;
    }

    /// decode: moe() end and the ffn sub-block's hc_post, for the sub-batch at pass row b0: the shared expert, plus
    /// the routed sum, into the residual stream h.
    void moe_finish(int l, int T, int b0) {
        auto& y = L[l];
        const bf16* xf = pf.xf + (size_t) b0 * kDim;
        fp8_rows(xf, T, y.sh_w1, pf.g);
        fp8_rows(xf, T, y.sh_w3, pf.u);
        ops::swiglu(pf.g, pf.u, kSwigluLimit, pf.sh_h, T * kMoeInter);
        fp8_rows(pf.sh_h, T, y.sh_w2, pf.sh_out);
        ops::add_f32_bf16(pf.routed + (size_t) b0 * kDim, pf.sh_out, pf.ffn_out, T * kDim);
        ops::hc_post(pf.ffn_out, pf.h2 + (size_t) b0 * kHc * kDim, pf.ffn_post + b0 * kHc, pf.ffn_comb + b0 * kHc * kHc,
                     pf.h + (size_t) b0 * kHc * kDim, T);
        ck(cudaMemcpyAsync(pf.pre_mix + b0 * kHc, pf.ffn_pre + b0 * kHc, (size_t) T * kHc * 4, cudaMemcpyDeviceToDevice,
                           0), "pre_mix");
    }

    /// The engram rows of a pass, table by table (engram layer order), into the pinned buffer in token order. A row
    /// that several positions share is read once. Runs on its own thread while the GPU works; done[t] is set when
    /// table t is in the buffer (or carries the read error).
    void engram_pass(int S, int p0, std::vector<std::promise<void>>& done, double& ms) {
        const double t0 = now_ms();
        size_t t = 0;
        try {
            const auto& hs = pack.engram_hash();
            const int cols = (hs.max_ngram - 1) * hs.n_heads;
            if (cols != kEngRows) throw std::runtime_error("engram: the prefill buffers assume 24 rows per token");
            const size_t eng_table = (size_t) pf.cap * kEngRows * (256 + 8);
            for (int l = 0; l < kLayers; ++l) {
                if (!is_engram_layer(l)) continue;
                const int li = (int) t;
                std::vector<int64_t>& all = eng_ids_pf[li];
                for (int i = 0; i < S; ++i) {
                    engram_ids(l, li, p0 + i);
                    std::copy(eng_ids[li].begin(), eng_ids[li].begin() + cols, all.begin() + (size_t) i * cols);
                }
                const size_t n = (size_t) S * cols;
                std::vector<int64_t> uniq(all.begin(), all.begin() + n);
                std::sort(uniq.begin(), uniq.end());
                uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
                std::vector<uint8_t> uw(uniq.size() * 256), us(uniq.size() * 8);
                const size_t batch = (size_t) kEngBatch * kEngRows;
                for (size_t r = 0; r < uniq.size(); r += batch)
                    eng_rows_pf[li]->read({uniq.data() + r}, (int) std::min(batch, uniq.size() - r), {uw.data() + r * 256},
                                          {us.data() + r * 8});
                uint8_t* w = pfh.eng + (size_t) li * eng_table;
                uint8_t* sc = w + (size_t) pf.cap * kEngRows * 256;
                for (size_t i = 0; i < n; ++i) {
                    const size_t u = (size_t) (std::lower_bound(uniq.begin(), uniq.end(), all[i]) - uniq.begin());
                    std::memcpy(w + i * 256, uw.data() + u * 256, 256);
                    std::memcpy(sc + i * 8, us.data() + u * 8, 8);
                }
                ptm->engram_rows += (int64_t) n;
                ptm->engram_unique += (int64_t) uniq.size();
                done[li].set_value();
                ++t;
            }
        } catch (...) {
            for (; t < done.size(); ++t) done[t].set_exception(std::current_exception());
        }
        ms = now_ms() - t0;
    }

    /// One pass: tokens[c0 .. c0 + S) at positions p0 ..; with nll, the nll of every next token inside `tokens`.
    void prefill_pass(const std::vector<int>& tokens, int c0, int S, int p0, std::vector<float>* nll) {
        const int B = pf.sub;
        for (int i = 0; i < S; ++i) history.push_back(pack.engram_hash().token_map[tokens[c0 + i]]);
        const size_t eng_table = (size_t) pf.cap * kEngRows * (256 + 8);   // pinned bytes per engram table
        // the engram rows are read on their own thread while the GPU works; a layer waits for its table only
        std::vector<std::promise<void>> eng_done(n_eng);
        std::vector<std::future<void>> eng_ready;
        for (auto& d : eng_done) eng_ready.push_back(d.get_future());
        double eng_thread_ms = 0;
        std::thread eng_thread([&] { engram_pass(S, p0, eng_done, eng_thread_ms); });
        struct Join {
            std::thread& t;
            ~Join() { if (t.joinable()) t.join(); }
        } eng_join{eng_thread};
        for (int i = 0; i < S; ++i) pfh.tok[i] = tokens[c0 + i];
        ck(cudaMemcpyAsync(pf.tok, pfh.tok, (size_t) S * 4, cudaMemcpyHostToDevice, 0), "tokens up");
        ck(cudaMemcpyAsync(pf.pre_mix, pfh.pre, (size_t) S * kHc * 4, cudaMemcpyHostToDevice, 0), "pre_mix");
        prefill::embed_rows(embed, pf.tok, S, pf.h);
        // stream mode: with this many tokens nearly every expert is routed in every layer, so the stream starts
        // with all non-resident experts of all layers before any routing is known (upstream's stream_all)
        // The first engram table is needed at layer 1; its rows are small random reads that lose most of their
        // throughput when the expert stream's large reads share the SSD (measured at 32K: 10.3 s alone, 14.9 s
        // shared). So the stream starts with layer 0's experts only and gets the rest once that table is read.
        std::vector<std::pair<int, int>> later_jobs;
        if (S >= kStreamAll) {
            layer_first_job.assign(kLayers + 1, 0);
            std::vector<std::pair<int, int>> jobs;
            for (int l = 0; l < kLayers; ++l) {
                for (int e = 0; e < kExperts; ++e)
                    if (!resident(l, e)) jobs.push_back({l, e});
                layer_first_job[l + 1] = (int64_t) jobs.size();
            }
            const size_t now = n_eng ? (size_t) layer_first_job[1] : jobs.size();
            later_jobs.assign(jobs.begin() + now, jobs.end());
            jobs.resize(now);
            const int64_t first = estream->push(jobs);   // later_jobs follow in the same numbering
            for (auto& j : layer_first_job) j += first;
        }
        bool later_pushed = later_jobs.empty();
        int eng_i = 0;
        for (int l = 0; l < kLayers; ++l) {
            auto& y = L[l];
            for (int b0 = 0; b0 < S; b0 += B) {
                const int T = std::min(B, S - b0);
                bf16* h = pf.h + (size_t) b0 * kHc * kDim;
                bf16* h2 = pf.h2 + (size_t) b0 * kHc * kDim;
                if (is_engram_layer(l)) {   // this sub-batch's rows of this table, pinned -> device
                    if (b0 == 0) {
                        const double w0 = now_ms();
                        eng_ready[eng_i].get();   // rethrows a read error
                        ptm->engram_ms += now_ms() - w0;
                        if (!later_pushed) {      // the first table is in: the SSD goes to the expert stream
                            estream->push(later_jobs);
                            later_pushed = true;
                        }
                    }
                    const uint8_t* hw = pfh.eng + (size_t) eng_i * eng_table;
                    uint8_t* dw = pf.eng_dev;
                    const size_t rows = (size_t) T * kEngRows, r0 = (size_t) b0 * kEngRows;
                    ck(cudaMemcpyAsync(dw, hw + r0 * 256, rows * 256, cudaMemcpyHostToDevice, 0), "engram rows");
                    ck(cudaMemcpyAsync(dw + rows * 256, hw + (size_t) pf.cap * kEngRows * 256 + r0 * 8, rows * 8,
                                       cudaMemcpyHostToDevice, 0), "engram scales");
                    ops::engram_dequant(dw, dw + rows * 256, (int) rows, pf.eng_vals);
                    fp8_rows(pf.eng_vals, T, y.eng_wkv, pf.eng_kv);
                    ops::engram_apply(h, pf.eng_kv, y.eng_qw, y.eng_kw, kNormEps, T);
                }
                hc_rows(h, T, y.hc_attn_fn, y.hc_attn_scale, y.hc_attn_base, pf.pre_mix + b0 * kHc, pf.xa, pf.attn_pre,
                        pf.attn_post, pf.attn_comb);
                ops::rmsnorm(pf.xa, y.attn_norm, pf.xa, kDim, kNormEps, T);
                attention_rows(l, T, p0 + b0, b0);
                ops::hc_post(pf.attn_out, h, pf.attn_post, pf.attn_comb, h2, T);
                bf16* xf = pf.xf + (size_t) b0 * kDim;
                hc_rows(h2, T, y.hc_ffn_fn, y.hc_ffn_scale, y.hc_ffn_base, pf.attn_pre, xf, pf.ffn_pre + b0 * kHc,
                        pf.ffn_post + b0 * kHc, pf.ffn_comb + b0 * kHc * kHc);
                ops::rmsnorm(xf, y.ffn_norm, xf, kDim, kNormEps, T);
                moe_route(l, T, b0);
            }
            if (is_engram_layer(l)) ++eng_i;
            moe_experts(l, S);
            for (int b0 = 0; b0 < S; b0 += B) moe_finish(l, std::min(B, S - b0), b0);
        }
        const int n = (int) tokens.size();
        if (nll) {
            for (int i = 0; i < S; ++i) pfh.targets[i] = c0 + i + 1 < n ? tokens[c0 + i + 1] : -1;
            ck(cudaMemcpyAsync(pf.targets, pfh.targets, (size_t) S * 4, cudaMemcpyHostToDevice, 0), "targets up");
        }
        for (int b0 = 0; b0 < S; b0 += B) {
            const int T = std::min(B, S - b0);
            ops::hc_pre(pf.h + (size_t) b0 * kHc * kDim, pf.pre_mix + b0 * kHc, pf.final_x, T);
            ops::rmsnorm(pf.final_x, final_norm, pf.final_x, kDim, kNormEps, T);
            if (nll)
                for (int r0 = 0; r0 < T; r0 += kNllRows) {
                    const int nb = std::min(kNllRows, T - r0);
                    prefill::bf16_gemm(pf.final_x + (size_t) r0 * kDim, head, nb, kDim, kVocab, nullptr, pf.nll_logits);
                    prefill::nll_rows(pf.nll_logits, nb, kVocab, pf.targets + b0 + r0, pf.nll_out + b0 + r0);
                }
            if (c0 + b0 + T == n) {   // the last token's logits, as step() computes them
                ops::bf16_linear(pf.final_x + (size_t) (T - 1) * kDim, nullptr, head, kDim, kVocab, nullptr, logits);
                lg.resize(kVocab);
                ck(cudaMemcpy(lg.data(), logits, kVocab * 4, cudaMemcpyDeviceToHost), "logits");
            }
        }
        if (nll) {
            ck(cudaMemcpy(pfh.nll, pf.nll_out, (size_t) S * 4, cudaMemcpyDeviceToHost), "nll down");
            for (int i = 0; i < S; ++i)
                if (c0 + i + 1 < n) (*nll)[c0 + i] = pfh.nll[i];
        }
        if (S >= kStreamAll) estream->drain();   // every job of the pass was consumed
    }

    int prefill(const std::vector<int>& tokens, int pos, std::vector<float>* nll, PrefillTiming& pt) {
        if (pos != (int) history.size()) throw std::runtime_error("prefill must continue at the tokens fed so far");
        const int n = (int) tokens.size();
        if (n == 0) throw std::runtime_error("prefill of no tokens");
        if (pos + n > max_seq) throw std::runtime_error("prefill past max_seq");
        pt = PrefillTiming{};
        ptm = &pt;
        const double t0 = now_ms();
        if (nll) nll->assign(n - 1, 0.0f);
        int next = -1;
        if (opt.prefill_chunk <= 0) {   // token by token
            Timing tm;
            for (int i = 0; i < n; ++i) {
                next = step(tokens[i], pos + i, nullptr, tm);
                if (nll && i + 1 < n) {
                    double mx = -1e300, se = 0;
                    for (float v : lg) mx = std::max(mx, (double) v);
                    for (float v : lg) se += std::exp((double) v - mx);
                    (*nll)[i] = (float) (mx + std::log(se) - lg[tokens[i + 1]]);
                }
            }
            pt.total_ms = now_ms() - t0;
            return next;
        }
        prefill_begin(n);
        try {
            pt.chunk_tokens = pf.cap;
            pt.sub_batch = pf.sub;
            for (int c0 = 0; c0 < n; c0 += pf.cap) {
                prefill_pass(tokens, c0, std::min(pf.cap, n - c0), pos + c0, nll);
                ++pt.chunks;
            }
            estream->drain();
            const auto st = estream->take_stats();
            pt.from_ram = st.from_ram;
            pt.from_cache = st.from_cache;
            pt.from_ssd = st.from_ssd;
            pt.stream_wait_ms = st.consumer_wait_ms;
        } catch (...) {
            prefill_end();
            throw;
        }
        prefill_end();
        pt.total_ms = now_ms() - t0;
        return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    }

    // ------------------------------------------------------------------------------------- step
    int step(int token, int pos, StepDump* dump, Timing& tm) {
        if (pos != (int) history.size()) throw std::runtime_error("tokens must be fed in order from position 0");
        if (pos >= max_seq) throw std::runtime_error("position past max_seq");
        tm = Timing{};
        const double t_start = now_ms();
        history.push_back(pack.engram_hash().token_map[token]);
        {
            const double t0 = now_ms();
            int li = 0;
            for (int l = 0; l < kLayers; ++l)
                if (is_engram_layer(l)) engram_ids(l, li++, pos);
            engram_read_all();
            tm.engram_ms = now_ms() - t0;
        }
        if (vram) tm.vram_swaps = vram->between_steps();   // the device is idle: the last step ended in a sync
        db->reset();
        worker_us = 0;
        worker_misses = 0;
        worker_ram = worker_file = worker_ssd = 0;
        {
            std::lock_guard<std::mutex> lk(mu);
            ++go;
        }
        cv.notify_one();
        if (dump) {
            dump->hidden.assign((size_t) kLayers * kHc * kDim, 0);
            dump->routes.assign(kLayers, {});
            dump->weights.assign(kLayers, {});
        }
        if (n_eng)
            ck(cudaMemcpyAsync(eng_dev, eng_host, (size_t) n_eng * kEngRows * (256 + 8), cudaMemcpyHostToDevice, 0),
               "engram rows");
        ops::window_index(pos, idx_dev);
        ops::embed(embed, token, h);
        const float one_hot[kHc] = {1, 0, 0, 0};
        ck(cudaMemcpy(pre_mix, one_hot, sizeof one_hot, cudaMemcpyHostToDevice), "pre_mix");
        int eng_i = 0;
        for (int l = 0; l < kLayers; ++l) {
            auto& y = L[l];
            if (is_engram_layer(l)) engram(l, eng_i++);
            const bool dbg_layer = dbg && (l == 1 || l == 2);
            if (dbg_layer) dbg_write(h, kHc * kDim);                    // block input (after engram)
            // attention sub-block: h -> h2
            kernels::hc_mixes_pre(h, 1, y.hc_attn_fn, y.hc_attn_scale, y.hc_attn_base, pre_mix, xa, attn_pre, attn_post,
                                  attn_comb, 0);
            ops::rmsnorm(xa, y.attn_norm, xa, kDim, kNormEps);
            if (dbg_layer) dbg_write(xa, kDim);                         // attention input
            attention(l, pos);
            if (dbg_layer) dbg_write(attn_out, kDim);                   // attention output
            ops::hc_post(attn_out, h, attn_post, attn_comb, h2);
            // ffn sub-block: h2 -> h
            kernels::hc_mixes_pre(h2, 1, y.hc_ffn_fn, y.hc_ffn_scale, y.hc_ffn_base, attn_pre, xf, ffn_pre, ffn_post,
                                  ffn_comb, 0);
            ops::rmsnorm(xf, y.ffn_norm, xf, kDim, kNormEps);
            if (dbg_layer) dbg_write(xf, kDim);                         // ffn input
            moe(l);
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
        int32_t r[kLayers * kTopK];
        ck(cudaMemcpy(r, routes_dev, sizeof r, cudaMemcpyDeviceToHost), "routes");
        if (vram) vram->count(r, kTopK);
        if (dump) {
            float wv[kLayers * kTopK];
            ck(cudaMemcpy(wv, weights_dev, sizeof wv, cudaMemcpyDeviceToHost), "dump weights");
            for (int l = 0; l < kLayers; ++l)
                for (int i = 0; i < kTopK; ++i) {
                    dump->routes[l][i] = r[l * kTopK + i];
                    dump->weights[l][i] = wv[l * kTopK + i];
                }
            std::vector<int> order(kVocab);
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + 8, order.end(), [&](int a, int b) { return lg[a] > lg[b]; });
            dump->top_logits.clear();
            for (int i = 0; i < 8; ++i) dump->top_logits.push_back({order[i], lg[order[i]]});
        }
        tm.total_ms = now_ms() - t_start;
        tm.cpu_experts_ms = worker_us.load() / 1000.0;
        tm.expert_total = kLayers * kTopK;
        tm.expert_hits = tm.expert_total - worker_misses.load();
        tm.ram_experts = worker_ram.load();
        tm.file_experts = worker_file.load();
        tm.ssd_experts = worker_ssd.load();
        if (lookahead) {
            const auto st = lookahead->take_stats();
            tm.warmed = (int) st.warmed;
            tm.warmed_useful = (int) st.useful;
        }
        tm.gpu_ms = tm.total_ms - tm.engram_ms;
        return best;
    }
};

Engine::Engine(const std::string& pack_dir, const EngineOptions& opt) : impl_(new Impl(pack_dir, opt)) {
    impl_->init();
}

Engine::Engine(const std::string& pack_dir, int max_seq, int cpu_threads)
    : Engine(pack_dir, [&] {
          EngineOptions o;
          o.max_seq = max_seq;
          o.cpu_threads = cpu_threads;
          return o;
      }()) {}

int Engine::vram_expert_slots() const { return impl_->vram ? impl_->vram->slots() : 0; }

Engine::~Engine() = default;

int Engine::step(int token, int pos, StepDump* dump) { return impl_->step(token, pos, dump, timing_); }

int Engine::prefill(const std::vector<int>& tokens, int pos, std::vector<float>* nll) {
    return impl_->prefill(tokens, pos, nll, prefill_timing_);
}

const std::vector<float>& Engine::last_logits() const { return impl_->lg; }

}  // namespace strata::ds41
