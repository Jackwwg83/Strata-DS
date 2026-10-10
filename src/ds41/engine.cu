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
#include "strata/ds41/expert_prefetch.hpp"
#include "strata/ds41/expert_stream.hpp"
#include "strata/ds41/host_experts.hpp"
#include "strata/ds41/lookahead.hpp"
#include "strata/ds41/fp8_gemv.hpp"
#include "strata/ds41/kernels/k12_exl3_moe_prefill.hpp"
#include "strata/ds41/kernels/k13_sparse_attn_prefill.hpp"
#include "strata/ds41/kernels/k14_indexer_prefill.hpp"
#include "strata/ds41/kernels/k15_hc_prefill.hpp"
#include "strata/ds41/kernels/k2_fp8_gemm.hpp"
#include "strata/ds41/kernels/k3_sparse_attn.hpp"
#include "strata/ds41/kernels/k5_indexer.hpp"
#include "strata/ds41/kernels/k7_hc.hpp"
#include "strata/ds41/kernels/k8_router.hpp"
#include "strata/ds41/ops.hpp"
#include "strata/ds41/prefill_ops.hpp"
#include "strata/ds41/vram_experts.hpp"
#include "strata/ds41/wo_a_fp8.hpp"

#include "moe_mul1.h"   // third_party/exllamav3_moe

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <map>
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

/// decode: wait until the engram rows of this step are in the pinned buffer (the host's reader raises the flag to the
/// step's engram epoch, dp[4])
__global__ void wait_engram_k(const volatile uint32_t* flag, const int* dp) {
    const uint32_t want = (uint32_t) dp[4];
    while (*flag < want) __nanosleep(200);
    __threadfence_system();
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
        Fp8 wo_a8{};   // wo_a as FP8 with block scales (packs from 2026-10-06); wo_a below is then null
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
    // the worker's critical-path timing of the current step (Timing::worker_*): stored by the worker before it marks
    // the last layer done, read after the step's device sync (the GPU cannot finish before that mark)
    std::atomic<double> worker_wake_at{0}, worker_end_at{0}, worker_wait{0}, worker_first_wait{0}, worker_admit{0};
    std::atomic<int> worker_misses{0}; // routed uses outside VRAM (CPU plus zero-copy)
    std::unique_ptr<VramExperts> vram;  // the VRAM tier (null: none)
    std::unique_ptr<HostExperts> host;  // the RAM tier (null: none); the rest is read from the mapped file
    std::atomic<int> worker_ram{0}, worker_file{0}, worker_ssd{0};   // CPU experts of the step by tier
    std::atomic<bool> admit_warned{false};
    std::vector<unsigned char> mincore_buf;
    std::unique_ptr<RouterLookahead> lookahead;   // warms the next layer's file-tier experts (DS41_LOOKAHEAD=0: off)
    bool fetch_now = true;             // ask for a layer's missing file pages before computing (DS41_FETCH_NOW=0: off)
    // The adaptive RAM tier (HostExperts::enable_adapt) keeps N free slots per slot size (DS41_RAM_ADAPT=N, default
    // 8; 0: the static tier); a miss is read into one and stays. Its default budget keeps 8 GiB free instead of 24 (the
    // file cache it replaces needs no room). ds41/docs/cache-design-2026-10-08.html. The default since 2026-10-10:
    // RTX 5090 Laptop, 60 GB RAM, automatic budgets, 256 tokens: code 11.4 -> 19.4 tok/s, agent 9.6 -> 17.6, zh_chat
    // 10.7 -> 19.2 (static 29 GiB tier against adaptive 47 GiB)
    int ram_adapt = [] {
        const char* v = std::getenv("DS41_RAM_ADAPT");
        if (!v || !*v) return 8;
        char* end = nullptr;
        const long n = std::strtol(v, &end, 10);
        if (end == v || *end || n < 0 || n > 64) throw std::invalid_argument("DS41_RAM_ADAPT must be an integer in [0, 64]");
        return (int) n;
    }();
    int32_t* gpu_sel = nullptr;        // [6] per-call descriptor indices (-1: CPU)
    std::shared_ptr<int> zc_quota;      // [layers] device values; update only between steps
    std::shared_ptr<void> zc_workspace; // used when there is no VRAM tier
    std::unique_ptr<ExpertStaging> zc_stage;
    std::shared_ptr<ExpertBlob> zc_blobs; // immutable [layers][experts] pack metadata
    std::unique_ptr<ExpertPrefetch> prefetch;   // DS41_PREFETCH=N: the next layer's N guessed experts, copied ahead
    std::atomic<int> worker_prefetched{0};
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
        ck(cudaStreamSynchronize(st), "debug dump");
        std::vector<uint16_t> b(n);
        ck(cudaMemcpy(b.data(), dev, (size_t) n * 2, cudaMemcpyDeviceToHost), "debug dump");
        std::fwrite(b.data(), 2, n, dbg);
    }

    // decode on its own stream, captured as CUDA graphs (upstream session_capture_token / Verifier): the position
    // dependent values come from fixed pinned staging, copied to the device by the graph's first node
    struct StepParams { int token, pos, t1, t2, eng_epoch; };   // t1, t2: compressed lengths at ratio 1 and 2
    cudaStream_t st = nullptr;
    StepParams* hp = nullptr;          // pinned staging of the next replay
    int* dp = nullptr;                 // device copy: dp[0] token, dp[1] pos, dp[2] t1, dp[3] t2, dp[4] engram epoch
    // decode engram rows read beside the step's first layer: a reader thread fills eng_host and raises eng_flag
    // (mapped) to the step's epoch; the graph waits for it before copying the rows, just before layer 1
    uint32_t* eng_flag = nullptr;      // mapped host word
    uint32_t* eng_flag_dev = nullptr;
    uint32_t eng_epoch = 0;
    std::thread eng_thread;
    std::mutex eng_mu;
    std::condition_variable eng_cv;
    bool eng_posted = false, eng_done = true, eng_quit = false;
    std::exception_ptr eng_error;
    double eng_read_ms = 0;
    int* d_next = nullptr;             // device argmax of the logits
    int* hp_next = nullptr;            // pinned: the argmax, the logits and the routes, after the step
    float* lg_pinned = nullptr;
    int32_t* routes_pinned = nullptr;
    float* one_hot_dev = nullptr;      // {1, 0, 0, 0}: pre_mix at the first layer
    bool use_graph = true;             // DS41_GRAPH=0: eager launches (also with a dump or DS41_DEBUG)
    int parity = 0;                    // pos % 2 of the step being enqueued (the ratio-2 compressor's slot)
    int64_t tcap1 = 1, tcap2 = 1;      // indexer capacities of the graph being enqueued
    std::map<uint64_t, cudaGraphExec_t> graphs;
    int graph_captures = 0;

    struct VerifyWorkspace;
    std::shared_ptr<VerifyWorkspace> verify_ws;
    bool verify_pending = false, verify_failed = false;
    void enqueue_verify(int m);
    VerifyResult verify(const std::vector<int>& window, int pos, bool logits, Timing& tm);
    void commit_verify(int n_keep);

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
        {
            std::lock_guard<std::mutex> lk(eng_mu);
            eng_quit = true;
        }
        eng_cv.notify_all();
        if (eng_thread.joinable()) eng_thread.join();
        if (eng_flag) __atomic_store_n(eng_flag, ~0u, __ATOMIC_RELEASE);   // a step that threw: release its wait
        // the lookahead calls into both tiers, and the VRAM tier's copy thread writes into RAM slots: stop them first
        lookahead.reset();
        vram.reset();
        host.reset();
        for (auto& [k, e] : graphs) cudaGraphExecDestroy(e);
        for (auto& [k, e] : batch_graphs) cudaGraphExecDestroy(e);
        if (st) cudaStreamSynchronize(st);
        prefill_end_quiet();
        for (const auto& y : L)
            if (y.moe_handle >= 0) exl3_moe_cpu_free_layer(y.moe_handle);
        for (void* p : owned_dev) cudaFree(p);
        for (cudaEvent_t e : pfp.ev) cudaEventDestroy(e);
        if (dbg) std::fclose(dbg);
        if (eng_host) cudaFreeHost(eng_host);
        if (eng_flag) cudaFreeHost(eng_flag);
        if (hp) cudaFreeHost(hp);
        if (hp_next) cudaFreeHost(hp_next);
        if (lg_pinned) cudaFreeHost(lg_pinned);
        if (routes_pinned) cudaFreeHost(routes_pinned);
        if (st) cudaStreamDestroy(st);
        if (pfh.base) cudaFreeHost(pfh.base);
    }

    // device allocations of this engine (decode state, scratch, tables): freed in ~Impl
    std::vector<void*> owned_dev;
    template <typename T>
    T* dalloc_own(size_t n) {
        T* p = dalloc<T>(n);
        owned_dev.push_back(p);
        return p;
    }

    // a failure after the state changed (history pushed, a step released to the CPU worker, a prefill pass started)
    // leaves the KV caches and the history out of step: every later call is refused
    bool broken = false;
    bool pf_started = false;           // the running prefill has begun changing the state
    int file_trace = [] { const char* v = std::getenv("DS41_FILE_TRACE"); return v ? std::atoi(v) : 0; }();
    // DS41_PREDICT_STATS=1: how well layer l+1's router on layer l's expert input predicts layer l+1's experts (the
    // basis of a prefetch). Recall of all routed experts and of the misses (not in VRAM), with 6, 9 and 12 guesses.
    static constexpr int kPredK = 12;
    bool pred_stats = [] { const char* v = std::getenv("DS41_PREDICT_STATS"); return v && v[0] == '1'; }();
    std::vector<int32_t> pred_ids = std::vector<int32_t>(kLayers * kPredK, -1);
    int64_t pred_all[3] = {}, pred_miss[3] = {}, pred_n_all = 0, pred_n_miss = 0;
    void predict_tally() {
        static constexpr int ks[3] = {6, 9, 12};
        for (int l = 1; l < kLayers; ++l)
            for (int i = 0; i < kTopK; ++i) {
                const int32_t e = routes_pinned[l * kTopK + i];
                const bool miss = !vram || vram->res_host()[(size_t) l * kExperts + e] < 0;
                ++pred_n_all;
                pred_n_miss += miss;
                for (int k = 0; k < 3; ++k) {
                    const int32_t* p = pred_ids.data() + l * kPredK;
                    const bool found = std::find(p, p + ks[k], e) != p + ks[k];
                    pred_all[k] += found;
                    pred_miss[k] += found && miss;
                }
            }
        if (pred_n_all && pred_n_all % (kTopK * (kLayers - 1) * 64) == 0)
            std::fprintf(stderr, "ds41 predict: recall all %.3f %.3f %.3f, misses %.3f %.3f %.3f (6 / 9 / 12 guesses, "
                                 "%lld uses)\n", pred_all[0] / (double) pred_n_all, pred_all[1] / (double) pred_n_all,
                         pred_all[2] / (double) pred_n_all, pred_miss[0] / (double) std::max<int64_t>(1, pred_n_miss),
                         pred_miss[1] / (double) std::max<int64_t>(1, pred_n_miss),
                         pred_miss[2] / (double) std::max<int64_t>(1, pred_n_miss), (long long) pred_n_all);
    }
    PrefillProgress progress;          // prefill progress (set_prefill_progress); a false return cancels
    // snapshot slots: the window rings of all layers, then the compressor states of the kv sources with ratio > 1
    struct Snapshot { int pos = -1; bf16* win = nullptr; float* comp = nullptr; };
    // batch slots (EngineOptions.batch_slots, src/ds41/batch.cu): per slot, every layer's attention state and its tokens
    struct SlotLayer {
        bf16 *window = nullptr, *comp = nullptr, *idx_keys = nullptr;
        float *kv_state = nullptr, *score_state = nullptr;
    };
    struct SlotState {
        std::vector<SlotLayer> L;
        std::vector<int32_t> history;
    };
    std::vector<SlotState> slot_states;
    std::shared_ptr<VerifyWorkspace> batch_ws;   // the rows' buffers: a verify workspace has the same shapes
    std::unique_ptr<ExpertStaging> batch_stage;          // the staging copies of all rows (4 x 6 experts)
    std::unique_ptr<EngramRows> eng_rows_batch;          // every row's engram rows in one read
    std::map<std::array<int64_t, 5>, cudaGraphExec_t> batch_graphs;
    bool batch_warmed = false;
    int batch_row_slot[kVerifyMaxTokens] = {}, batch_row_parity[kVerifyMaxTokens] = {};
    void alloc_slots();
    void slot_copy(int slot, bool to_slot);
    void enqueue_slots(int m);
    std::vector<int> step_slots(const std::vector<int>& slots, const std::vector<int>& tokens, Timing& tm);
    std::vector<Snapshot> snaps;
    size_t snap_comp_floats = 0;
    std::exception_ptr worker_error;   // set by the CPU worker; rethrown by the step that ran into it
    void usable() const {
        if (broken) throw std::runtime_error("ds41 engine: unusable after an earlier failure; create a new Engine");
    }
    static void check_token(int t) {
        if (t < 0 || t >= kVocab) throw std::invalid_argument("ds41 engine: token id " + std::to_string(t) + " is outside [0, " + std::to_string(kVocab) + ")");
    }
    /// DS41_TEST_FAULT=<name>: throw at that point once, then the variable is cleared (tests of the failure paths)
    static bool fault(const char* name) {
        const char* v = std::getenv("DS41_TEST_FAULT");
        if (!v || std::strcmp(v, name) != 0) return false;
        unsetenv("DS41_TEST_FAULT");
        return true;
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
        // A caller that asks for static residency (adapt_every 0) gets a static RAM tier too, unless DS41_RAM_ADAPT
        // says otherwise: with the adaptive tier the prefetch computes guessed RAM experts on the GPU, following the
        // tier's history, and step() and step_slots() would no longer give the same tokens (engine.hpp's promise)
        if (opt.adapt_every == 0 && !std::getenv("DS41_RAM_ADAPT")) ram_adapt = 0;
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
            if (pack.has_dense(p + "attn.wo_a.scale")) y.wo_a8 = fp8(p + "attn.wo_a");
            else y.wo_a = bf(p + "attn.wo_a.weight");
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
                y.comp = dalloc_own<bf16>((size_t) (max_seq / y.ratio + 1) * kHeadDim);
                y.idx_keys = dalloc_own<bf16>((size_t) (max_seq / y.ratio + 1) * kIndexDim);
                if (y.ratio > 1) {
                    y.kv_state = dalloc_own<float>((size_t) y.ratio * kHeadDim);
                    y.score_state = dalloc_own<float>((size_t) y.ratio * kHeadDim);
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
            y.window = dalloc_own<bf16>((size_t) kWindow * kHeadDim);
            register_cpu_experts(l);
        }
        // rope tables
        auto plain = rope_table(max_seq, false), yarn = rope_table(max_seq, true);
        rope_plain = dalloc_own<float>(plain.size());
        rope_yarn = dalloc_own<float>(yarn.size());
        ck(cudaMemcpy(rope_plain, plain.data(), plain.size() * 4, cudaMemcpyHostToDevice), "rope");
        ck(cudaMemcpy(rope_yarn, yarn.data(), yarn.size() * 4, cudaMemcpyHostToDevice), "rope");
        // engram tables
        {
            std::vector<EngramRows::Table> tabs;
            for (const auto& t : pack.engram_tables()) tabs.push_back({t.path, t.weight_offset, t.scale_offset});
            n_eng = (int) tabs.size();
            // every request of a step in flight at once: 24 rows x (weight, scale) per table, one thread each (on a disk
            // with 0.6 ms random-read latency, 16 threads made it 5.9 ms per token)
            if (n_eng) eng_rows = std::make_unique<EngramRows>(tabs, kEngRows, 256, 8, 2 * kEngRows);   // per table
            eng_ids.assign(n_eng, std::vector<int64_t>(kEngRows, 0));
        }
        const size_t eng_bytes = (size_t) n_eng * kEngRows * (256 + 8);
        ck(cudaHostAlloc((void**) &eng_host, std::max<size_t>(eng_bytes, 1), cudaHostAllocDefault), "engram pinned");
        eng_dev = dalloc_own<uint8_t>(std::max<size_t>(eng_bytes, 1));
        ck(cudaHostAlloc((void**) &eng_flag, sizeof(uint32_t), cudaHostAllocMapped), "engram flag");
        *eng_flag = 0;
        ck(cudaHostGetDevicePointer((void**) &eng_flag_dev, eng_flag, 0), "engram flag alias");
        if (n_eng) eng_thread = std::thread([this] { engram_worker(); });
        // scratch
        h = dalloc_own<bf16>(kHc * kDim);
        h2 = dalloc_own<bf16>(kHc * kDim);
        xa = dalloc_own<bf16>(kDim);
        xf = dalloc_own<bf16>(kDim);
        qr = dalloc_own<bf16>(kQLora);
        q = dalloc_own<bf16>(kHeads * kHeadDim);
        kvv = dalloc_own<bf16>(kHeadDim);
        o = dalloc_own<bf16>(kHeads * kHeadDim);
        oa = dalloc_own<bf16>(kOGroups * kOLora);
        attn_out = dalloc_own<bf16>(kDim);
        latent = dalloc_own<bf16>(kHeadDim);
        ik = dalloc_own<bf16>(kIndexDim);
        iq = dalloc_own<bf16>(kIndexHeads * kIndexDim);
        iw_raw = dalloc_own<bf16>(kIndexHeads);
        iw = dalloc_own<bf16>(kIndexHeads);
        g = dalloc_own<bf16>(kMoeInter);
        u = dalloc_own<bf16>(kMoeInter);
        sh_h = dalloc_own<bf16>(kMoeInter);
        sh_out = dalloc_own<bf16>(kDim);
        ffn_out = dalloc_own<bf16>(kDim);
        eng_vals = dalloc_own<bf16>(24 * 256);
        eng_kv = dalloc_own<bf16>((kHc + 1) * kDim);
        final_x = dalloc_own<bf16>(kDim);
        act = dalloc_own<float>(8192);
        pre_mix = dalloc_own<float>(kHc);
        pre = dalloc_own<float>(kHc);
        post = dalloc_own<float>(kHc);
        comb = dalloc_own<float>(kHc * kHc);
        attn_pre = dalloc_own<float>(kHc);
        attn_post = dalloc_own<float>(kHc);
        attn_comb = dalloc_own<float>(kHc * kHc);
        ffn_pre = dalloc_own<float>(kHc);
        ffn_post = dalloc_own<float>(kHc);
        ffn_comb = dalloc_own<float>(kHc * kHc);
        ckv = dalloc_own<float>(kHeadDim);
        cscore = dalloc_own<float>(kHeadDim);
        scores = dalloc_own<float>(max_seq + 1);
        routed = dalloc_own<float>(kDim);
        logits = dalloc_own<float>(kVocab);
        x_half_dev = dalloc_own<uint16_t>(kDim);
        idx_dev = dalloc_own<int32_t>(kWindow + kIndexTopK);
        routes_dev = dalloc_own<int32_t>(kLayers * kTopK);
        weights_dev = dalloc_own<float>(kLayers * kTopK);
        cand_dev = dalloc_own<uint8_t>(max_seq + 1);
        history.reserve(max_seq);
        alloc_snapshots();   // before the VRAM tier sizes itself from the free memory
        db = std::make_unique<ExpertDoorbell>(1, kTopK, kDim);
        gpu_sel = dalloc_own<int32_t>(kTopK);
        // decode stream, graph staging, outputs
        ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "decode stream");
        ck(cudaHostAlloc((void**) &hp, sizeof(StepParams), cudaHostAllocDefault), "step params");
        ck(cudaHostAlloc((void**) &hp_next, sizeof(int), cudaHostAllocDefault), "next token");
        ck(cudaHostAlloc((void**) &lg_pinned, (size_t) kVocab * 4, cudaHostAllocDefault), "logits");
        ck(cudaHostAlloc((void**) &routes_pinned, kLayers * kTopK * 4, cudaHostAllocDefault), "routes");
        dp = dalloc_own<int>(5);
        d_next = dalloc_own<int>(1);
        one_hot_dev = dalloc_own<float>(kHc);
        {
            const float oh[kHc] = {1, 0, 0, 0};
            ck(cudaMemcpy(one_hot_dev, oh, sizeof oh, cudaMemcpyHostToDevice), "one hot");
        }
        kernels::hc_init();       // K7 / K8 scratch, allocated outside any capture
        kernels::router_init();
        if (const char* v = std::getenv("DS41_GRAPH")) use_graph = v[0] != '0';
        // Reserve staging before the automatic VRAM cache consumes free memory. Keep this reservation at q=0
        // too, so quota sweeps within staged mode use the same initial residency.
        const char* stage_env = std::getenv("DS41_ZC_STAGE");
        if (stage_env && std::strcmp(stage_env, "0") != 0 && std::strcmp(stage_env, "1") != 0)
            throw std::invalid_argument("DS41_ZC_STAGE must be 0 or 1");
        if ((!stage_env || stage_env[0] != '0') && !opt.expert_profile.empty() && opt.ram_budget_gib != 0) {
            std::vector<ExpertBlob> blobs;
            size_t largest = 0;
            for (int l = 0; l < kLayers; ++l) {
                for (int e = 0; e < kExperts; ++e) {
                    const auto& slot = pack.expert(l, e);
                    blobs.push_back({size_t(slot.bytes), size_t(slot.comp_off[0])});
                    largest = std::max(largest, size_t(slot.bytes));
                }
            }
            zc_stage = std::make_unique<ExpertStaging>(kTopK, largest);
            zc_blobs = std::shared_ptr<ExpertBlob>(dalloc<ExpertBlob>(blobs.size()),
                                                  [](ExpertBlob* p) { cudaFree(p); });
            ck(cudaMemcpy(zc_blobs.get(), blobs.data(), blobs.size() * sizeof(ExpertBlob), cudaMemcpyHostToDevice),
               "staging blob metadata");
            ck(cudaStreamSynchronize(nullptr), "staging initialization");
            std::fprintf(stderr, "ds41: zero-copy staging %d slots x %zu bytes (pack max %zu)\n",
                         kTopK, zc_stage->stride(), largest);
            // the prefetch buffers before the VRAM tier takes the free memory (DS41_PREFETCH_MB per layer parity).
            // With the adaptive RAM tier 4 guesses by default: on the laptop (50 GiB tier, fixed continuations, 2
            // rounds) code 52.2 -> 49.6 ms per token, agent 58.2 -> 53.6, zh_chat 52.3 -> 50.3 (3 guesses: 49.4,
            // 54.3, 50.4; 6: code 51.0)
            const char* pf_env = std::getenv("DS41_PREFETCH");
            const int guesses = pf_env ? std::atoi(pf_env) : (ram_adapt ? 4 : 0);
            if (guesses > 0) {
                const char* mb_env = std::getenv("DS41_PREFETCH_MB");
                const size_t mb = mb_env ? (size_t) std::atoi(mb_env) : 96;
                // copies with the copy engine from a host thread; DS41_PREFETCH_DMA=0: the copy kernel (slower at
                // every guess count: it slows the main stream's kernels while it reads host memory)
                const char* dma_env = std::getenv("DS41_PREFETCH_DMA");
                const bool dma = !(dma_env && dma_env[0] == '0');
                prefetch = std::make_unique<ExpertPrefetch>(std::min(guesses, ExpertPrefetch::kMaxGuesses),
                                                            std::max<size_t>(mb, 1) << 20, kExperts, kDim, dma);
                std::fprintf(stderr, "ds41: prefetch %d guesses per layer, 2 x %zu MiB, %s\n", prefetch->guesses(), mb,
                             dma ? "DMA copies" : "copy kernel");
            }
        }
        alloc_slots();   // the batch slots' state and staging, before the VRAM tier sizes itself
        // the VRAM expert tier last: an automatic slot count takes what the rest left free
        if (!opt.expert_profile.empty() && opt.vram_expert_slots != 0) {
            VramExperts::Adapt ad;
            ad.every = opt.adapt_every;
            ad.decay = opt.adapt_decay;
            ad.max_swaps = opt.adapt_swaps;
            vram = std::make_unique<VramExperts>(pack, opt.expert_profile, opt.vram_expert_slots,
                                                 opt.vram_reserve_bytes, ad);
            std::fprintf(stderr, "ds41: %d VRAM expert slots (%.2f GiB)\n", vram->slots(),
                         vram->arena_bytes() / (double) (1ull << 30));
        }
        // the RAM tier after it: the hottest experts the VRAM tier does not hold (upstream's resident budget)
        if (!opt.expert_profile.empty() && opt.ram_budget_gib != 0) {
            const size_t budget = opt.ram_budget_gib < 0 ? auto_ram_budget(ram_adapt ? 8ull << 30 : 24ull << 30)
                                                         : (size_t) (opt.ram_budget_gib * (double) (1ull << 30));
            std::vector<int64_t> handles;
            for (const auto& y : L) handles.push_back(y.moe_handle);
            const double t0 = now_ms();
            host = std::make_unique<HostExperts>(
                pack, read_expert_profile(opt.expert_profile, kLayers, kExperts),
                vram ? vram->res_host() : std::vector<int32_t>((size_t) kLayers * kExperts, -1), budget, handles, 8);
            if (vram) vram->set_host(host.get());
            std::fprintf(stderr, "ds41: RAM tier %d experts (%.1f GiB, %s), filled in %.1f s\n", host->slots(),
                         host->arena_bytes() / (double) (1ull << 30),
                         host->locked() ? "locked" : "not locked", (now_ms() - t0) / 1000.0);
            if (ram_adapt) {
                host->enable_adapt(ram_adapt);
                std::fprintf(stderr, "ds41: adaptive RAM tier, %d free slots per slot size (%d free)\n", ram_adapt,
                             host->free_slots());
            }
        }
        if (host && host->experts_dev()) {
            // Four PCIe reads cost about 1.05 ms; two CPU experts cost about 1.0-1.2 ms.
            // This is a starting quota for six misses, not a measured optimum.
            // With the adaptive RAM tier the CPU no longer waits for page faults and computes RAM experts faster than
            // the GPU reads them over PCIe: on the laptop (PCIe 5.0 x8, 256 tokens, code and agent prompts) quota
            // 0 / 1 / 2 / 4 / 6 took 59.7 / 60.1 / 61.7 / 63.7 / 66.6 ms per token. Its default is 0.
            int quota = ram_adapt ? 0 : 4;
            if (const char* value = std::getenv("DS41_ZC_QUOTA")) {
                char* end = nullptr;
                const long parsed = std::strtol(value, &end, 10);
                if (end == value || *end || parsed < 0 || parsed > kTopK)
                    throw std::invalid_argument("DS41_ZC_QUOTA must be an integer in [0, 6]");
                quota = (int) parsed;
            }
            zc_quota = std::shared_ptr<int>(dalloc<int>(kLayers), [](int* p) { cudaFree(p); });
            const std::vector<int> quotas(kLayers, quota);
            ck(cudaMemcpy(zc_quota.get(), quotas.data(), quotas.size() * sizeof(int), cudaMemcpyHostToDevice),
               "zero-copy quotas");
            if (!vram)
                zc_workspace = std::shared_ptr<void>(dalloc<uint8_t>(VramExperts::kWorkspaceBytes),
                                                     [](void* p) { cudaFree(p); });
            ck(cudaStreamSynchronize(nullptr), "zero-copy initialization");
            std::fprintf(stderr, "ds41: mapped RAM experts enabled, GPU quota %d per token per layer, %s\n",
                         quota, zc_stage ? "staged" : "direct");
        }
        // the router lookahead: every expert outside the VRAM and RAM tiers is read from the file (upstream turns it
        // on with a RAM budget; here the file tier exists whenever the experts do not all fit in RAM)
        if (const char* f = std::getenv("DS41_FETCH_NOW")) fetch_now = f[0] != '0';
        // With the adaptive RAM tier a miss is read with O_DIRECT, past the file cache: pages warmed there would only
        // take SSD time from those reads. DS41_LOOKAHEAD=1 keeps the lookahead on.
        const char* la_env = std::getenv("DS41_LOOKAHEAD");
        if (la_env ? la_env[0] != '0' : !(host && host->reserve() > 0)) {
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
        fp8_quantize_activation_f32((const uint16_t*) x, 1, w.k, act, st);
        fp8_block_gemv_q(act, 1, w.k, w.w, w.s, w.n, (uint16_t*) y, st);
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
            const double wake = now_ms();
            double ready = wake, wait = 0, first_wait = 0, admit = 0;   // ready: since when the worker waits for the GPU
            for (int l = 0; l < kLayers; ++l) {
                if (!db->wait_published(l + 1, stop)) return;
                const double published = now_ms();
                wait += published - ready;
                if (l == 0) first_wait = published - ready;
                if (lookahead) lookahead->post(l, db->x());   // predict layer l+1 while this layer computes
                if (lookahead && pred_stats && l + 1 < kLayers)   // DS41_PREDICT_STATS: layer l+1's top 12 from x_l
                    lookahead->predict_now(l + 1, (const uint16_t*) db->x(), kPredK, pred_ids.data() + (l + 1) * kPredK);
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
                    if (host && host->in_memory(l, e)) { ++worker_ram; continue; }
                    ++worker_file;
                    if (file_trace > 0) {   // DS41_FILE_TRACE=N: the first N CPU experts read from the file
                        --file_trace;
                        std::fprintf(stderr, "ds41 file expert: layer %d expert %d vram %d ram %d step pos %d\n", l, e,
                                     vram ? vram->res_host()[(size_t) l * kExperts + e] : -2,
                                     host ? host->slot_of(l, e) : -2, (int) history.size() - 1);
                    }
                    file_ids[n_file++] = e;
                }
                // the adaptive RAM tier reads the layer's file experts into free slots, all at once; the rest (no free
                // slot, or a failed read) come from the file as before
                bool kept[kTopK] = {}, missing[kTopK] = {};
                const bool adaptive = host && host->reserve() > 0 && n_file > 0;
                if (adaptive && !host->reads_direct())   // copied from the map: only missing pages read the SSD
                    for (int j = 0; j < n_file; ++j) missing[j] = file_pages_missing(l, file_ids[j]);
                if (adaptive) {
                    // wake the CPU expert pool before the reads: its workers nap after ~1 ms idle. Laptop, 4 pairs
                    // (code, agent): 0.6-1.2 ms per token less, mostly as shorter reads (busy cores, quicker I/O
                    // completions); in the bench a nap costs a call ~100 us after a 2 ms gap
                    try {
                        exl3_moe_cpu_pool_prime(cpu_threads);
                    } catch (...) {   // e.g. no thread could start: the forward below reports it as before
                    }
                    const double a0 = now_ms();
                    try {   // kept[] holds what was read even when admit() throws
                        host->admit(l, file_ids, n_file, kept);
                    } catch (const std::exception& ex) {   // e.g. no thread for the reads: the file path still works
                        if (!admit_warned.exchange(true))
                            std::fprintf(stderr, "ds41: adaptive RAM tier read failed (%s); using the file\n", ex.what());
                    }
                    admit += now_ms() - a0;
                }
                for (int j = 0; j < n_file; ++j) {
                    if (kept[j]) {   // computed from its new RAM slot
                        --worker_file;
                        ++worker_ram;
                        if (host->reads_direct() || missing[j]) ++worker_ssd;
                        continue;
                    }
                    if (!file_pages_missing(l, file_ids[j])) continue;
                    ++worker_ssd;
                    // upstream fetches a layer's missing experts in one batch before computing: ask for the whole
                    // range now, so the reads run in parallel instead of page fault by page fault
                    if (fetch_now) warm_file_expert(l, file_ids[j]);
                }
                if (lookahead) lookahead->observe(l, file_ids, n_file);
                // Keep expert_hits as VRAM hits. Timing derives CPU and zero-copy counts from the partition.
                worker_misses += misses + db->counts().zero_copy + db->counts().prefetched;
                worker_prefetched += db->counts().prefetched;
                const double t0 = now_ms();
                try {
                    if (fault("worker")) throw std::runtime_error("ds41 test fault: CPU expert worker");
                    if (misses)
                        exl3_moe_cpu_forward_raw(L[l].moe_handle, (const at::Half*) db->x(), db->ids(), wh, db->y(), 1,
                                                 kTopK, cpu_threads);
                    else
                        std::fill_n(db->y(), kDim, 0.0f);
                } catch (...) {
                    // keep the protocol (the GPU waits for this layer); the step rethrows after its sync
                    std::fill_n(db->y(), kDim, 0.0f);
                    std::lock_guard<std::mutex> lk(mu);
                    if (!worker_error) worker_error = std::current_exception();
                }
                worker_us += (int64_t) ((now_ms() - t0) * 1000.0);
                ready = now_ms();
                if (l + 1 == kLayers) {
                    worker_wake_at = wake;
                    worker_wait = wait;
                    worker_first_wait = first_wait;
                    worker_admit = admit;
                    worker_end_at = ready;
                }
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

    /// verify and batch rows (on their CPU worker): a layer's misses that are in no tier are read into the adaptive
    /// RAM tier, each expert once for all rows, as the decode worker does. ids: [n], -1 for none. The rows' routes
    /// are recorded with host->end_step() after the window is committed or the batch step ends.
    int admit_rows(int l, const int32_t* ids, int n) {
        if (!(host && host->reserve() > 0)) return 0;
        int32_t uniq[kVerifyMaxTokens * kTopK];
        int u = 0;
        for (int i = 0; i < n && u < kVerifyMaxTokens * kTopK; ++i) {
            const int32_t e = ids[i];
            if (e < 0 || e >= kExperts || host->in_memory(l, e)) continue;
            if (vram && vram->res_host()[(size_t) l * kExperts + e] >= 0) continue;
            if (std::find(uniq, uniq + u, e) == uniq + u) uniq[u++] = e;
        }
        if (u == 0) return 0;
        try {
            exl3_moe_cpu_pool_prime(cpu_threads);
        } catch (...) {
        }
        bool kept[kVerifyMaxTokens * kTopK] = {};
        try {
            return host->admit(l, uniq, u, kept);
        } catch (const std::exception& ex) {   // the file path still works
            if (!admit_warned.exchange(true))
                std::fprintf(stderr, "ds41: adaptive RAM tier read failed (%s); using the file\n", ex.what());
            return 0;
        }
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
    void engram_ids(int l, int li, int pos) { engram_ids(l, li, pos, history); }
    /// the rows of engram layer l (table li) for position pos of the token sequence `hist` (a slot's, or the main one)
    void engram_ids(int l, int li, int pos, const std::vector<int32_t>& hist) {
        const auto& hs = pack.engram_hash();
        const int n = hs.max_ngram, nh = hs.n_heads, cols = (n - 1) * nh;
        std::vector<int64_t> toks(n);
        for (int s = 0; s < n; ++s) toks[s] = pos - s >= 0 ? hist[pos - s] : hs.pad;
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

    /// The decode engram reader: one step's rows at a time (post_engram), then the flag the graph waits for. The flag
    /// rises even when the read fails (the GPU must not wait forever); finish_engram rethrows the error after the
    /// step. Reading beside layer 0 instead of before the step hides most of the read (~1.4 ms on the laptop).
    void engram_worker() {
        std::unique_lock<std::mutex> lk(eng_mu);
        while (true) {
            eng_cv.wait(lk, [&] { return eng_quit || eng_posted; });
            if (!eng_posted) return;
            eng_posted = false;
            const uint32_t epoch = eng_epoch;
            lk.unlock();
            const double t0 = now_ms();
            std::exception_ptr error;
            try {
                engram_read_all();
            } catch (...) {
                error = std::current_exception();
            }
            const double ms = now_ms() - t0;
            __atomic_store_n(eng_flag, epoch, __ATOMIC_RELEASE);
            lk.lock();
            eng_error = error;
            eng_read_ms = ms;
            eng_done = true;
            eng_cv.notify_all();
        }
    }
    /// step_body: this step's rows (engram_ids filled), read beside the step's first layer; returns the epoch
    uint32_t post_engram() {
        std::unique_lock<std::mutex> lk(eng_mu);
        eng_cv.wait(lk, [&] { return eng_done; });   // a previous step that threw may still be reading
        ++eng_epoch;
        eng_done = false;
        eng_posted = true;
        eng_error = nullptr;
        eng_cv.notify_all();
        return eng_epoch;
    }
    /// after the step's sync: the read time, and the read's error if it failed
    void finish_engram(Timing& tm) {
        std::unique_lock<std::mutex> lk(eng_mu);
        eng_cv.wait(lk, [&] { return eng_done; });
        tm.engram_ms = eng_read_ms;
        if (eng_error) {
            std::exception_ptr e = eng_error;
            eng_error = nullptr;
            std::rethrow_exception(e);
        }
    }

    /// Engram.forward for layer l (engram layer li) from the rows engram_read put on the device.
    void engram(int l, int li) {
        const auto& hs = pack.engram_hash();
        const int cols = (hs.max_ngram - 1) * hs.n_heads;
        const uint8_t* w = eng_dev + (size_t) li * kEngRows * (256 + 8);
        ops::engram_dequant(w, w + kEngRows * 256, cols, eng_vals, st);
        fp8_linear(eng_vals, L[l].eng_wkv, eng_kv);
        ops::engram_apply(h, eng_kv, L[l].eng_qw, L[l].eng_kw, kNormEps, 1, st);
    }

    // ------------------------------------------------------------------------------------- indexer
    /// Top-k compressed positions for this layer (Indexer.forward, decode with one query), offset by kWindow,
    /// written after the window part of idx_dev.
    void indexer(int l, bool have_latent) {
        auto& y = L[l];
        const int ratio = y.ratio;
        const int64_t tcap = ratio == 1 ? tcap1 : tcap2;
        if (is_kv_source(l) && have_latent) {
            ops::bf16_linear(latent, nullptr, y.idx_wk, kHeadDim, kIndexDim, ik, nullptr, st);
            ops::rmsnorm(ik, y.idx_knorm, ik, kIndexDim, kNormEps, 1, st);
            ops::rope_device(ik, 1, kIndexDim, rope_yarn, dp + 1, 1 - ratio, false, st);
            ops::fp4_quant_inplace(ik, kIndexDim, 32, false, st);
            ops::row_copy_device(y.idx_keys, ik, kIndexDim * 2, dp + 1, ratio, max_seq + 1, st);   // row pos / ratio
        }
        if (is_kv_source(l)) cur_index_k = y.idx_keys;
        fp8_linear(qr, y.idx_wq_b, iq);
        ops::rope_device(iq, kIndexHeads, kIndexDim, rope_yarn, dp + 1, 0, false, st);
        ops::fp4_quant_inplace(iq, kIndexHeads * kIndexDim, 32, false, st);
        ops::bf16_linear(xa, nullptr, y.idx_wp, kDim, kIndexHeads, iw_raw, nullptr, st);
        ops::scale_bf16(iw_raw, (float) (std::pow(kIndexDim, -0.5) * std::pow(kIndexHeads, -0.5)), iw, kIndexHeads, st);
        // the candidate layer selects blocks from its own unmasked scores; the layers after it mask with them.
        // t = (pos + 1) / ratio on the device: no work while it is 0 (the first incomplete group)
        const uint8_t* cand = l > kCandidateLayer ? cand_dev : nullptr;
        kernels::indexer_topk_device(iq, cur_index_k, dp + 1, ratio, tcap, iw, cand, kIndexTopK, kWindow, scores,
                                     idx_dev + kWindow, st);
        if (l == kCandidateLayer)
            kernels::candidate_blocks_device(scores, dp + 1, ratio, tcap, kCandidateBlocks, kCandidateBlock, cand_dev, st);
    }

    // ------------------------------------------------------------------------------------- attention
    void attention(int l) {
        auto& y = L[l];
        const bool yarn = y.ratio > 0;
        const float* rope = yarn ? rope_yarn : rope_plain;
        fp8_linear(xa, y.wq_a, qr);
        ops::rmsnorm(qr, y.q_norm, qr, kQLora, kNormEps, 1, st);
        fp8_linear(qr, y.wq_b, q);
        ops::rope_device(q, kHeads, kHeadDim, rope, dp + 1, 0, false, st);
        // sliding window: row pos % 128
        fp8_linear(xa, y.wkv, kvv);
        ops::rmsnorm(kvv, y.kv_norm, kvv, kHeadDim, kNormEps, 1, st);
        ops::rope_device(kvv, 1, kHeadDim, rope, dp + 1, 0, false, st);
        ops::act_quant_inplace(kvv, kHeadDim, st);
        ops::row_copy_device(y.window, kvv, kHeadDim * 2, dp + 1, 1, kWindow, st);
        // idx_dev holds the window part for the whole step (window_index_device at step start)
        if (y.ratio > 0) {
            const int ratio = y.ratio;
            bool have_latent = false;
            if (is_kv_source(l)) {
                if (ratio == 1) {
                    ops::bf16_linear(xa, nullptr, y.c_wkv, kDim, kHeadDim, latent, nullptr, st);
                    ops::rmsnorm(latent, y.c_norm, latent, kHeadDim, kNormEps, 1, st);
                    have_latent = true;
                } else {   // ratio 2: slot pos % 2; a group completes at odd positions (one graph per parity)
                    ops::bf16_linear(xa, nullptr, y.c_wkv, kDim, kHeadDim, nullptr, y.kv_state + parity * kHeadDim, st);
                    ops::bf16_linear(xa, nullptr, y.c_wgate, kDim, kHeadDim, nullptr, y.score_state + parity * kHeadDim,
                                     st);
                    if (parity == ratio - 1) {
                        ops::compress_pool(y.kv_state, y.score_state, ratio, latent, 1, st);
                        ops::rmsnorm(latent, y.c_norm, latent, kHeadDim, kNormEps, 1, st);
                        have_latent = true;
                    }
                }
                cur_comp = y.comp;
            }
            if (is_index_source(l)) indexer(l, have_latent);
            if (have_latent) {
                ops::rope_device(latent, 1, kHeadDim, rope_yarn, dp + 1, 1 - ratio, false, st);
                ops::fp4_quant_inplace(latent, kHeadDim, 16, true, st);
                ops::row_copy_device(y.comp, latent, kHeadDim * 2, dp + 1, ratio, max_seq + 1, st);   // row pos / ratio
            }
            // n_idx = 128 + min(512, (pos + 1) / ratio): this group's index source wrote them (none yet: 0)
            kernels::sparse_attn_decode_device(q, y.window, cur_comp, idx_dev, ratio == 1 ? dp + 2 : dp + 3, y.sink,
                                               (float) std::pow(kHeadDim, -0.5), o, st);
        } else {
            kernels::sparse_attn_decode(q, y.window, nullptr, idx_dev, 1, kWindow, y.sink,
                                        (float) std::pow(kHeadDim, -0.5), o, st);
        }
        ops::rope_device(o, kHeads, kHeadDim, rope, dp + 1, 0, true, st);
        if (y.wo_a8.w) wo_a_grouped_fp8(o, y.wo_a8.w, y.wo_a8.s, oa, st);
        else ops::wo_a_grouped(o, y.wo_a, oa, st);
        fp8_linear(oa, y.wo_b, attn_out);
    }

    // ------------------------------------------------------------------------------------- moe
    void moe(int l) {
        auto& y = L[l];
        int32_t* ids = routes_dev + l * kTopK;
        float* w = weights_dev + l * kTopK;
        kernels::router_topk(xf, 1, y.gate_w, y.gate_bias, ids, w, st);
        // K10 computes VRAM hits and a quota of RAM misses (staged or direct) while the CPU computes the rest.
        ops::to_half_fp8q(xf, x_half_dev, kDim, st);
        const bool tier = vram && vram->slots() > 0;
        const auto* ram = host && host->experts_dev() ? host->experts_dev() + (size_t) l * kExperts : nullptr;
        // prefetch: layer l's guesses were planned and copied during layer l - 1 (none for layer 0)
        const bool pf = prefetch && ram && zc_blobs;
        db->publish(x_half_dev, ids, w, 1, tier ? vram->res_dev() + (size_t) l * kExperts : nullptr, gpu_sel,
                    (uint32_t) (l + 1), st, tier ? vram->experts_dev() : nullptr, ram,
                    zc_quota ? zc_quota.get() + l : nullptr,
                    ram ? zc_stage.get() : nullptr, ram && zc_blobs ? zc_blobs.get() + (size_t) l * kExperts : nullptr,
                    pf && l > 0 ? prefetch->ids(l) : nullptr, pf && l > 0 ? prefetch->descs(l) : nullptr,
                    pf ? prefetch->guesses() : 0);
        // layer l + 1's guesses from this layer's expert input, and their copy, beside the rest of this layer
        if (pf && l + 1 < kLayers)
            prefetch->plan(l + 1, xf, L[l + 1].gate_w, L[l + 1].gate_bias,
                           tier ? vram->res_dev() + (size_t) (l + 1) * kExperts : nullptr,
                           host->experts_dev() + (size_t) (l + 1) * kExperts,
                           zc_blobs.get() + (size_t) (l + 1) * kExperts, st);
        const bool staged = ram && zc_stage;
        if (staged) zc_stage->fork_copy(st);
        ck(cudaMemsetAsync(routed, 0, kDim * sizeof(float), st), "routed");
        if (pf && l > 0) prefetch->join(l, st);   // the GPU computes the prefetched experts below
        if (!staged && (tier || ram))
            kernels::exl3_moe_decode((const __half*) x_half_dev, 1, gpu_sel, w, kTopK, db->gpu_experts(), routed,
                                     vram ? vram->workspace() : zc_workspace.get(), VramExperts::kWorkspaceBytes, st);
        // Shared expert overlaps the staging copy and CPU work. One K10 call preserves route reduction order.
        fp8_linear(xf, y.sh_w1, g);
        fp8_linear(xf, y.sh_w3, u);
        ops::swiglu(g, u, kSwigluLimit, sh_h, kMoeInter, st);
        fp8_linear(sh_h, y.sh_w2, sh_out);
        if (staged) {
            zc_stage->join(st);
            kernels::exl3_moe_decode((const __half*) x_half_dev, 1, gpu_sel, w, kTopK, db->gpu_experts(), routed,
                                     vram ? vram->workspace() : zc_workspace.get(), VramExperts::kWorkspaceBytes, st);
        }
        db->wait_add(routed, 1, (uint32_t) (l + 1), st);
        // layer l + 1's guesses are read by its publish, and the next layer overwrites xf, which they read
        if (pf && l + 1 < kLayers) prefetch->ready(l + 1, st);
        ops::add_f32_bf16(routed, sh_out, ffn_out, kDim, st);
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
    static constexpr int kEngIoThreads = 128;   ///< per table; both tables read at once

    /// DS41_PF_PROFILE=1: the GPU time of prefill by phase, printed at its end. Events in stream order; the time
    /// between two marks goes to the phase of the first one, so host waits (the routes' sort, expert copies not yet
    /// issued) count in the phase during which the GPU stood idle.
    struct PfProfile {
        enum { kEngram, kHc, kProj, kComp, kIndexer, kAttn, kOut, kRouter, kExperts, kShared, kHead, kN };
        bool on = false;
        std::vector<cudaEvent_t> ev;
        std::vector<int> cat;
        size_t n = 0;
        void mark(int c) {
            if (!on) return;
            if (n == ev.size()) {
                cudaEvent_t e;
                ck(cudaEventCreate(&e), "profile event");
                ev.push_back(e);
                cat.push_back(0);
            }
            ck(cudaEventRecord(ev[n], 0), "profile mark");
            cat[n++] = c;
        }
        void report() {
            if (!on || n < 2) return;
            static const char* names[kN] = {"engram", "hc mixes + norms", "q/kv projections", "compressor",
                                             "indexer", "attention", "o projection", "routing + sort",
                                             "routed experts", "shared expert + hc post", "head"};
            ck(cudaEventSynchronize(ev[n - 1]), "profile end");
            double t[kN] = {}, total = 0;
            for (size_t i = 0; i + 1 < n; ++i) {
                float ms = 0;
                ck(cudaEventElapsedTime(&ms, ev[i], ev[i + 1]), "profile time");
                t[cat[i]] += ms;
                total += ms;
            }
            std::fprintf(stderr, "ds41 prefill profile (GPU timeline, %.0f ms):", total);
            for (int c = 0; c < kN; ++c) std::fprintf(stderr, " %s %.0f ms (%.0f%%);", names[c], t[c], 100 * t[c] / total);
            std::fprintf(stderr, "\n");
            n = 0;
        }
    } pfp;

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
            *eng_kv, *final_x, *latent, *ik, *attn_kv, *wo_a_deq;
        float *attn_pre, *attn_post, *attn_comb, *ftmp, *router_logits, *ckv, *csc, *nll_logits;
        int32_t* idx;
        uint8_t* eng_dev;
        kernels::Exl3Expert* desc;
        void *k2_ws, *k12_ws, *k14_ws, *k15_ws;
        size_t k12_bytes = 0, k14_bytes = 0, k15_bytes = 0;
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
        // one layer's wo_a dequantized from FP8 (64 MiB), when the pack keeps it FP8
        p.wo_a_deq = carve<bf16>(base, u, L[0].wo_a8.w ? (size_t) kOGroups * kOLora * (kHeads * kHeadDim / kOGroups) : 1);
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
        p.k15_bytes = kernels::hc_mixes_pre_rows_workspace_bytes(sub);
        p.k15_ws = carve<uint8_t>(base, u, std::max<size_t>(p.k15_bytes, 1));
        return (u + 255) & ~(size_t) 255;
    }

    /// Chooses the pass and sub-batch sizes and gets the scratch and the ring: lent VRAM tier slots (up to 90% of the
    /// tier, as upstream) and free VRAM, in that order of preference. Larger passes first (each pass copies every
    /// expert to the GPU once, so the pass size decides the prefill speed of long prompts); for a pass, the largest
    /// sub-batch (up to opt.prefill_batch) and ring (opt.prefill_ring, at least 16 slots) that fit.
    void prefill_begin(int n) {
        pf = Prefill{};
        if (const char* v = std::getenv("DS41_PF_PROFILE")) pfp.on = v[0] && v[0] != '0';
        // a ring slot holds any expert of the pack (upstream: MAXBLOB); the tier's slots may be smaller
        for (int l = 0; l < kLayers; ++l)
            for (int e = 0; e < kExperts; ++e) pf.slot_bytes = std::max<size_t>(pf.slot_bytes, pack.expert(l, e).bytes);
        pf.slot_bytes = (pf.slot_bytes + 255) & ~(size_t) 255;
        // the bytes of the last 90% of the tier's slots (upstream lends up to 90% of the tier)
        const size_t lendable = vram ? vram->tail_bytes(vram->slots() * 9 / 10) : 0;
        size_t free_b = 0, total_b = 0;
        ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        const size_t spare = free_b > (512ull << 20) ? free_b - (512ull << 20) : 0;   // cuBLAS and K13 scratch
        size_t scratch = 0, ring = 0;
        int plan = -1;   // 0: both lent; 1: scratch lent, ring cudaMalloc; 2: ring lent, scratch cudaMalloc; 3: both cudaMalloc
        int cap = std::max(1, std::min(n, opt.prefill_chunk)), sub = 0;
        for (; plan < 0; cap /= 2) {
            // the smallest pass is 16 tokens, or the whole prompt when it is shorter
            if (cap < std::min(n, 16)) throw std::runtime_error("ds41 prefill: not enough VRAM for a 16-token pass");
            for (sub = std::max(1, std::min(cap, opt.prefill_batch)); plan < 0 && sub >= std::min(cap, 512); sub /= 2) {
                scratch = layout(pf, nullptr, cap, sub);
                for (int r = std::max(opt.prefill_ring, kMinRing); plan < 0 && r >= kMinRing; r /= 2) {
                    pf.ring_slots = r;
                    ring = (size_t) r * pf.slot_bytes;
                    if (scratch + ring <= lendable) plan = 0;
                    else if (scratch <= lendable && ring <= spare) plan = 1;
                    else if (ring <= lendable && scratch <= spare) plan = 2;
                    else if (scratch + ring <= spare) plan = 3;
                }
                if (plan >= 0) break;
            }
            if (plan >= 0) break;
        }
        pf.cap = cap;
        pf.sub = sub;
        const size_t lend_bytes = plan == 0 ? scratch + ring : plan == 1 ? scratch : plan == 2 ? ring : 0;
        if (lend_bytes) pf.lent = vram->lend_bytes(lend_bytes);
        uint8_t* scratch_base = plan <= 1 ? pf.lent : nullptr;
        if (plan >= 2) ck(cudaMalloc((void**) &pf.own_scratch, scratch), "prefill scratch");
        if (plan == 1 || plan == 3) ck(cudaMalloc((void**) &pf.own_ring, ring), "prefill ring");
        if (!scratch_base) scratch_base = pf.own_scratch;
        layout(pf, scratch_base, cap, sub);
        pf.ring = plan == 0 ? pf.lent + scratch : plan == 2 ? pf.lent : pf.own_ring;
        std::fprintf(stderr, "ds41 prefill: pass %d tokens, sub-batch %d; scratch %.2f GiB %s; ring %d slots %s; %d tier "
                     "slots lent\n", cap, sub, scratch / 1073741824.0, plan <= 1 ? "in lent slots" : "cudaMalloc",
                     pf.ring_slots, plan == 0 || plan == 2 ? "in lent slots" : "cudaMalloc",
                     lend_bytes ? vram->lent() : 0);
        // pinned host buffers
        if (pfh.cap < cap) {
            if (pfh.base) cudaFreeHost(pfh.base);
            pfh.base = nullptr;   // a failed allocation below must not leave the freed block behind
            pfh.cap = 0;
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
                                                 std::max(1, opt.prefill_threads), std::max(1, opt.prefill_host_buffers),
                                                 prefill_unbuffered());
    }


    /// Upstream's file-tier rule (v0.1.40 file_tier_unbuffered / file_cache_keeps, #577): the experts outside the
    /// RAM tier are read with O_DIRECT when the file cache could not keep them anyway (available RAM, after the RAM
    /// tier, less 4 GiB, below their bytes); otherwise through the file cache. The VRAM tier's experts count too:
    /// prefill lends their slots and refills them. DS41_UNBUFFERED=0 / 1 forces it.
    bool prefill_unbuffered() {
        if (const char* v = std::getenv("DS41_UNBUFFERED"); v && v[0]) return v[0] != '0';
        uint64_t read = 0;
        for (int l = 0; l < kLayers; ++l)
            for (int e = 0; e < kExperts; ++e)
                if (!(host && host->slot_of(l, e) >= 0)) read += pack.expert(l, e).bytes;
        const uint64_t avail = auto_ram_budget(0), margin = 4ull << 30;
        const bool keeps = avail > margin && avail - margin >= read;
        std::fprintf(stderr, "ds41 prefill: %.1f GiB available, %.1f GiB of experts read from the pack: %s\n",
                     avail / 1073741824.0, read / 1073741824.0,
                     keeps ? "through the file cache" : "unbuffered (O_DIRECT): the file cache could not keep them");
        return !keeps;
    }

    /// Returns the lent slots (or frees the scratch). After an error the stream is stopped without draining (its
    /// unreleased jobs would never complete).
    /// The destructor's form: the VRAM tier (owner of lent slots) is gone already
    void prefill_end_quiet() {
        estream.reset();
        if (pf.own_scratch) cudaFree(pf.own_scratch);
        if (pf.own_ring) cudaFree(pf.own_ring);
        pf.lent = pf.own_scratch = pf.own_ring = nullptr;
    }

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

    /// K7's hyper-connection mixes for T rows (task K15)
    void hc_rows(const bf16* x, int T, const float* fn, const float* scale, const float* base, const float* pre_in,
                 bf16* y, float* pre, float* post, float* comb) {
        kernels::hc_mixes_pre_rows(x, T, fn, scale, base, pre_in, y, pre, post, comb, pf.k15_ws, pf.k15_bytes, 0);
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
        pfp.mark(PfProfile::kProj);
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
            pfp.mark(PfProfile::kComp);
            if (is_kv_source(l)) compress_rows(l, T, p0);
            pfp.mark(PfProfile::kIndexer);
            if (is_index_source(l)) indexer_rows(l, T, p0, b0);
            pfp.mark(PfProfile::kAttn);
            const int c_end = (p0 + T) / y.ratio;   // compressed rows any query of the chunk may see
            if (c_end)
                ck(cudaMemcpyAsync(pf.attn_kv, cur_comp, (size_t) c_end * kHeadDim * 2, cudaMemcpyDeviceToDevice, 0),
                   "attention kv: compressed rows");
            win_base = c_end;
            n_idx = kWindow + kIndexTopK;
            topk = pf.topk + (size_t) b0 * kIndexTopK;
        }
        // window rows: the ring's positions before the chunk, then the chunk's own; then the ring for decode
        pfp.mark(PfProfile::kAttn);
        const int prev = std::min(p0, kWindow - 1);
        prefill::window_gather(y.window, p0, prev, pf.attn_kv + (size_t) (win_base + kWindow - 1 - prev) * kHeadDim);
        ck(cudaMemcpyAsync(pf.attn_kv + (size_t) (win_base + kWindow - 1) * kHeadDim, pf.kvv, (size_t) T * kHeadDim * 2,
                           cudaMemcpyDeviceToDevice, 0), "attention kv: window rows");
        prefill::window_scatter(y.window, pf.kvv, p0, T);
        prefill::attn_index_rows(T, p0, win_base, topk, kIndexTopK, pf.idx, n_idx);
        kernels::sparse_attn_prefill(pf.q, pf.attn_kv, pf.idx, T, n_idx, y.sink, (float) std::pow(kHeadDim, -0.5), pf.o,
                                     0);
        pfp.mark(PfProfile::kOut);
        prefill::rope_rows(pf.o, T, kHeads, kHeadDim, table, p0, 1, true);
        const bf16* wo_a = y.wo_a;
        if (y.wo_a8.w) {   // FP8 pack: the BF16 weight convert.py would store, then the same GEMM
            dequant_wo_a(y.wo_a8.w, y.wo_a8.s, pf.wo_a_deq);
            wo_a = pf.wo_a_deq;
        }
        prefill::wo_a_grouped_rows(pf.o, wo_a, T, pf.oa, pf.ftmp);
        fp8_rows(pf.oa, T, y.wo_b, pf.attn_out);
    }


    /// decode: moe(), routing part, for the sub-batch at pass row b0: router (GPU logits), the routes into the pass
    /// arrays, and the experts' input (FP8-quantized, fp16).
    void moe_route(int l, int T, int b0) {
        auto& y = L[l];
        const bf16* xf = pf.xf + (size_t) b0 * kDim;
        pfp.mark(PfProfile::kRouter);
        prefill::bf16_gemm(xf, y.gate_w, T, kDim, kExperts, nullptr, pf.router_logits);
        prefill::route_rows(pf.router_logits, y.gate_bias, T, pf.ids + (size_t) b0 * kTopK, pf.wts + (size_t) b0 * kTopK);
        ops::to_half_fp8q(xf, pf.x_half + (size_t) b0 * kDim, T * kDim);
    }

    /// decode: moe(), routed experts, for all S tokens of the pass at once: the host sorts the (token, expert) rows
    /// by expert, VRAM tier experts first (one K12 call from their slots), then the streamed experts in job order
    /// (one K12 call per group of ring slots, each released when its call has run). Each expert is copied once.
    void moe_experts(int l, int S) {
        pfp.mark(PfProfile::kRouter);
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
        pfp.mark(PfProfile::kExperts);
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
        pfp.mark(PfProfile::kShared);
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
        const size_t eng_table = (size_t) pf.cap * kEngRows * (256 + 8);
        // the rows of each table first (cheap), then both tables' reads at once, each on its reader's threads: the
        // reads are random 4 KiB ones, so the depth of the queue sets the speed
        std::vector<std::vector<int64_t>> uniq(done.size());
        std::vector<std::thread> readers;
        std::vector<char> started(done.size(), 0);
        try {
            const auto& hs = pack.engram_hash();
            const int cols = (hs.max_ngram - 1) * hs.n_heads;
            if (cols != kEngRows) throw std::runtime_error("engram: the prefill buffers assume 24 rows per token");
            size_t t = 0;
            for (int l = 0; l < kLayers; ++l) {
                if (!is_engram_layer(l)) continue;
                const int li = (int) t++;
                std::vector<int64_t>& all = eng_ids_pf[li];
                for (int i = 0; i < S; ++i) {
                    engram_ids(l, li, p0 + i);
                    std::copy(eng_ids[li].begin(), eng_ids[li].begin() + cols, all.begin() + (size_t) i * cols);
                }
                std::vector<int64_t>& u = uniq[li];
                u.assign(all.begin(), all.begin() + (size_t) S * cols);
                std::sort(u.begin(), u.end());
                u.erase(std::unique(u.begin(), u.end()), u.end());
                ptm->engram_rows += (int64_t) S * cols;
                ptm->engram_unique += (int64_t) u.size();
            }
            for (size_t li = 0; li < done.size(); ++li) {
                readers.emplace_back([&, li, cols] {
                    try {
                        const std::vector<int64_t>& u = uniq[li];
                        const std::vector<int64_t>& all = eng_ids_pf[li];
                        std::vector<uint8_t> uw(u.size() * 256), us(u.size() * 8);
                        const size_t batch = (size_t) kEngBatch * kEngRows;
                        for (size_t r = 0; r < u.size(); r += batch)
                            eng_rows_pf[li]->read({u.data() + r}, (int) std::min(batch, u.size() - r),
                                                  {uw.data() + r * 256}, {us.data() + r * 8});
                        uint8_t* w = pfh.eng + li * eng_table;
                        uint8_t* sc = w + (size_t) pf.cap * kEngRows * 256;
                        for (size_t i = 0; i < (size_t) S * cols; ++i) {
                            const size_t k = (size_t) (std::lower_bound(u.begin(), u.end(), all[i]) - u.begin());
                            std::memcpy(w + i * 256, uw.data() + k * 256, 256);
                            std::memcpy(sc + i * 8, us.data() + k * 8, 8);
                        }
                        done[li].set_value();
                    } catch (...) {
                        done[li].set_exception(std::current_exception());
                    }
                });
                started[li] = 1;
            }
        } catch (...) {
            for (size_t li = 0; li < done.size(); ++li)
                if (!started[li]) done[li].set_exception(std::current_exception());
        }
        for (auto& r : readers) r.join();
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
                    pfp.mark(PfProfile::kEngram);
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
                pfp.mark(PfProfile::kHc);
                hc_rows(h, T, y.hc_attn_fn, y.hc_attn_scale, y.hc_attn_base, pf.pre_mix + b0 * kHc, pf.xa, pf.attn_pre,
                        pf.attn_post, pf.attn_comb);
                ops::rmsnorm(pf.xa, y.attn_norm, pf.xa, kDim, kNormEps, T);
                attention_rows(l, T, p0 + b0, b0);
                pfp.mark(PfProfile::kHc);
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
            if (progress && !progress(c0 + (int) ((int64_t) S * (l + 1) / kLayers), (int) tokens.size()))
                throw PrefillCancelled();   // prefill_end stops the stream and syncs; prefill() resets
        }
        const int n = (int) tokens.size();
        if (nll) {
            for (int i = 0; i < S; ++i) pfh.targets[i] = c0 + i + 1 < n ? tokens[c0 + i + 1] : -1;
            ck(cudaMemcpyAsync(pf.targets, pfh.targets, (size_t) S * 4, cudaMemcpyHostToDevice, 0), "targets up");
        }
        pfp.mark(PfProfile::kHead);
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
        pfp.mark(PfProfile::kHead);
        pfp.report();
        if (S >= kStreamAll) estream->drain();   // every job of the pass was consumed
    }

    int prefill(const std::vector<int>& tokens, int pos, std::vector<float>* nll, PrefillTiming& pt) {
        usable();
        if (pos != (int) history.size()) throw std::runtime_error("prefill must continue at the tokens fed so far");
        const int n = (int) tokens.size();
        if (n == 0) throw std::runtime_error("prefill of no tokens");
        if (pos + n > max_seq) throw std::runtime_error("prefill past max_seq");
        for (int t : tokens) check_token(t);
        if (opt.prefill_chunk <= 0) {
            try {
                return prefill_steps(tokens, pos, nll, pt);
            } catch (const PrefillCancelled&) {
                reset();
                throw;
            }
        }
        pf_started = false;
        try {
            return prefill_batched(tokens, pos, nll, pt);
        } catch (const PrefillCancelled&) {
            // the pass stopped between layers and prefill_end synced the device: the caches hold rows past the old
            // position, which a reset makes harmless (every row is written again before it is read)
            reset();
            throw;
        } catch (...) {
            if (pf_started) broken = true;   // a pass may have written part of the layers' caches and the history
            throw;
        }
    }

    int prefill_steps(const std::vector<int>& tokens, int pos, std::vector<float>* nll, PrefillTiming& pt) {
        const int n = (int) tokens.size();
        pt = PrefillTiming{};
        ptm = &pt;
        const double t0 = now_ms();
        if (nll) nll->assign(n - 1, 0.0f);
        int next = -1;
        {   // token by token
            Timing tm;
            for (int i = 0; i < n; ++i) {
                next = step(tokens[i], pos + i, nullptr, tm);
                if (progress && !progress(i + 1, n)) throw PrefillCancelled();
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
    }

    int prefill_batched(const std::vector<int>& tokens, int pos, std::vector<float>* nll, PrefillTiming& pt) {
        const int n = (int) tokens.size();
        pt = PrefillTiming{};
        ptm = &pt;
        const double t0 = now_ms();
        if (nll) nll->assign(n - 1, 0.0f);
        try {
            prefill_begin(n);   // a partial failure returns what it took (lent slots, scratch, ring)
            pt.chunk_tokens = pf.cap;
            pt.sub_batch = pf.sub;
            if (fault("prefill")) throw std::runtime_error("ds41 test fault: before the first prefill pass");
            pf_started = true;
            if (fault("prefill_pass")) throw std::runtime_error("ds41 test fault: inside a prefill pass");
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
    /// The GPU work of one decode step on stream st: every position dependent value comes from dp (staged from hp
    /// by the first node), so one capture serves every position with the same parity and indexer capacities.
    /// dump: eager only (it reads the device between layers).
    void enqueue_step(StepDump* dump) {
        ck(cudaMemcpyAsync(dp, hp, sizeof(StepParams), cudaMemcpyHostToDevice, st), "step params");
        ops::window_index_device(dp + 1, idx_dev, st);
        ops::embed_device(embed, dp, h, st);
        // the stream's collapse weights: one-hot at the first layer, then the previous layer's ffn_pre. Read in place:
        // a 16-byte copy node per layer left a 12-24 us gap on the GPU chain (nsys, RTX 5090 Laptop)
        const float* pre_in = one_hot_dev;
        int eng_i = 0;
        for (int l = 0; l < kLayers; ++l) {
            auto& y = L[l];
            if (is_engram_layer(l) && eng_i == 0) {   // every table's rows, read beside the layers before
                wait_engram_k<<<1, 1, 0, st>>>(eng_flag_dev, dp);
                ck(cudaGetLastError(), "engram wait");
                ck(cudaMemcpyAsync(eng_dev, eng_host, (size_t) n_eng * kEngRows * (256 + 8), cudaMemcpyHostToDevice,
                                   st), "engram rows");
            }
            if (is_engram_layer(l)) engram(l, eng_i++);
            const bool dbg_layer = dbg && (l == 1 || l == 2);
            if (dbg_layer) dbg_write(h, kHc * kDim);                    // block input (after engram)
            // attention sub-block: h -> h2
            kernels::hc_mixes_pre(h, 1, y.hc_attn_fn, y.hc_attn_scale, y.hc_attn_base, pre_in, xa, attn_pre, attn_post,
                                  attn_comb, st);
            ops::rmsnorm(xa, y.attn_norm, xa, kDim, kNormEps, 1, st);
            if (dbg_layer) dbg_write(xa, kDim);                         // attention input
            attention(l);
            if (dbg_layer) dbg_write(attn_out, kDim);                   // attention output
            ops::hc_post(attn_out, h, attn_post, attn_comb, h2, 1, st);
            // ffn sub-block: h2 -> h
            kernels::hc_mixes_pre(h2, 1, y.hc_ffn_fn, y.hc_ffn_scale, y.hc_ffn_base, attn_pre, xf, ffn_pre, ffn_post,
                                  ffn_comb, st);
            ops::rmsnorm(xf, y.ffn_norm, xf, kDim, kNormEps, 1, st);
            if (dbg_layer) dbg_write(xf, kDim);                         // ffn input
            moe(l);
            if (dbg_layer) dbg_write(ffn_out, kDim);                    // ffn output
            ops::hc_post(ffn_out, h2, ffn_post, ffn_comb, h, 1, st);
            pre_in = ffn_pre;   // read by the next layer before its ffn sub-block writes ffn_pre again
            if (dump) {
                ck(cudaStreamSynchronize(st), "dump hidden");
                ck(cudaMemcpy(dump->hidden.data() + (size_t) l * kHc * kDim, h, kHc * kDim * 2, cudaMemcpyDeviceToHost),
                   "dump hidden");
            }
        }
        ops::hc_pre(h, pre_in, final_x, 1, st);
        ops::rmsnorm(final_x, final_norm, final_x, kDim, kNormEps, 1, st);
        ops::bf16_linear(final_x, nullptr, head, kDim, kVocab, nullptr, logits, st);
        ops::argmax_logits(logits, d_next, st);
        ck(cudaMemcpyAsync(hp_next, d_next, sizeof(int), cudaMemcpyDeviceToHost, st), "next token");
        ck(cudaMemcpyAsync(lg_pinned, logits, (size_t) kVocab * 4, cudaMemcpyDeviceToHost, st), "logits");
        ck(cudaMemcpyAsync(routes_pinned, routes_dev, kLayers * kTopK * 4, cudaMemcpyDeviceToHost, st), "routes");
    }

    /// Indexer capacity of a graph: the next power of two of t (at least 1), at most the allocated rows
    static int64_t cap_of(int64_t t, int64_t rows) {
        int64_t c = 1;
        while (c < t) c <<= 1;
        return std::min(c, rows);
    }

    void reset() {
        usable();
        history.clear();
        verify_pending = verify_failed = false;
    }

    void alloc_snapshots() {
        snap_comp_floats = 0;
        for (auto& y : L)
            if (y.kv_state) snap_comp_floats += 2 * (size_t) y.ratio * kHeadDim;
        snaps.resize(std::max(0, opt.snapshots));
        for (auto& sn : snaps) {
            sn.win = dalloc_own<bf16>((size_t) kLayers * kWindow * kHeadDim);
            sn.comp = dalloc_own<float>(std::max<size_t>(1, snap_comp_floats));
        }
    }

    Snapshot& snapshot_slot(int slot) {
        if (slot < 0 || slot >= (int) snaps.size())
            throw std::invalid_argument("ds41 snapshot: slot " + std::to_string(slot) + " does not exist");
        return snaps[slot];
    }

    /// Copies the window rings and compressor states between the layers and a slot (to_slot: save). The device is
    /// idle between calls (each step ends in a sync, a prefill in prefill_end's), so one stream and one sync suffice.
    void snapshot_copy(Snapshot& sn, bool to_slot) {
        const size_t win = (size_t) kWindow * kHeadDim * sizeof(bf16);
        size_t c = 0;
        for (int l = 0; l < kLayers; ++l) {
            auto& y = L[l];
            uint8_t* slot_win = (uint8_t*) sn.win + (size_t) l * win;
            ck(cudaMemcpyAsync(to_slot ? (void*) slot_win : (void*) y.window, to_slot ? (void*) y.window : (void*) slot_win,
                               win, cudaMemcpyDeviceToDevice, st), "snapshot window");
            if (!y.kv_state) continue;
            const size_t n = (size_t) y.ratio * kHeadDim;
            for (float* state : {y.kv_state, y.score_state}) {
                ck(cudaMemcpyAsync(to_slot ? sn.comp + c : state, to_slot ? state : sn.comp + c, n * sizeof(float),
                                   cudaMemcpyDeviceToDevice, st), "snapshot compressor");
                c += n;
            }
        }
        ck(cudaStreamSynchronize(st), "snapshot");
    }

    void save_snapshot(int slot) {
        usable();
        if (verify_pending) throw std::logic_error("commit the pending verify window first");
        Snapshot& sn = snapshot_slot(slot);
        sn.pos = -1;   // a failed copy leaves the slot empty
        snapshot_copy(sn, true);
        sn.pos = (int) history.size();
    }

    int restore_snapshot(int slot) {
        usable();
        Snapshot& sn = snapshot_slot(slot);
        if (sn.pos < 0) throw std::logic_error("ds41 snapshot: slot " + std::to_string(slot) + " is empty");
        if (sn.pos > (int) history.size())
            throw std::logic_error("ds41 snapshot: slot " + std::to_string(slot) + " is past the current position");
        verify_pending = verify_failed = false;
        snapshot_copy(sn, false);
        history.resize(sn.pos);
        return sn.pos;
    }

    int step(int token, int pos, StepDump* dump, Timing& tm) {
        usable();
        check_token(token);
        if (pos != (int) history.size()) throw std::runtime_error("tokens must be fed in order from position 0");
        if (pos >= max_seq) throw std::runtime_error("position past max_seq");
        bool released = false;
        try {
            return step_body(token, pos, dump, tm, released);
        } catch (...) {
            // nothing reached the device or the worker: undo the history, the step may be retried
            if (released) broken = true;
            else if ((int) history.size() == pos + 1) history.pop_back();
            throw;
        }
    }

    int step_body(int token, int pos, StepDump* dump, Timing& tm, bool& released) {
        tm = Timing{};
        const double t_start = now_ms();
        history.push_back(pack.engram_hash().token_map[token]);
        {
            int li = 0;
            for (int l = 0; l < kLayers; ++l)
                if (is_engram_layer(l)) engram_ids(l, li++, pos);
            if (fault("engram")) throw std::runtime_error("ds41 test fault: engram read");
        }
        const uint32_t engram_epoch = n_eng ? post_engram() : 0;   // read beside layer 0; the graph waits before layer 1
        const double t_swaps = now_ms();
        if (vram) tm.vram_swaps = vram->between_steps();   // the device is idle: the last step ended in a sync
        tm.swaps_ms = now_ms() - t_swaps;
        db->reset();
        if (prefetch) prefetch->begin_step();
        worker_us = 0;
        worker_misses = 0;
        worker_ram = worker_file = worker_ssd = 0;
        worker_prefetched = 0;
        {
            std::lock_guard<std::mutex> lk(mu);
            ++go;
        }
        released = true;
        cv.notify_one();
        if (dump) {
            dump->hidden.assign((size_t) kLayers * kHc * kDim, 0);
            dump->routes.assign(kLayers, {});
            dump->weights.assign(kLayers, {});
        }
        *hp = StepParams{token, pos, pos + 1, (pos + 1) / 2, (int) engram_epoch};
        parity = pos & 1;
        tcap1 = cap_of(pos + 1, max_seq + 1);
        tcap2 = cap_of((pos + 1) / 2, max_seq / 2 + 1);
        if (use_graph && !dump && !dbg) {
            const uint64_t key = (uint64_t) parity | (uint64_t) tcap1 << 1 | (uint64_t) tcap2 << 33;
            auto it = graphs.find(key);
            if (it == graphs.end()) {
                // relaxed: other threads (the tier's copier) keep using CUDA during the capture
                ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed), "begin capture");
                enqueue_step(nullptr);
                cudaGraph_t gr;
                ck(cudaStreamEndCapture(st, &gr), "end capture");
                cudaGraphExec_t ex;
                ck(cudaGraphInstantiate(&ex, gr, 0), "instantiate");
                cudaGraphDestroy(gr);
                it = graphs.emplace(key, ex).first;
                ++graph_captures;
            }
            ck(cudaGraphLaunch(it->second, st), "graph launch");
        } else {
            enqueue_step(dump);
        }
        ck(cudaStreamSynchronize(st), "step");
        if (n_eng) finish_engram(tm);
        if (prefetch && prefetch->failed()) throw std::runtime_error("ds41 prefetch: a DMA copy failed");
        const double t_end = now_ms();
        {
            std::lock_guard<std::mutex> lk(mu);
            if (worker_error) {
                std::exception_ptr e = worker_error;
                worker_error = nullptr;
                std::rethrow_exception(e);
            }
        }
        const int best = *hp_next;
        lg.assign(lg_pinned, lg_pinned + kVocab);
        if (pred_stats && lookahead) predict_tally();
        if (vram) vram->count(routes_pinned, kTopK);
        if (host && host->reserve() > 0) host->end_step(routes_pinned, kTopK);   // publish this step's reads, evict
        if (dump) {
            float wv[kLayers * kTopK];
            ck(cudaMemcpy(wv, weights_dev, sizeof wv, cudaMemcpyDeviceToHost), "dump weights");
            for (int l = 0; l < kLayers; ++l)
                for (int i = 0; i < kTopK; ++i) {
                    dump->routes[l][i] = routes_pinned[l * kTopK + i];
                    dump->weights[l][i] = wv[l * kTopK + i];
                }
            std::vector<int> order(kVocab);
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + 8, order.end(), [&](int a, int b) { return lg[a] > lg[b]; });
            dump->top_logits.clear();
            for (int i = 0; i < 8; ++i) dump->top_logits.push_back({order[i], lg[order[i]]});
        }
        tm.total_ms = now_ms() - t_start;
        tm.end_ms = now_ms() - t_end;
        tm.worker_lead_ms = worker_wake_at.load() - t_start;
        tm.worker_span_ms = worker_end_at.load() - worker_wake_at.load();
        tm.worker_wait_ms = worker_wait.load();
        tm.worker_first_wait_ms = worker_first_wait.load();
        tm.admit_ms = worker_admit.load();
        tm.cpu_experts_ms = worker_us.load() / 1000.0;
        tm.expert_total = kLayers * kTopK;
        tm.expert_hits = tm.expert_total - worker_misses.load();
        tm.ram_experts = worker_ram.load();
        tm.file_experts = worker_file.load();
        tm.ssd_experts = worker_ssd.load();
        tm.prefetched = worker_prefetched.load();
        if (lookahead) {
            const auto st_ = lookahead->take_stats();
            tm.warmed = (int) st_.warmed;
            tm.warmed_useful = (int) st_.useful;
        }
        tm.gpu_ms = tm.total_ms;   // the engram rows are read beside layer 0, not before the step
        return best;
    }
};

#include "verify.cu"
#include "batch.cu"

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

int Engine::step(int token, int pos, StepDump* dump) {
    if (impl_->verify_pending) throw std::logic_error("commit the pending verify window first");
    return impl_->step(token, pos, dump, timing_);
}

int Engine::prefill(const std::vector<int>& tokens, int pos, std::vector<float>* nll) {
    if (impl_->verify_pending) throw std::logic_error("commit the pending verify window first");
    return impl_->prefill(tokens, pos, nll, prefill_timing_);
}

const std::vector<float>& Engine::last_logits() const { return impl_->lg; }

void Engine::set_prefill_progress(PrefillProgress fn) { impl_->progress = std::move(fn); }

void Engine::reset() { impl_->reset(); }

void Engine::save_snapshot(int slot) { impl_->save_snapshot(slot); }

int Engine::restore_snapshot(int slot) { return impl_->restore_snapshot(slot); }

int Engine::snapshot_slots() const { return (int) impl_->snaps.size(); }

int Engine::position() const { return (int) impl_->history.size(); }

}  // namespace strata::ds41
