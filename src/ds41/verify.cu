// Included by engine.cu after Impl. This keeps private engine types in one translation unit.
// No allocation or host wait occurs in enqueue_verify or its kernels.
namespace {
// Save the append-only tail. Rows outside the allocated context are never read.
__global__ void verify_tail_copy(bf16* backup, bf16* cache, const int* params, int width,
                                int ratio, int capacity, int keep, bool restore) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kVerifyMaxTokens * width) return;
    const int row = params[1] / ratio + i / width;
    if (row >= capacity) return;
    const int j = row * width + i % width;
    if (!restore) backup[i] = cache[j];
    else if (row >= (params[1] + keep) / ratio) cache[j] = backup[i];
}
} // namespace

struct Engine::Impl::VerifyWorkspace {
    static constexpr int M = kVerifyMaxTokens;
    std::vector<std::shared_ptr<void>> device, pinned;
    template<class T> T* alloc(size_t n) {
        T* p = dalloc<T>(std::max<size_t>(1, n));
        device.emplace_back(p, [](void* q) { cudaFree(q); }); return p;
    }
    template<class T> T* pin(size_t n) {
        T* p = nullptr;
        ck(cudaHostAlloc((void**)&p, std::max<size_t>(1, n) * sizeof(T), cudaHostAllocDefault), "verify pinned");
        pinned.emplace_back(p, [](void* q) { cudaFreeHost(q); }); return p;
    }
    struct State {
        bf16 *ring = nullptr, *rows = nullptr, *comp_backup = nullptr, *key_backup = nullptr;
        float *kv = nullptr, *score = nullptr, *kv_rows = nullptr, *score_rows = nullptr;
    };
    State state[kLayers];
    bf16 *h, *h2, *xa, *xf, *qr, *q, *kvv, *o, *oa, *attn_out;
    bf16 *g, *u, *sh_h, *sh_out, *ffn_out, *final_x, *eng_vals, *eng_kv;
    float *act, *mix, *apre, *apost, *acomb, *fpre, *fpost, *fcomb, *routed, *logits;
    uint16_t* half;
    int32_t *routes, *routes_host, *selected, *indices;
    float *weights, *logits_host;
    int *params, *params_host, *next, *next_host;
    uint8_t *candidates, *eng, *eng_host;
    size_t eng_bytes;
    std::unique_ptr<ExpertDoorbell> db;
    std::map<std::array<int64_t, 4>, cudaGraphExec_t> graphs;
    std::vector<int> tokens;
    int pos = 0, m = 0;
    bool warmed = false;
    VerifyWorkspace(Impl& e) {
        const int S = e.max_seq;
        params = alloc<int>(M * 4); params_host = pin<int>(M * 4);
        next = alloc<int>(M); next_host = pin<int>(M);
        h = alloc<bf16>(M*kHc*kDim); h2 = alloc<bf16>(M*kHc*kDim);
        xa = alloc<bf16>(M*kDim); xf = alloc<bf16>(M*kDim);
        qr = alloc<bf16>(M*kQLora); q = alloc<bf16>(M*kHeads*kHeadDim);
        kvv = alloc<bf16>(M*kHeadDim); o = alloc<bf16>(M*kHeads*kHeadDim);
        oa = alloc<bf16>(M*kOGroups*kOLora); attn_out = alloc<bf16>(M*kDim);
        g = alloc<bf16>(M*kMoeInter); u = alloc<bf16>(M*kMoeInter); sh_h = alloc<bf16>(M*kMoeInter);
        sh_out = alloc<bf16>(M*kDim); ffn_out = alloc<bf16>(M*kDim); final_x = alloc<bf16>(M*kDim);
        eng_vals = alloc<bf16>(M*24*256); eng_kv = alloc<bf16>(M*(kHc+1)*kDim);
        act = alloc<float>(M*8192); mix = alloc<float>(M*kHc);
        apre = alloc<float>(M*kHc); apost = alloc<float>(M*kHc); acomb = alloc<float>(M*kHc*kHc);
        fpre = alloc<float>(M*kHc); fpost = alloc<float>(M*kHc); fcomb = alloc<float>(M*kHc*kHc);
        routed = alloc<float>(M*kDim); logits = alloc<float>(M*kVocab); logits_host = pin<float>(M*kVocab);
        half = alloc<uint16_t>(M*kDim);
        routes = alloc<int32_t>(kLayers*M*kTopK); routes_host = pin<int32_t>(kLayers*M*kTopK);
        weights = alloc<float>(kLayers*M*kTopK); selected = alloc<int32_t>(M*kTopK);
        indices = alloc<int32_t>(M*(kWindow+kIndexTopK)); candidates = alloc<uint8_t>(size_t(M)*(S+1));
        eng_bytes = size_t(e.n_eng)*kEngRows*(256+8);
        eng = alloc<uint8_t>(M*eng_bytes); eng_host = pin<uint8_t>(M*eng_bytes);
        db = std::make_unique<ExpertDoorbell>(M, kTopK, kDim);
        for (int l = 0; l < kLayers; ++l) {
            auto& s = state[l];
            s.ring = alloc<bf16>(kWindow*kHeadDim); s.rows = alloc<bf16>(M*kHeadDim);
            if (is_kv_source(l)) {
                s.comp_backup = alloc<bf16>(M*kHeadDim); s.key_backup = alloc<bf16>(M*kIndexDim);
                if (e.L[l].ratio > 1) {
                    s.kv = alloc<float>(2*kHeadDim); s.score = alloc<float>(2*kHeadDim);
                    s.kv_rows = alloc<float>(M*2*kHeadDim); s.score_rows = alloc<float>(M*2*kHeadDim);
                }
            }
        }
    }
    ~VerifyWorkspace() {
        for (auto& entry : graphs) cudaGraphExecDestroy(entry.second);
    }
};

void Engine::Impl::enqueue_verify(int m) {
    auto& v = *verify_ws;
    auto copy = [&](void* dst, const void* src, size_t bytes) {
        ck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, st), "verify copy");
    };
    auto linear = [&](const bf16* x, const Fp8& w, bf16* out) {
        fp8_quantize_activation_f32((const uint16_t*)x, m, w.k, v.act, st);
        fp8_block_gemv_q(v.act, m, w.k, w.w, w.s, w.n, (uint16_t*)out, st);
    };
    ck(cudaMemcpyAsync(v.params, v.params_host, m*4*sizeof(int), cudaMemcpyHostToDevice, st), "verify params");
    if (n_eng) ck(cudaMemcpyAsync(v.eng, v.eng_host, m*v.eng_bytes, cudaMemcpyHostToDevice, st), "verify engram");
    for (int t = 0; t < m; ++t) {
        ops::window_index_device(v.params+t*4+1, v.indices+t*(kWindow+kIndexTopK), st);
        ops::embed_device(embed, v.params+t*4, v.h+t*kHc*kDim, st);
        copy(v.mix+t*kHc, one_hot_dev, kHc*sizeof(float));
    }
    const bf16 *comp = nullptr, *keys = nullptr;
    int eng_i = 0;
    for (int l = 0; l < kLayers; ++l) {
        auto& y = L[l]; auto& s = v.state[l];
        // Formal rings and compressor tails stay unchanged until commit.
        copy(s.ring, y.window, kWindow*kHeadDim*sizeof(bf16));
        if (is_kv_source(l)) {
            const int capacity = max_seq/y.ratio+1;
            verify_tail_copy<<<(VerifyWorkspace::M*kHeadDim+255)/256,256,0,st>>>(
                s.comp_backup, y.comp, v.params, kHeadDim, y.ratio, capacity, 0, false);
            verify_tail_copy<<<(VerifyWorkspace::M*kIndexDim+255)/256,256,0,st>>>(
                s.key_backup, y.idx_keys, v.params, kIndexDim, y.ratio, capacity, 0, false);
            comp = y.comp; keys = y.idx_keys;
            if (y.ratio > 1) {
                copy(s.kv, y.kv_state, 2*kHeadDim*sizeof(float));
                copy(s.score, y.score_state, 2*kHeadDim*sizeof(float));
            }
        }
        if (is_engram_layer(l)) {
            const int cols = (pack.engram_hash().max_ngram-1)*pack.engram_hash().n_heads;
            for (int t = 0; t < m; ++t) {
                const auto* w = v.eng+t*v.eng_bytes+size_t(eng_i)*kEngRows*(256+8);
                ops::engram_dequant(w, w+kEngRows*256, cols, v.eng_vals+t*y.eng_wkv.k, st);
            }
            linear(v.eng_vals, y.eng_wkv, v.eng_kv);
            ops::engram_apply(v.h, v.eng_kv, y.eng_qw, y.eng_kw, kNormEps, m, st);
            ++eng_i;
        }
        kernels::hc_mixes_pre(v.h, m, y.hc_attn_fn, y.hc_attn_scale, y.hc_attn_base, v.mix,
                             v.xa, v.apre, v.apost, v.acomb, st);
        ops::rmsnorm(v.xa, y.attn_norm, v.xa, kDim, kNormEps, m, st);
        linear(v.xa, y.wq_a, v.qr);
        ops::rmsnorm(v.qr, y.q_norm, v.qr, kQLora, kNormEps, m, st);
        linear(v.qr, y.wq_b, v.q);
        linear(v.xa, y.wkv, v.kvv);
        ops::rmsnorm(v.kvv, y.kv_norm, v.kvv, kHeadDim, kNormEps, m, st);
        for (int t = 0; t < m; ++t) {
            const int* p = v.params+4*t;
            const bf16* x = v.xa+t*kDim;
            bf16* qt = v.q+t*kHeads*kHeadDim;
            bf16* kt = v.kvv+t*kHeadDim;
            bf16* ot = v.o+t*kHeads*kHeadDim;
            int32_t* idx = v.indices+t*(kWindow+kIndexTopK);
            uint8_t* cand = v.candidates+size_t(t)*(max_seq+1);
            const float* rope = y.ratio ? rope_yarn : rope_plain;
            ops::rope_device(qt, kHeads, kHeadDim, rope, p+1, 0, false, st);
            ops::rope_device(kt, 1, kHeadDim, rope, p+1, 0, false, st);
            ops::act_quant_inplace(kt, kHeadDim, st);
            copy(s.rows+t*kHeadDim, kt, kHeadDim*sizeof(bf16));
            ops::row_copy_device(s.ring, kt, kHeadDim*sizeof(bf16), p+1, 1, kWindow, st);
            bool have_latent = false;
            if (is_kv_source(l)) {
                if (y.ratio == 1) {
                    ops::bf16_linear(x, nullptr, y.c_wkv, kDim, kHeadDim, latent, nullptr, st);
                    ops::rmsnorm(latent, y.c_norm, latent, kHeadDim, kNormEps, 1, st);
                    have_latent = true;
                } else {
                    const int slot = (parity+t)&1;
                    ops::bf16_linear(x, nullptr, y.c_wkv, kDim, kHeadDim, nullptr, s.kv+slot*kHeadDim, st);
                    ops::bf16_linear(x, nullptr, y.c_wgate, kDim, kHeadDim, nullptr, s.score+slot*kHeadDim, st);
                    if (slot == 1) {
                        ops::compress_pool(s.kv, s.score, 2, latent, 1, st);
                        ops::rmsnorm(latent, y.c_norm, latent, kHeadDim, kNormEps, 1, st);
                        have_latent = true;
                    }
                    copy(s.kv_rows+t*2*kHeadDim, s.kv, 2*kHeadDim*sizeof(float));
                    copy(s.score_rows+t*2*kHeadDim, s.score, 2*kHeadDim*sizeof(float));
                }
            }
            if (is_index_source(l)) {
                if (have_latent) {
                    ops::bf16_linear(latent, nullptr, y.idx_wk, kHeadDim, kIndexDim, ik, nullptr, st);
                    ops::rmsnorm(ik, y.idx_knorm, ik, kIndexDim, kNormEps, 1, st);
                    ops::rope_device(ik, 1, kIndexDim, rope_yarn, p+1, 1-y.ratio, false, st);
                    ops::fp4_quant_inplace(ik, kIndexDim, 32, false, st);
                    ops::row_copy_device(y.idx_keys, ik, kIndexDim*sizeof(bf16), p+1, y.ratio, max_seq+1, st);
                }
                fp8_linear(v.qr+t*kQLora, y.idx_wq_b, iq);
                ops::rope_device(iq, kIndexHeads, kIndexDim, rope_yarn, p+1, 0, false, st);
                ops::fp4_quant_inplace(iq, kIndexHeads*kIndexDim, 32, false, st);
                ops::bf16_linear(x, nullptr, y.idx_wp, kDim, kIndexHeads, iw_raw, nullptr, st);
                ops::scale_bf16(iw_raw, float(std::pow(kIndexDim,-0.5)*std::pow(kIndexHeads,-0.5)), iw, kIndexHeads, st);
                const int64_t cap = y.ratio == 1 ? tcap1 : tcap2;
                kernels::indexer_topk_device(iq, keys, p+1, y.ratio, cap, iw, l > kCandidateLayer ? cand : nullptr,
                                            kIndexTopK, kWindow, scores, idx+kWindow, st);
                if (l == kCandidateLayer)
                    kernels::candidate_blocks_device(scores, p+1, y.ratio, cap, kCandidateBlocks, kCandidateBlock, cand, st);
            }
            if (have_latent) {
                ops::rope_device(latent, 1, kHeadDim, rope_yarn, p+1, 1-y.ratio, false, st);
                ops::fp4_quant_inplace(latent, kHeadDim, 16, true, st);
                // Only append beyond the committed length. Each query uses its own causal length.
                ops::row_copy_device(y.comp, latent, kHeadDim*sizeof(bf16), p+1, y.ratio, max_seq+1, st);
            }
            if (y.ratio)
                kernels::sparse_attn_decode_device(qt, s.ring, comp, idx, p+(y.ratio == 1 ? 2 : 3), y.sink,
                                                   float(std::pow(kHeadDim,-0.5)), ot, st);
            else
                kernels::sparse_attn_decode(qt, s.ring, nullptr, idx, 1, kWindow, y.sink,
                                            float(std::pow(kHeadDim,-0.5)), ot, st);
            ops::rope_device(ot, kHeads, kHeadDim, rope, p+1, 0, true, st);
            if (y.wo_a8.w) wo_a_grouped_fp8(ot, y.wo_a8.w, y.wo_a8.s, v.oa+t*kOGroups*kOLora, st);
            else ops::wo_a_grouped(ot, y.wo_a, v.oa+t*kOGroups*kOLora, st);
        }
        linear(v.oa, y.wo_b, v.attn_out);
        ops::hc_post(v.attn_out, v.h, v.apost, v.acomb, v.h2, m, st);
        kernels::hc_mixes_pre(v.h2, m, y.hc_ffn_fn, y.hc_ffn_scale, y.hc_ffn_base, v.apre,
                             v.xf, v.fpre, v.fpost, v.fcomb, st);
        ops::rmsnorm(v.xf, y.ffn_norm, v.xf, kDim, kNormEps, m, st);
        int32_t* ids = v.routes+l*VerifyWorkspace::M*kTopK;
        float* w = v.weights+l*VerifyWorkspace::M*kTopK;
        kernels::router_topk(v.xf, m, y.gate_w, y.gate_bias, ids, w, st);
        ops::to_half_fp8q(v.xf, v.half, m*kDim, st);
        const bool tier = vram && vram->slots() > 0;
        const auto* ram = host && host->experts_dev() ? host->experts_dev()+size_t(l)*kExperts : nullptr;
        v.db->publish(v.half, ids, w, m, tier ? vram->res_dev()+size_t(l)*kExperts : nullptr, v.selected,
                      l+1, st, tier ? vram->experts_dev() : nullptr, ram, zc_quota ? zc_quota.get()+l : nullptr);
        ck(cudaMemsetAsync(v.routed, 0, m*kDim*sizeof(float), st), "verify routed");
        if (tier || ram)
            kernels::exl3_moe_decode((const __half*)v.half, m, v.selected, w, kTopK, v.db->gpu_experts(), v.routed,
                                    vram ? vram->workspace() : zc_workspace.get(), VramExperts::kWorkspaceBytes, st);
        linear(v.xf, y.sh_w1, v.g); linear(v.xf, y.sh_w3, v.u);
        ops::swiglu(v.g, v.u, kSwigluLimit, v.sh_h, m*kMoeInter, st);
        linear(v.sh_h, y.sh_w2, v.sh_out);
        v.db->wait_add(v.routed, m, l+1, st);
        ops::add_f32_bf16(v.routed, v.sh_out, v.ffn_out, m*kDim, st);
        ops::hc_post(v.ffn_out, v.h2, v.fpost, v.fcomb, v.h, m, st);
        copy(v.mix, v.fpre, m*kHc*sizeof(float));
    }
    ops::hc_pre(v.h, v.mix, v.final_x, m, st);
    ops::rmsnorm(v.final_x, final_norm, v.final_x, kDim, kNormEps, m, st);
    for (int t = 0; t < m; ++t) {
        ops::bf16_linear(v.final_x+t*kDim, nullptr, head, kDim, kVocab, nullptr, v.logits+t*kVocab, st);
        ops::argmax_logits(v.logits+t*kVocab, v.next+t, st);
    }
    ck(cudaMemcpyAsync(v.next_host, v.next, m*sizeof(int), cudaMemcpyDeviceToHost, st), "verify next");
    ck(cudaMemcpyAsync(v.logits_host, v.logits, size_t(m)*kVocab*sizeof(float), cudaMemcpyDeviceToHost, st), "verify logits");
    ck(cudaMemcpyAsync(v.routes_host, v.routes, kLayers*VerifyWorkspace::M*kTopK*sizeof(int32_t), cudaMemcpyDeviceToHost, st), "verify routes");
}

VerifyResult Engine::Impl::verify(const std::vector<int>& window, int pos, bool want_logits, Timing& tm) {
    if (verify_pending) throw std::logic_error("commit the pending verify window first");
    const int m = int(window.size());
    if (m < 1 || m > kVerifyMaxTokens) throw std::invalid_argument("verify supports 1..4 rows on the CPU path");
    if (pos < 0 || pos != int(history.size()) || pos > max_seq-m)
        throw std::invalid_argument("verify position is not the committed position or exceeds max_seq");
    for (int token : window)
        if (token < 0 || token >= kVocab) throw std::invalid_argument("verify token outside vocabulary");
    if (!verify_ws) verify_ws = std::make_shared<VerifyWorkspace>(*this);
    auto& v = *verify_ws;
    v.tokens = window; v.pos = pos; v.m = m;
    tm = Timing{};
    const double begin = now_ms();
    // Stage hashes from each tentative prefix. Restore formal history even if an I/O read fails.
    try {
        for (int t = 0; t < m; ++t) {
            history.push_back(pack.engram_hash().token_map[window[t]]);
            int li = 0;
            for (int l = 0; l < kLayers; ++l)
                if (is_engram_layer(l)) engram_ids(l, li++, pos+t);
            engram_read_all();
            if (n_eng) std::memcpy(v.eng_host+t*v.eng_bytes, eng_host, v.eng_bytes);
            const int params[] = {window[t], pos+t, pos+t+1, (pos+t+1)/2};
            std::copy_n(params, 4, v.params_host+4*t);
        }
    } catch (...) { history.resize(pos); throw; }
    history.resize(pos);
    tm.engram_ms = now_ms()-begin;
    if (vram) tm.vram_swaps = vram->between_steps();
    parity = pos&1;
    tcap1 = cap_of(pos+m, max_seq+1);
    tcap2 = cap_of((pos+m)/2, max_seq/2+1);
    const std::array<int64_t, 4> key{m, parity, tcap1, tcap2};
    // The first window is eager. Capture only after every kernel has run once.
    cudaGraphExec_t executable = nullptr;
    bool reused = false;
    if (use_graph && v.warmed) {
        auto it = v.graphs.find(key);
        reused = it != v.graphs.end();
        if (it == v.graphs.end()) {
            cudaGraph_t graph = nullptr;
            ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed), "verify capture");
            try {
                enqueue_verify(m);
                ck(cudaStreamEndCapture(st, &graph), "verify end capture");
                ck(cudaGraphInstantiate(&executable, graph, 0), "verify instantiate");
            } catch (...) {
                if (!graph) cudaStreamEndCapture(st, &graph);
                if (graph) cudaGraphDestroy(graph);
                throw;
            }
            cudaGraphDestroy(graph);
            it = v.graphs.emplace(key, executable).first;
        }
        executable = it->second;
    }
    v.db->reset();
    std::atomic<bool> cancel{false};
    std::exception_ptr worker_error;
    // The original decode worker is idle. Use a separate doorbell and leave its protocol unchanged.
    std::thread cpu([&] {
        for (int l = 0; l < kLayers; ++l) {
            if (!v.db->wait_published(l+1, cancel)) return;
            const auto counts = v.db->counts();
            tm.expert_hits += counts.vram;
            if (!worker_error) {
                try {
                    c10::Half weights[kVerifyMaxTokens*kTopK];
                    int misses = 0;
                    for (int i = 0; i < m*kTopK; ++i) {
                        weights[i] = c10::Half(__half_as_ushort(__float2half_rn(v.db->w()[i])), c10::Half::from_bits());
                        const int e = v.db->ids()[i];
                        if (e < 0) continue;
                        ++misses;
                        if (host && host->slot_of(l, e) >= 0) ++tm.ram_experts;
                        else {
                            ++tm.file_experts;
                            if (file_pages_missing(l, e)) {
                                ++tm.ssd_experts;
                                if (fetch_now) warm_file_expert(l, e);
                            }
                        }
                    }
                    const double start = now_ms();
                    if (misses)
                        // moe_mul1 groups rows by expert before its weight passes. m never exceeds four.
                        exl3_moe_cpu_forward_raw(L[l].moe_handle, (const at::Half*)v.db->x(), v.db->ids(), weights,
                                                v.db->y(), m, kTopK, cpu_threads);
                    else std::fill_n(v.db->y(), m*kDim, 0.0f);
                    tm.cpu_experts_ms += now_ms()-start;
                } catch (...) { worker_error = std::current_exception(); }
            }
            // Finish the protocol on failure. The caller rejects the entire window after the stream drains.
            if (worker_error) std::fill_n(v.db->y(), m*kDim, 0.0f);
            v.db->mark_done(l+1);
        }
    });
    verify_pending = true;
    verify_failed = true;
    try {
        if (executable) ck(cudaGraphLaunch(executable, st), "verify launch");
        else enqueue_verify(m);
        ck(cudaStreamSynchronize(st), "verify completion");
    } catch (...) {
        cancel = true;
        cpu.join();
        // Release any enqueued wait before workspace destruction. Never commit this failed window.
        v.db->mark_done(kLayers);
        cudaStreamSynchronize(st);
        throw;
    }
    cpu.join();
    if (worker_error) std::rethrow_exception(worker_error);
    v.warmed = true;
    VerifyResult result;
    result.graph_reused = reused;
    result.next.assign(v.next_host, v.next_host+m);
    if (want_logits) {
        result.logits.resize(m);
        for (int t = 0; t < m; ++t)
            result.logits[t].assign(v.logits_host+t*kVocab, v.logits_host+(t+1)*kVocab);
    }
    tm.expert_total = m*kLayers*kTopK;
    tm.total_ms = now_ms()-begin;
    tm.gpu_ms = tm.total_ms-tm.engram_ms;
    verify_failed = false;
    return result;
}

void Engine::Impl::commit_verify(int keep) {
    if (!verify_pending || verify_failed) throw std::logic_error("no successful verify window to commit");
    auto& v = *verify_ws;
    if (keep < 1 || keep > v.m) throw std::invalid_argument("commit count outside verify window");
    verify_failed = true; // A partial CUDA failure must prevent reuse of this Engine.
    // All positions come from the unchanged window parameter buffer. These copies are capturable.
    for (int l = 0; l < kLayers; ++l) {
        auto& y = L[l]; auto& s = v.state[l];
        for (int t = 0; t < keep; ++t)
            ops::row_copy_device(y.window, s.rows+t*kHeadDim, kHeadDim*sizeof(bf16), v.params+t*4+1, 1, kWindow, st);
        if (!is_kv_source(l)) continue;
        if (y.ratio > 1) {
            ck(cudaMemcpyAsync(y.kv_state, s.kv_rows+(keep-1)*2*kHeadDim, 2*kHeadDim*sizeof(float),
                               cudaMemcpyDeviceToDevice, st), "commit compressor KV");
            ck(cudaMemcpyAsync(y.score_state, s.score_rows+(keep-1)*2*kHeadDim, 2*kHeadDim*sizeof(float),
                               cudaMemcpyDeviceToDevice, st), "commit compressor scores");
        }
        // Restore the original bytes of every rejected append. Accepted appends are now visible.
        const int capacity = max_seq/y.ratio+1;
        verify_tail_copy<<<(VerifyWorkspace::M*kHeadDim+255)/256,256,0,st>>>(
            s.comp_backup, y.comp, v.params, kHeadDim, y.ratio, capacity, keep, true);
        verify_tail_copy<<<(VerifyWorkspace::M*kIndexDim+255)/256,256,0,st>>>(
            s.key_backup, y.idx_keys, v.params, kIndexDim, y.ratio, capacity, keep, true);
    }
    ck(cudaStreamSynchronize(st), "verify commit");
    for (int t = 0; t < keep; ++t) history.push_back(pack.engram_hash().token_map[v.tokens[t]]);
    lg.assign(v.logits_host+(keep-1)*kVocab, v.logits_host+keep*kVocab);
    if (vram) {
        int32_t row[kLayers*kTopK];
        for (int t = 0; t < keep; ++t) {
            for (int l = 0; l < kLayers; ++l)
                std::copy_n(v.routes_host+(l*VerifyWorkspace::M+t)*kTopK, kTopK, row+l*kTopK);
            vram->count(row, kTopK);
        }
    }
    // Step reconstructs its index list and candidate mask. No tentative row uses its scratch.
    verify_pending = false;
    verify_failed = false;
}

VerifyResult Engine::verify(const std::vector<int>& window, int pos, bool logits) {
    return impl_->verify(window, pos, logits, timing_);
}
void Engine::commit(int n_keep) { impl_->commit_verify(n_keep); }
