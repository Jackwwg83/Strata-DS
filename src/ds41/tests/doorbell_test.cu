// src/ds41/tests/doorbell_test.cu - ExpertDoorbell: GPU publish -> CPU thread -> GPU wait_add, eagerly and as a
// captured CUDA graph replayed twice. The CPU side computes a known function of what it read, so a lost row, a
// stale read or a wrong mask shows up as a wrong sum.
#include "strata/ds41/doorbell.hpp"

#include "bench_util.hpp"

#include <cuda_fp16.h>

#include <thread>

using namespace ds41test;
using strata::ds41::ExpertDoorbell;

namespace {

constexpr int kDim = 5120, kTopK = 6, kExperts = 64, kRounds = 5;

/// what the CPU thread returns per token: y[d] = sum over its (non -1) slots of w * (x[d] + id)
void cpu_rows(const uint16_t* x, const int32_t* ids, const float* w, int m, float* y) {
    for (int t = 0; t < m; ++t)
        for (int d = 0; d < kDim; ++d) {
            const float xv = __half2float(__ushort_as_half(x[t * kDim + d]));
            float acc = 0;
            for (int j = 0; j < kTopK; ++j) {
                const int32_t id = ids[t * kTopK + j];
                if (id >= 0) acc += w[t * kTopK + j] * (xv + (float) id);
            }
            y[t * kDim + d] = acc;
        }
}

struct Case {
    int m;
    std::vector<std::vector<uint16_t>> x;   // per round
    std::vector<std::vector<int32_t>> ids;
    std::vector<std::vector<float>> w;
    std::vector<int32_t> res;               // [kExperts]: slot or -1
};

Case make_case(int m, uint32_t seed) {
    Case c;
    c.m = m;
    std::mt19937 g(seed);
    std::uniform_int_distribution<int> e(0, kExperts - 1);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    c.res.assign(kExperts, -1);
    for (int i = 0; i < kExperts; i += 3) c.res[i] = 100 + i;   // every third expert "resident"
    for (int r = 0; r < kRounds; ++r) {
        std::vector<uint16_t> x(m * kDim);
        for (auto& v : x) v = __half_as_ushort(__float2half(u(g)));
        std::vector<int32_t> ids(m * kTopK);
        for (auto& v : ids) v = e(g);
        std::vector<float> w(m * kTopK);
        for (auto& v : w) v = 0.5f + 0.5f * u(g);
        c.x.push_back(x);
        c.ids.push_back(ids);
        c.w.push_back(w);
    }
    return c;
}

/// expected out per round: the CPU rows for the non-resident slots
std::vector<float> expected(const Case& c, int r) {
    std::vector<int32_t> ids = c.ids[r];
    for (auto& v : ids) if (c.res[v] >= 0) v = -1;
    std::vector<float> y(c.m * kDim);
    cpu_rows(c.x[r].data(), ids.data(), c.w[r].data(), c.m, y.data());
    return y;
}

/// the CPU thread: kRounds rounds, rounds numbered 1.., computing cpu_rows on what was published
void responder(ExpertDoorbell& db, int m, const std::atomic<bool>& stop, bool& ok) {
    ok = true;
    for (uint32_t r = 1; r <= kRounds; ++r) {
        if (!db.wait_published(r, stop)) { ok = false; return; }
        cpu_rows(db.x(), db.ids(), db.w(), m, db.y());
        db.mark_done(r);
    }
}

}  // namespace

int main() {
    require_gpu();
    Verdict v;
    for (int m : {1, 4, 8}) {
        Case c = make_case(m, 1234 + m);
        ExpertDoorbell db(8, kTopK, kDim);
        std::vector<Dev<uint16_t>*> dx;
        std::vector<Dev<int32_t>*> dids;
        std::vector<Dev<float>*> dw, dout;
        for (int r = 0; r < kRounds; ++r) {
            dx.push_back(new Dev<uint16_t>(c.x[r]));
            dids.push_back(new Dev<int32_t>(c.ids[r]));
            dw.push_back(new Dev<float>(c.w[r]));
            dout.push_back(new Dev<float>(std::vector<float>(m * kDim, 0.0f)));
        }
        Dev<int32_t> dres(c.res);
        Dev<int32_t> gpu_sel((size_t) kRounds * m * kTopK);
        cudaStream_t s;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        auto enqueue = [&]() {
            for (int r = 0; r < kRounds; ++r) {
                ck(cudaMemsetAsync(dout[r]->p, 0, (size_t) m * kDim * 4, s), "memset");
                db.publish(dx[r]->p, dids[r]->p, dw[r]->p, m, dres.p, gpu_sel.p + r * m * kTopK, r + 1, s);
                db.wait_add(dout[r]->p, m, r + 1, s);
            }
        };
        auto check = [&](const char* mode) {
            for (int r = 0; r < kRounds; ++r) {
                const double err = rel_l2(dout[r]->down(), expected(c, r));
                v.check(err < 1e-6, std::string(mode) + " m=" + std::to_string(m) + " round " + std::to_string(r) +
                                        " rel_l2=" + std::to_string(err));
            }
            const auto sel = gpu_sel.down();
            for (int r = 0; r < kRounds; ++r)
                for (int i = 0; i < m * kTopK; ++i)
                    v.check(sel[r * m * kTopK + i] == c.res[c.ids[r][i]], std::string(mode) + " gpu_sel");
        };
        auto run = [&](auto launch, const char* mode) {
            db.reset();
            std::atomic<bool> stop{false};
            bool ok = false;
            std::thread th(responder, std::ref(db), m, std::cref(stop), std::ref(ok));
            launch();
            const cudaError_t e = cudaStreamSynchronize(s);
            stop = true;
            th.join();
            ck(e, mode);
            v.check(ok, std::string(mode) + " responder finished");
            check(mode);
        };
        // 1. eager
        run([&] { enqueue(); }, "eager");
        // 2. one captured graph, replayed twice with a reset in between
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "begin capture");
        enqueue();
        ck(cudaStreamEndCapture(s, &graph), "end capture");
        ck(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
        for (int rep = 0; rep < 2; ++rep) run([&] { ck(cudaGraphLaunch(exec, s), "graph launch"); }, "graph");
        // 3. handoff latency: one round trip with an empty CPU side, median of 200
        {
            std::vector<double> us;
            for (int i = 0; i < 200; ++i) {
                db.reset();
                std::atomic<bool> stop{false};
                std::thread th([&] { if (db.wait_published(1, stop)) db.mark_done(1); });
                cudaEvent_t a, b;
                cudaEventCreate(&a);
                cudaEventCreate(&b);
                cudaEventRecord(a, s);
                db.publish(dx[0]->p, dids[0]->p, dw[0]->p, m, dres.p, gpu_sel.p, 1, s);
                db.wait_add(dout[0]->p, m, 1, s);
                cudaEventRecord(b, s);
                ck(cudaEventSynchronize(b), "latency");
                th.join();
                float ms = 0;
                cudaEventElapsedTime(&ms, a, b);
                us.push_back(ms * 1000.0);
                cudaEventDestroy(a);
                cudaEventDestroy(b);
            }
            std::sort(us.begin(), us.end());
            std::printf("m=%d round trip median %.1f us\n", m, us[us.size() / 2]);
            v.metric("roundtrip_us_m" + std::to_string(m), us[us.size() / 2]);
        }
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(s);
        for (int r = 0; r < kRounds; ++r) { delete dx[r]; delete dids[r]; delete dw[r]; delete dout[r]; }
    }
    return v.finish();
}
