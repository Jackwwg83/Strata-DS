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
    // 4. upstream's route tail skip (tail_rank 4: ranks 4 and 5 of 0..5). Experts with id % 3 == 0 are resident.
    //    token 0: 1 2 4 5 7 8      token 1: 11 13 14 16 5 18
    //    7 (rank 4) and 8 (rank 5) are missed and every token routes them low: skipped. 5 is rank 4 in token 1 but rank 3
    //    in token 0: kept for both. 18 (rank 5) is resident: a VRAM hit, never skipped.
    {
        const int m = 2;
        ExpertDoorbell db(m, kTopK, kDim);
        std::vector<int32_t> ids = {1, 2, 4, 5, 7, 8, 11, 13, 14, 16, 5, 18};
        std::vector<int32_t> res(kExperts, -1);
        for (int i = 0; i < kExperts; i += 3) res[i] = 100 + i;
        Dev<uint16_t> dx(std::vector<uint16_t>(m * kDim, 0));
        Dev<int32_t> did(ids), dres(res), sel(m * kTopK);
        Dev<float> dw(std::vector<float>(m * kTopK, 0.5f)), dout(std::vector<float>(m * kDim, 0.0f));
        cudaStream_t s;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        std::vector<int32_t> mids;
        ExpertDoorbell::Counts counts{};
        std::atomic<bool> stop{false};
        std::thread th([&] {
            if (!db.wait_published(1, stop)) return;
            mids.assign(db.ids(), db.ids() + m * kTopK);
            counts = db.counts();
            std::fill_n(db.y(), m * kDim, 0.0f);
            db.mark_done(1);
        });
        db.publish(dx.p, did.p, dw.p, m, dres.p, sel.p, 1, s, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                   nullptr, 0, 4);
        db.wait_add(dout.p, m, 1, s);
        ck(cudaStreamSynchronize(s), "tail skip");
        th.join();
        const std::vector<int32_t> want = {1, 2, 4, 5, -1, -1, 11, 13, 14, 16, 5, -1};
        v.check(mids == want, "tail skip: the low-ranked misses go to neither the CPU nor the GPU; a shared expert ranked "
                              "higher by another token and a resident one stay");
        v.check(counts.skipped == 2 && counts.vram == 1 && counts.cpu == 9,
                "tail skip counts: 2 skipped, 1 VRAM hit, 9 CPU uses");
        cudaStreamDestroy(s);
    }
    // 5. the SSD tail skip (ssd_rank 3, no VRAM tier): experts with an even id are in the RAM tier.
    //    token 0: 1 2 4 5 7 8      token 1: 11 13 14 16 5 19
    //    5 (best rank 3, odd), 7 (rank 4, odd) and 19 (rank 5, odd) come from the SSD and are routed low: skipped, 5 in
    //    both tokens. 8 (rank 5) is in RAM: kept. 1, 11, 13 are odd but ranked 0..2: kept.
    {
        const int m = 2;
        ExpertDoorbell db(m, kTopK, kDim);
        std::vector<int32_t> ids = {1, 2, 4, 5, 7, 8, 11, 13, 14, 16, 5, 19};
        std::vector<strata::ds41::kernels::Exl3Expert> ram(kExperts);
        for (int i = 0; i < kExperts; i += 2) ram[i].w1.trellis = reinterpret_cast<const uint16_t*>(uintptr_t(4096));
        Dev<strata::ds41::kernels::Exl3Expert> dram(ram);
        Dev<uint16_t> dx(std::vector<uint16_t>(m * kDim, 0));
        Dev<int32_t> did(ids), sel(m * kTopK);
        Dev<float> dw(std::vector<float>(m * kTopK, 0.5f)), dout(std::vector<float>(m * kDim, 0.0f));
        cudaStream_t s;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        std::vector<int32_t> mids;
        ExpertDoorbell::Counts counts{};
        std::atomic<bool> stop{false};
        std::thread th([&] {
            if (!db.wait_published(1, stop)) return;
            mids.assign(db.ids(), db.ids() + m * kTopK);
            counts = db.counts();
            std::fill_n(db.y(), m * kDim, 0.0f);
            db.mark_done(1);
        });
        db.publish(dx.p, did.p, dw.p, m, nullptr, sel.p, 1, s, nullptr, dram.p, nullptr, nullptr, nullptr, nullptr,
                   nullptr, 0, 0, 3);
        db.wait_add(dout.p, m, 1, s);
        ck(cudaStreamSynchronize(s), "ssd tail skip");
        th.join();
        const std::vector<int32_t> want = {1, 2, 4, -1, -1, 8, 11, 13, 14, 16, -1, -1};
        v.check(mids == want, "ssd tail skip: low-ranked experts outside RAM are left out; those in RAM or ranked high stay");
        v.check(counts.skipped == 4 && counts.cpu == 8, "ssd tail skip counts: 4 skipped, 8 CPU uses");
        cudaStreamDestroy(s);
    }
    // 5b. the late list (unified memory): the CPU hands uses 1 and 4 of token 0 and use 2 of token 1 back to the GPU
    //     with their descriptors; wait_add(late) copies the list to the device, and without late it does not
    {
        const int m = 2;
        ExpertDoorbell db(m, kTopK, kDim);
        std::vector<int32_t> ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        Dev<uint16_t> dx(std::vector<uint16_t>(m * kDim, 0));
        Dev<int32_t> did(ids), sel(m * kTopK);
        Dev<float> dw(std::vector<float>(m * kTopK, 0.5f)), dout(std::vector<float>(m * kDim, 0.0f));
        cudaStream_t s;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        std::atomic<bool> stop{false};
        auto desc_of = [](int i) {
            strata::ds41::kernels::Exl3Expert d{};
            d.w1.trellis = reinterpret_cast<const uint16_t*>(uintptr_t(4096 * (i + 1)));
            d.w2.k = 1000 + i;
            return d;
        };
        for (int round = 1; round <= 2; ++round) {
            std::thread th([&] {
                if (!db.wait_published(round, stop)) return;
                for (int i = 0; i < m * kTopK; ++i) {
                    const bool late = i == 1 || i == 4 || i == 8;
                    db.late_sel()[i] = late && round == 1 ? i : -1;
                    db.late_desc()[i] = desc_of(i);
                }
                std::fill_n(db.y(), m * kDim, 0.0f);
                db.mark_done(round);
            });
            db.publish(dx.p, did.p, dw.p, m, nullptr, sel.p, round, s);
            db.wait_add(dout.p, m, round, s, round == 1);
            ck(cudaStreamSynchronize(s), "late list");
            th.join();
        }
        std::vector<int32_t> got(m * kTopK);
        std::vector<strata::ds41::kernels::Exl3Expert> gd(m * kTopK);
        ck(cudaMemcpy(got.data(), db.late_sel_dev(), got.size() * 4, cudaMemcpyDeviceToHost), "late down");
        ck(cudaMemcpy(gd.data(), db.late_desc_dev(), gd.size() * sizeof(gd[0]), cudaMemcpyDeviceToHost), "late down");
        const std::vector<int32_t> want = {-1, 1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1};
        v.check(got == want, "late list: round 1's list reaches the device; round 2 (late off) leaves it alone");
        v.check(gd[4].w1.trellis == desc_of(4).w1.trellis && gd[8].w2.k == 1008, "late list: the descriptors arrive whole");
        cudaStreamDestroy(s);
    }
    // 6. the default skip ranks: tail 4 and SSD 3 with a VRAM tier under 90% of the experts; both off at 90% or
    //    without a VRAM tier; the env values win, and DS41_ROUTE_TAIL_SKIP=0 also turns the SSD default off.
    {
        using strata::ds41::skip_ranks;
        auto is = [](strata::ds41::SkipRanks r, int t, int s) { return r.tail == t && r.ssd == s; };
        v.check(is(skip_ranks(nullptr, nullptr, true, 0.11), 4, 3), "skip ranks: defaults tail 4, SSD 3");
        v.check(is(skip_ranks(nullptr, nullptr, true, 0.90), 0, 0), "skip ranks: off at 90% cached");
        v.check(is(skip_ranks(nullptr, nullptr, false, 0.0), 0, 0), "skip ranks: off without a VRAM tier");
        v.check(is(skip_ranks("0", nullptr, true, 0.11), 0, 0), "skip ranks: tail 0 also turns the SSD default off");
        v.check(is(skip_ranks(nullptr, "0", true, 0.11), 4, 0), "skip ranks: SSD 0 keeps the tail skip");
        v.check(is(skip_ranks("5", "2", false, 0.0), 5, 2), "skip ranks: env values win");
        v.check(is(skip_ranks("6", "-1", true, 0.11), 0, 0), "skip ranks: out of range is off");
    }
    return v.finish();
}
