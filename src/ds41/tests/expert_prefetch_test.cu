// src/ds41/tests/expert_prefetch_test.cu - ExpertPrefetch: the guesses are the router's best experts in order (as a
// host reference computes them), the plan skips VRAM residents, experts outside the RAM tier and what does not fit the
// buffer, and the copied bytes and rebased descriptors point at the expert in the buffer. Then the doorbell's publish:
// a routed miss that was prefetched goes to the GPU with the buffer's descriptor, outside the zero-copy quota.
#include "strata/ds41/doorbell.hpp"
#include "strata/ds41/expert_prefetch.hpp"
#include "bench_util.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

using namespace ds41test;
namespace sd = strata::ds41;

int main() {
    Verdict v;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("RESULT skip (no GPU)\n");
        return 77;
    }
    constexpr int N = 384, D = 5120, G = 9;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<__nv_bfloat16> x(D), w((size_t) N * D);
    std::vector<float> bias(N);
    for (auto& t : x) t = __float2bfloat16(nd(rng));
    for (auto& t : w) t = __float2bfloat16(nd(rng) * 0.02f);
    for (auto& t : bias) t = nd(rng) * 0.05f;
    // host reference: logits in double, score sqrt(softplus) + bias, the G best (ties: lower id)
    std::vector<double> score(N);
    for (int e = 0; e < N; ++e) {
        double l = 0;
        for (int d = 0; d < D; ++d) l += (double) __bfloat162float(x[d]) * __bfloat162float(w[(size_t) e * D + d]);
        const double sp = l > 20 ? l : std::log1p(std::exp(l));
        score[e] = std::sqrt(sp) + bias[e];
    }
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return score[a] > score[b]; });

    // the tiers: rank 0 and 3 in VRAM, rank 1 outside the RAM tier, the rest in RAM with sizes 64 KiB .. 576 KiB
    std::vector<int32_t> res(N, -1);
    res[order[0]] = 5;
    res[order[3]] = 9;
    const size_t host_bytes = (size_t) N * (1u << 20);
    uint8_t *host = nullptr, *host_dev = nullptr;
    cudaHostAlloc((void**) &host, host_bytes, cudaHostAllocMapped);
    cudaHostGetDevicePointer((void**) &host_dev, host, 0);
    for (size_t i = 0; i < host_bytes; ++i) host[i] = (uint8_t) (i * 2654435761u >> 13);
    std::vector<sd::kernels::Exl3Expert> ram(N);
    std::vector<sd::ExpertBlob> blobs(N);
    for (int e = 0; e < N; ++e) {
        const size_t bytes = (64u << 10) * (1 + e % 9), first = 512;
        uint8_t* base = host_dev + (size_t) e * (1u << 20);
        blobs[e] = {bytes, first};
        if (e == order[1]) continue;   // not in RAM: null descriptor
        ram[e].w1 = {(const uint16_t*) (base + first), (const __half*) (base + 1024), (const __half*) (base + 2048), 1, 1, 1};
        ram[e].w3 = {(const uint16_t*) (base + 4096), (const __half*) (base + 8192), (const __half*) (base + 8704), 1, 1, 1};
        ram[e].w2 = {(const uint16_t*) (base + 16384), (const __half*) (base + 20480), (const __half*) (base + 24576), 1, 1, 1};
    }
    // a buffer that holds the first few eligible guesses but not all of them
    std::vector<int> eligible;
    for (int r = 0; r < G; ++r)
        if (r != 0 && r != 1 && r != 3) eligible.push_back(order[r]);
    size_t cap = 0;
    for (size_t i = 0; i + 1 < eligible.size(); ++i) cap = (cap + 255) / 256 * 256 + blobs[eligible[i]].bytes;
    std::vector<int> want(eligible.begin(), eligible.end() - 1);   // the last eligible guess does not fit
    sd::ExpertPrefetch pf(G, cap, N, D);

    __nv_bfloat16 *dx, *dw;
    float* db;
    int32_t* dres;
    sd::kernels::Exl3Expert* dram;
    sd::ExpertBlob* dblobs;
    cudaMalloc(&dx, D * 2);
    cudaMalloc(&dw, (size_t) N * D * 2);
    cudaMalloc(&db, N * 4);
    cudaMalloc(&dres, N * 4);
    cudaMalloc(&dram, N * sizeof(sd::kernels::Exl3Expert));
    cudaMalloc(&dblobs, N * sizeof(sd::ExpertBlob));
    cudaMemcpy(dx, x.data(), D * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), (size_t) N * D * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(db, bias.data(), N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dres, res.data(), N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dram, ram.data(), N * sizeof(sd::kernels::Exl3Expert), cudaMemcpyHostToDevice);
    cudaMemcpy(dblobs, blobs.data(), N * sizeof(sd::ExpertBlob), cudaMemcpyHostToDevice);
    cudaStream_t st;
    cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    // layers 7 and 8 eagerly (both buffers), then layer 9 captured as a graph and replayed (the engine's decode step
    // is a graph): plan() forks the guesses and the copy to the prefetch stream; ready() and join() bring them back
    cudaGraphExec_t replay = nullptr;
    for (int layer : {7, 8, 9}) {
        if (layer == 9) {
            cudaGraph_t g;
            v.check(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal) == cudaSuccess, "capture begins");
            pf.plan(layer, dx, dw, db, dres, dram, dblobs, st);
            pf.ready(layer, st);
            pf.join(layer, st);
            v.check(cudaStreamEndCapture(st, &g) == cudaSuccess && cudaGraphInstantiate(&replay, g, 0) == cudaSuccess,
                    "plan, ready and join capture into one graph (the prefetch stream joins main again)");
            cudaMemsetAsync((void*) pf.ids(layer), 0x55, G * sizeof(int32_t), st);   // stale ids must be replaced
            v.check(cudaGraphLaunch(replay, st) == cudaSuccess, "the graph replays");
        } else {
            pf.plan(layer, dx, dw, db, dres, dram, dblobs, st);
            pf.ready(layer, st);
            pf.join(layer, st);
        }
        v.check(cudaStreamSynchronize(st) == cudaSuccess, "the prefetch ran");
        int32_t ranked[G], ids[G];
        sd::kernels::Exl3Expert descs[G];
        cudaMemcpy(ranked, pf.ranked(layer), sizeof ranked, cudaMemcpyDeviceToHost);
        cudaMemcpy(ids, pf.ids(layer), sizeof ids, cudaMemcpyDeviceToHost);
        cudaMemcpy(descs, pf.descs(layer), sizeof descs, cudaMemcpyDeviceToHost);
        v.check(std::equal(ranked, ranked + G, order.begin()), "layer " + std::to_string(layer) +
                                                                    ": the guesses are the router's best, in order");
        bool ids_ok = true;
        for (int k = 0; k < G; ++k) ids_ok &= ids[k] == (k < (int) want.size() ? want[k] : -1);
        v.check(ids_ok, "layer " + std::to_string(layer) +
                            ": VRAM residents, experts outside RAM and what does not fit are not copied");
        bool bytes_ok = true;
        for (size_t k = 0; k < want.size(); ++k) {
            const int e = want[k];
            const size_t bytes = blobs[e].bytes;
            std::vector<uint8_t> got(bytes);
            const uint8_t* dst = (const uint8_t*) descs[k].w1.trellis - blobs[e].first_trellis;
            bytes_ok &= cudaMemcpy(got.data(), dst, bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
                        std::memcmp(got.data(), host + (size_t) e * (1u << 20), bytes) == 0;
            bytes_ok &= (const uint8_t*) descs[k].w2.svh - (const uint8_t*) descs[k].w1.trellis == 24576 - 512;
        }
        v.check(bytes_ok, "layer " + std::to_string(layer) + ": the buffer holds each copied expert, the descriptors "
                                                              "point at it");
    }
    // publish: routes = ranks 0 (VRAM), 2 (prefetched), 1 (RAM descriptor missing: CPU), and three RAM experts that
    // were not guessed; quota 1: one of those three is zero-copy, two go to the CPU
    {
        const int layer = 8;
        int32_t ids_h[G];
        sd::kernels::Exl3Expert descs_h[G];
        cudaMemcpy(ids_h, pf.ids(layer), sizeof ids_h, cudaMemcpyDeviceToHost);
        cudaMemcpy(descs_h, pf.descs(layer), sizeof descs_h, cudaMemcpyDeviceToHost);
        std::vector<int32_t> others;
        for (int e = 0; e < N && others.size() < 3; ++e)
            if (std::find(order.begin(), order.begin() + G, e) == order.begin() + G) others.push_back(e);
        const int32_t routes[6] = {order[0], order[2], order[1], others[0], others[1], others[2]};
        const float wts[6] = {1, 1, 1, 1, 1, 1};
        std::vector<sd::kernels::Exl3Expert> vram_desc(16);
        vram_desc[5].w1.k = 55;   // a marker: the descriptor of VRAM slot 5
        int32_t *droutes, *dsel;
        float* dwts;
        uint16_t* dxh;
        int* dquota;
        sd::kernels::Exl3Expert* dvram;
        cudaMalloc(&droutes, sizeof routes);
        cudaMalloc(&dwts, sizeof wts);
        cudaMalloc(&dsel, 6 * 4);
        cudaMalloc(&dxh, D * 2);
        cudaMalloc(&dquota, 4);
        cudaMalloc(&dvram, 16 * sizeof(sd::kernels::Exl3Expert));
        const int quota = 1;
        cudaMemcpy(droutes, routes, sizeof routes, cudaMemcpyHostToDevice);
        cudaMemcpy(dwts, wts, sizeof wts, cudaMemcpyHostToDevice);
        cudaMemcpy(dquota, &quota, 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dvram, vram_desc.data(), 16 * sizeof(sd::kernels::Exl3Expert), cudaMemcpyHostToDevice);
        cudaMemset(dxh, 0, D * 2);
        sd::ExpertDoorbell db(1, 6, D);
        db.publish(dxh, droutes, dwts, 1, dres, dsel, 1, st, dvram, dram, dquota, nullptr, nullptr, pf.ids(layer),
                   pf.descs(layer), G);
        v.check(cudaStreamSynchronize(st) == cudaSuccess, "publish ran");
        int32_t sel[6];
        sd::kernels::Exl3Expert call[6];
        cudaMemcpy(sel, dsel, sizeof sel, cudaMemcpyDeviceToHost);
        cudaMemcpy(call, db.gpu_experts(), sizeof call, cudaMemcpyDeviceToHost);
        const auto c = db.counts();
        std::printf("  counts: vram %d prefetched %d zero_copy %d cpu %d\n", c.vram, c.prefetched, c.zero_copy, c.cpu);
        v.check(c.vram == 1 && c.prefetched == 1 && c.zero_copy == 1 && c.cpu == 3,
                "one VRAM hit, one prefetched, one zero-copy (the quota), three for the CPU");
        v.check(sel[0] == 0 && call[0].w1.k == 55, "the VRAM hit uses its slot's descriptor");
        v.check(sel[1] == 1 && call[1].w1.trellis == descs_h[0].w1.trellis && ids_h[0] == order[2],
                "the prefetched miss uses the buffer's descriptor");
        v.check(sel[2] == -1 && db.ids()[2] == order[1], "a miss without a RAM descriptor goes to the CPU");
        v.check(sel[3] == 3 && sel[4] == -1 && sel[5] == -1, "the quota still caps the zero-copy misses");
        for (void* p : {(void*) droutes, (void*) dwts, (void*) dsel, (void*) dxh, (void*) dquota, (void*) dvram})
            cudaFree(p);
    }
    cudaStreamDestroy(st);
    return v.finish();
}
