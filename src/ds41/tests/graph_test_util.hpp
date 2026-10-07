#pragma once
#include "bench_util.hpp"

namespace ds41graph {
using namespace ds41test;

struct Stream {
    cudaStream_t s;
    Stream() { ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "create graph stream"); }
    ~Stream() { cudaStreamDestroy(s); }
    void sync() { ck(cudaStreamSynchronize(s), "sync graph stream"); }
};
struct Graph {
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    template<class F> Graph(cudaStream_t s, F f) {
        ck(cudaDeviceSynchronize(), "publish constructor uploads");
        ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal), "begin capture");
        f();
        ck(cudaStreamEndCapture(s, &graph), "end capture");
        ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate graph");
    }
    ~Graph() { cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); }
    void run(cudaStream_t s) { ck(cudaGraphLaunch(exec, s), "replay graph"); }
};
template<class T> void upload(Dev<T>& d, const std::vector<T>& values) {
    d.up(values);
    // Dev uses pageable H2D copies. Finish them before a nonblocking stream reads them.
    ck(cudaDeviceSynchronize(), "publish test inputs");
}
template<class T> void same(Verdict& v, const Dev<T>& a, const Dev<T>& b, const std::string& label) {
    auto x = a.down(), y = b.down();
    v.check(x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(T)) == 0, label);
}
template<class T> void fill(Dev<T>& d, cudaStream_t s) {
    ck(cudaMemsetAsync(d.p, 0xa5, d.n * sizeof(T), s), "poison output");
}
}  // namespace ds41graph
