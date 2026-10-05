// Strata-DS diagnostic-only instrumentation. Included in a generated copy of
// moe_mul1.cpp, never in the production build. Linux, C++17, no new libraries.
#pragma once
#include <cerrno>
#include <cstdint>
#include <ctime>
#include <unistd.h>

namespace k11diag {
constexpr int capacity = 512;
inline double wall() {
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline double cpu() {
    timespec t{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t)) { perror("clock_gettime"); std::abort(); }
    return t.tv_sec * 1e6 + t.tv_nsec * 1e-3;
}
struct alignas(64) Worker {
    double start = 0, end = 0, cpu_us = 0;
    int cpu_start = -1, cpu_end = -1, core_type = -1;
    uint64_t calls = 0, bands = 0, row_bands = 0, weight_tiles = 0, weight_bytes = 0;
};
struct alignas(64) Pin { int target = -1, error = 0; unsigned calls = 0; };
static Worker workers[6][capacity];
static Pin pins[capacity];
static thread_local Worker* current = nullptr;
static double phases[6], phase_starts[6];
static int seen_m[1024]{};
static uint64_t serial = 0;
static FILE* file = nullptr;
static bool initialized = false;
static cpu_set_t incoming;
inline int core_type() {
    // CPUID hybrid leaf is executed on each worker. 0x40 = Intel Core (P),
    // 0x20 = Intel Atom (E), -1 = unavailable. Do not infer type from CPU IDs.
#if defined(__GNUC__) && __GNUC__ >= 11
    unsigned a, b, c, d;
    if (__get_cpuid_max(0, nullptr) >= 0x1a && __get_cpuid(0x1a, &a, &b, &c, &d))
        return (a >> 24) & 255;
#endif
    return -1;
}
inline void pin(int worker, int target, int error) {
    if (worker < 0 || worker >= capacity) { std::fprintf(stderr,"K11 diagnostic supports at most 512 threads\n"); std::abort(); }
    pins[worker].target = target; pins[worker].error = error; ++pins[worker].calls;
}
inline void initialize() {
    if (initialized) return;
    initialized = true;
    CPU_ZERO(&incoming);
    if (sched_getaffinity(0, sizeof incoming, &incoming)) { perror("sched_getaffinity"); std::abort(); }
    const char* path = std::getenv("K11_DIAG_JSONL");
    if (!path || !*path || !(file = std::fopen(path, "w"))) {
        std::fprintf(stderr, "Set K11_DIAG_JSONL to a writable output file (errno=%d)\n", errno); std::abort();
    }
    std::fprintf(file,"{\"kind\":\"incoming_affinity\",\"cpus\":[");
    bool comma = false;
    for (int c=0; c<CPU_SETSIZE; ++c) if(CPU_ISSET(c,&incoming)) {
        std::fprintf(file,"%s%d",comma?",":"",c); comma=true;
    }
    std::fprintf(file,"]}\n");
}
inline void begin(int requested) {
    initialize();
    if(requested > capacity) { std::fprintf(stderr,"K11 diagnostic supports at most 512 threads\n"); std::abort(); }
    for (auto& phase : workers) for (auto& v : phase) v = Worker{};
    std::fill(phases, phases+6, 0.0);
}
struct Scope {
    Worker& w; double started_cpu;
    Scope(int phase, int worker):w(workers[phase][worker]) {
        current=&w; w.cpu_start=sched_getcpu(); w.core_type=core_type();
        w.start=wall(); started_cpu=cpu();
    }
    ~Scope() { w.cpu_us=cpu()-started_cpu; w.end=wall(); w.cpu_end=sched_getcpu(); current=nullptr; }
};
inline void work(int k, int bits, int hb, int rows, int t0, int t1) {
    if(!current) return;
    const uint64_t bands=(t1-t0)/8;
    const uint64_t tiles=uint64_t(t1-t0)*(k/16)*((rows+3)/4);
    ++current->calls; current->bands+=bands; current->row_bands+=bands*rows;
    current->weight_tiles+=tiles;
    current->weight_bytes+=tiles*(bits*32+hb*16);
}
inline void finish(int m, int requested, int active, int nc, int isa, int nphases,
                   const std::vector<int>& order, const float* output, size_t count) {
    std::fprintf(file,"{\"kind\":\"forward\",\"serial\":%llu,\"m\":%d,\"requested\":%d,\"active\":%d,\"chunks\":%d,\"isa\":%d,\"sample_for_m\":%d,\"core_order\":[",
        (unsigned long long)serial++,m,requested,active,nc,isa,m<1024?seen_m[m]:-1);
    for(size_t i=0;i<order.size();++i) std::fprintf(file,"%s%d",i?",":"",order[i]);
    std::fprintf(file,"],\"phases\":[");
    for(int p=0;p<nphases;++p) {
        std::fprintf(file,"%s{\"phase\":%d,\"wall_us\":%.3f,\"workers\":[",p?",":"",p,phases[p]);
        for(int w=0;w<active;++w) {
            const Worker& v=workers[p][w]; const Pin& pin=pins[w];
            std::fprintf(file,"%s{\"worker\":%d,\"start_us\":%.3f,\"end_us\":%.3f,\"cpu_us\":%.3f,\"cpu_start\":%d,\"cpu_end\":%d,\"core_type\":%d,\"pin_target\":%d,\"pin_errno\":%d,\"pin_calls\":%u,\"calls\":%llu,\"bands\":%llu,\"row_bands\":%llu,\"weight_tiles\":%llu,\"weight_bytes\":%llu}",
                w?",":"",w,v.start-phase_starts[p],v.end-phase_starts[p],v.cpu_us,v.cpu_start,v.cpu_end,v.core_type,
                pin.target,pin.error,pin.calls,(unsigned long long)v.calls,(unsigned long long)v.bands,
                (unsigned long long)v.row_bands,(unsigned long long)v.weight_tiles,(unsigned long long)v.weight_bytes);
        }
        std::fprintf(file,"]}");
    }
    uint64_t hash=14695981039346656037ull;
    const unsigned char* bytes=reinterpret_cast<const unsigned char*>(output);
    for(size_t i=0;i<count*sizeof(float);++i) { hash^=bytes[i];hash*=1099511628211ull; }
    std::fprintf(file,"],\"output_fnv1a64\":\"%016llx\"}\n",(unsigned long long)hash);
    std::fflush(file);
    if(m<1024 && seen_m[m]++==0) {
        const char* prefix=std::getenv("K11_DIAG_DUMP_PREFIX");
        if(prefix && *prefix) {
            const std::string path=std::string(prefix)+"-m"+std::to_string(m)+".f32";
            FILE* f=std::fopen(path.c_str(),"wb");
            if(!f || std::fwrite(output,sizeof(float),count,f)!=count || std::fclose(f)) {
                std::fprintf(stderr,"K11 output dump failed: %s\n",path.c_str()); std::abort();
            }
        }
    }
}
} // namespace k11diag
