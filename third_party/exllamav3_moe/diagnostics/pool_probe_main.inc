// Appended to a verbatim copy of the current Pool. Independent mechanism probe.
struct ProbeWorker { int cpu=-1, type=-1; double cpu_us=0; };
static ProbeWorker probe_workers[512];
static void probe(void*,int w,int) {
    probe_workers[w].cpu=sched_getcpu(); probe_workers[w].type=k11diag::core_type();
}
static void noop(void*,int,int) {}
int main(int argc,char** argv) {
    const int n=argc>1?atoi(argv[1]):8, iterations=argc>2?atoi(argv[2]):20;
    if(n<1 || n>512 || iterations<1) return 2;
    cpu_set_t mask; CPU_ZERO(&mask); sched_getaffinity(0,sizeof mask,&mask);
    printf("{\"kind\":\"pool_metadata\",\"requested\":%d,\"incoming_affinity\":[",n);
    bool comma=false;for(int c=0;c<CPU_SETSIZE;++c)if(CPU_ISSET(c,&mask)){printf("%s%d",comma?",":"",c);comma=true;}
    printf("]}\n");
    g_pool.ensure(n); g_pool.run(probe,nullptr);
    for(int w=0;w<n;++w) printf("{\"kind\":\"worker\",\"worker\":%d,\"cpu\":%d,\"core_type\":%d,\"pin_target\":%d,\"pin_errno\":%d}\n",w,probe_workers[w].cpu,probe_workers[w].type,k11diag::pins[w].target,k11diag::pins[w].error);
    for(int i=0;i<3;++i)g_pool.run(noop,nullptr);
    for(int sample=0;sample<11;++sample) {
        const double start=k11diag::wall();
        for(int i=0;i<iterations;++i) for(int phase=0;phase<5;++phase) g_pool.run(noop,nullptr);
        printf("{\"kind\":\"noop_five_dispatch_burst\",\"sample\":%d,\"iterations\":%d,\"us_per_five\":%.3f}\n",sample,iterations,(k11diag::wall()-start)/iterations);
    }
    fflush(stdout);std::_Exit(0); // Pool helpers intentionally live until process exit.
}
