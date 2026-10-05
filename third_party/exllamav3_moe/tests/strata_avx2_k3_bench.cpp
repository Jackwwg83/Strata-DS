// Supplemental synthetic microbenchmark; timings are not K11 acceptance scores.
// Build: g++ -O3 -std=c++17 -pthread strata_avx2_k3_bench.cpp -o /tmp/k11-k3-bench
#define main correctness_main
#include "strata_avx2_k3_test.cpp"
#undef main
#include <ctime>
#include <chrono>
#include <numeric>
using Fn=void (*)(const MoeCpuMatrix&,const PreparedIn&,float*,int,int,int);
__attribute__((noinline)) void base_call(const MoeCpuMatrix& mat,const PreparedIn& in,float* out,int m,int t0,int t1){ avx2_tiles<3,false>(mat,in,out,m,t0,t1); }
__attribute__((noinline)) void new_call(const MoeCpuMatrix& mat,const PreparedIn& in,float* out,int m,int t0,int t1){ candidate(mat,in,out,m,t0,t1); }
double cpu_ns(){ timespec t; clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return double(t.tv_sec)*1e9+t.tv_nsec; }
double wall_ns(){return std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count();}
int main(){
 std::mt19937 rng(221); volatile float sink=0;
 for(auto shape:{std::pair<int,int>{16,16},{5120,128},{5120,2304},{2304,5120}}){int k=shape.first,n=shape.second;
  std::vector<uint16_t> packed(size_t(k)*n*3/16);for(auto& p:packed)p=uint16_t(rng());
  std::vector<int32_t> dup(4*k);PreparedIn in{};in.splat_dup=dup.data();for(int i=0;i<4;++i){in.q[i]=.03125f;for(int j=0;j<k;++j){int v=int(rng()%255)-127;uint32_t d=uint16_t(v)|(uint32_t(uint16_t(v))<<16);memcpy(&dup[i*k+j],&d,4);in.sum_x8[i]+=v;}}
  MoeCpuMatrix mat{};mat.trellis=packed.data();mat.k=k;mat.n=n;mat.bits=3;std::vector<float> out(4*n);int iters=k==16?100000:k==5120&&n==128?20:3;
  for(int m=1;m<=4;++m){std::vector<double> c[2],w[2];Fn fn[2]={base_call,new_call};for(int r=0;r<25;++r){for(int ord=0;ord<2;++ord){int z=(r+ord)&1;double a=cpu_ns(),aw=wall_ns();for(int i=0;i<iters;++i){fn[z](mat,in,out.data(),m,0,n/16);sink=out[i%n];}double bw=wall_ns(),b=cpu_ns();if(r>=4){c[z].push_back((b-a)/iters);w[z].push_back((bw-aw)/iters);}}}for(int z=0;z<2;++z){std::sort(c[z].begin(),c[z].end());std::sort(w[z].begin(),w[z].end());}printf("k=%d n=%d m=%d base_cpu_ns=%.2f new_cpu_ns=%.2f ratio=%.4f base_wall_ns=%.2f new_wall_ns=%.2f\n",k,n,m,c[0][10],c[1][10],c[1][10]/c[0][10],w[0][10],w[1][10]);fflush(stdout);}
 }return sink==12345;
}
