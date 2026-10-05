// Independent host model of BF16 tile rounding and online rescaling.
// This is not a GPU acceptance test or a tensor-core execution emulator.
// Run with: c++ -O2 -std=c++17 validate_host.cpp -o /tmp/k3-model && /tmp/k3-model
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <vector>
static float bf(float x) { uint32_t u; std::memcpy(&u,&x,4); u=(u+(0x7fff+((u>>16)&1)))&0xffff0000u; std::memcpy(&x,&u,4); return x; }
static std::vector<float> randn(int n,int seed,bool rounded=true) { std::mt19937 g(seed);std::normal_distribution<float>d(0,1);std::vector<float>a(n);for(float&x:a){x=d(g);if(rounded)x=bf(x);}return a; }
static void validate_layout() {
    // Direct B-fragment loads cover every element of one 8-head query tile.
    int queries[8][512] = {};
    for (int lane = 0; lane < 32; ++lane)
        for (int k = 0; k < 32; ++k)
            for (int reg = 0; reg < 2; ++reg)
                for (int elem = 0; elem < 2; ++elem) {
                    const int h = lane / 4;
                    const int d = k * 16 + (lane % 4) * 2 + reg * 8 + elem;
                    assert(h < 8 && d < 512);
                    assert(++queries[h][d] == 1);
                }
    for (auto& head : queries) for (int n : head) assert(n == 1);

    // C fragments: the four registers are (row,h), (row,h+1),
    // (row+8,h), (row+8,h+1). Check coverage and shuffle ownership.
    int scores[8][16] = {};
    for (int lane = 0; lane < 32; ++lane) {
        const int h = (lane % 4) * 2, r = lane / 4;
        for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b)
            assert(++scores[h + a][r + b * 8] == 1);
        for (int off : {4, 8, 16}) assert((lane ^ off) % 4 == lane % 4);
    }
    for (auto& head : scores) for (int n : head) assert(n == 1);
    float lanes[32];
    for (int lane = 0; lane < 32; ++lane) lanes[lane] = (lane / 4 + 1) + (lane / 4 + 9);
    for (int off : {4, 8, 16}) {
        float old[32]; std::copy(lanes, lanes + 32, old);
        for (int lane = 0; lane < 32; ++lane) lanes[lane] += old[lane ^ off];
    }
    for (float value : lanes) assert(value == 136.0f);

    // The transposed ldmatrix source quadrants describe V^T exactly.
    int values[16][16] = {};
    for (int address_lane = 0; address_lane < 32; ++address_lane) {
        const int key = address_lane % 8 + address_lane / 16 * 8;
        const int dim = (address_lane / 8 % 2) * 8;
        for (int col = 0; col < 8; ++col) assert(++values[key][dim + col] == 1);
    }
    for (auto& key : values) for (int n : key) assert(n == 1);

    // Full launch coverage: each output element has exactly one writer.
    for (int m = 1; m <= 8; ++m) {
        const int od = m <= 2 ? 16 : m <= 4 ? 32 : 64;
        std::vector<int> written(m * 64 * 512);
        for (int t = 0; t < m; ++t) for (int group = 0; group < 2; ++group)
            for (int output = 0; output < 512; output += od)
                for (int warp = 0; warp < 4; ++warp) for (int lane = 0; lane < 32; ++lane)
                    for (int v = 0; v < od / 16; ++v)
                        for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) {
                            const int h = group * 32 + warp * 8 + (lane % 4) * 2 + a;
                            const int d = output + v * 16 + lane / 4 + b * 8;
                            assert(++written[(t * 64 + h) * 512 + d] == 1);
                        }
        for (int n : written) assert(n == 1);
    }
    // The producer packets cover each logical KV element, and all packets
    // have a 16-byte-aligned address despite the padded 520-element stride.
    int packets[16][64] = {};
    for (int thread = 0; thread < 128; ++thread)
        for (int i = thread; i < 16 * 64; i += 128) {
            const int row = i / 64, dim = (i % 64) * 8;
            assert((row * 520 + dim) % 8 == 0);
            assert(++packets[row][dim / 8] == 1);
        }
    for (auto& row : packets) for (int n : row) assert(n == 1);
    std::puts("PASS: register-Q/MMA/PV/shuffle/gather layouts and full m=1..8 output coverage");
}
int main() {
    validate_layout();
    auto w=randn(128*512,1), c=randn(4096*512,2), sink=randn(64,3,false);
    const float scale=1/std::sqrt(512.f);
    struct Case { int m,n,valid; int mode=0; };
    const Case cases[]={{1,640,128},{4,640,128},{8,640,128},{1,128,37},{2,300,128},{3,1024,5},{1,128,0},{1,0,0},{1,1,1},{1,17,13},{1,65,37},{1,1024,0},{1,1024,0,1},{2,65,0,1},{3,33,0,2},{1,17,1,3},{1,17,1,4}};
    bool pass=true;
    for(auto cc:cases) {
        auto q=randn(cc.m*64*512,10+cc.m);
        std::vector<int> idx(cc.m*cc.n,-1);
        std::mt19937 g(20+cc.m);
        for(int t=0;t<cc.m;t++) {
            for(int i=0;i<128&&i<cc.n;i++)idx[t*cc.n+i]=i<cc.valid?i:-1;
            std::set<int> picked;
            while((int)picked.size()<std::min(cc.n-128,4096))picked.insert(g()%4096);
            int j=128;for(int p:picked)idx[t*cc.n+j++]=128+p;
        }
        if(cc.mode == 1) std::fill(idx.begin(), idx.end(), -7);
        if(cc.mode == 2) for(size_t i=0;i<idx.size();++i) idx[i]=(i%4==0?-3:(i%3==0?130:7));
        double num=0,den=0;
        for(int t=0;t<cc.m;t++)for(int h=0;h<64;h++) {
            std::vector<float> reference_scores(cc.n,-INFINITY), scores(cc.n,-INFINITY);
            std::vector<const float*> rows(cc.n,nullptr);
            float maximum=-1e30f;
            for(int k=0;k<cc.n;k++) {
                const int j=idx[t*cc.n+k]; if(j<0)continue;
                const float* r=j<128?w.data()+j*512:c.data()+(j-128)*512;
                rows[k]=r;
                float lanes[32]={}, sequential=0;
                for(int lane=0;lane<32;lane++)for(int d=lane;d<512;d+=32)
                    lanes[lane]=std::fma(q[(t*64+h)*512+d],r[d],lanes[lane]);
                for(int off=16;off;off>>=1)for(int lane=0;lane<off;lane++)lanes[lane]+=lanes[lane+off];
                // Sequential FP32 accumulation is a sensitivity check for a
                // different QK reduction order, not a promise about hardware.
                for(int d=0;d<512;d++) sequential=std::fma(q[(t*64+h)*512+d],r[d],sequential);
                reference_scores[k]=lanes[0]*scale; scores[k]=sequential*scale;
                maximum=std::max(maximum,reference_scores[k]);
            }
            float ref[512]={}, out[512]={}, refsum=0;
            for(int k=0;k<cc.n;k++) {
                const float p=std::isinf(reference_scores[k])?0:std::exp(reference_scores[k]-maximum);
                refsum+=p;
                if(rows[k])for(int d=0;d<512;d++)ref[d]=std::fma(bf(p),rows[k][d],ref[d]);
            }
            const float sh = cc.mode == 3 ? 80.0f : cc.mode == 4 ? -80.0f : sink[h];
            refsum+=std::exp(sh-maximum);
            float running_max=-1e30f,running_sum=0;
            for(int first=0;first<cc.n;first+=16) {
                const int end=std::min(cc.n,first+16);
                float next_max=running_max;
                for(int k=first;k<end;k++)next_max=std::max(next_max,scores[k]);
                const float alpha=std::exp(running_max-next_max);
                for(int d=0;d<512;d++)out[d]*=alpha;
                float lanes[8]={};
                for(int k=first;k<end;k++) {
                    const float p=std::isinf(scores[k])?0:std::exp(scores[k]-next_max);
                    lanes[(k-first)%8]+=p;
                    if(rows[k])for(int d=0;d<512;d++)out[d]=std::fma(bf(p),rows[k][d],out[d]);
                }
                for(int off=1;off<=4;off*=2) { float old[8]; std::copy(lanes,lanes+8,old); for(int lane=0;lane<8;lane++)lanes[lane]+=old[lane^off]; }
                running_sum=std::fma(running_sum,alpha,lanes[0]);
                running_max=next_max;
            }
            running_sum+=std::exp(sh-running_max);
            for(int d=0;d<512;d++) {
                const float a=bf(ref[d]/refsum),b=bf(out[d]/running_sum);
                const double diff=b-a;num+=diff*diff;den+=double(a)*a;
            }
        }
        const double error=std::sqrt(num/std::max(den,1e-300));
        std::printf("CPU online16/register-Q model m=%d n_idx=%d window_valid=%d mode=%d rel_l2=%.9g %s\n",cc.m,cc.n,cc.valid,cc.mode,error,error<=3e-3?"PASS":"FAIL");
        pass &= error<=3e-3;
    }
    std::puts("Model covers BF16 P, FP32 denominator, sink-only denominator, tails, repeated/negative/all-empty indices; GPU parity remains untested.");
    return pass?0:1;
}
