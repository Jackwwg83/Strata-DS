// Host-only rejection paths; no CUDA device is needed to exercise these.
#include "strata/ds41/kernels/k15_hc_prefill.hpp"
#include <cstring>
#include <cstdio>
using namespace strata::ds41::kernels;
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    if(!std::strcmp(argv[1],"sizes")) {
        if(hc_mixes_pre_rows_workspace_bytes(0)!=0 || hc_mixes_pre_rows_workspace_bytes(-1)!=0 ||
           hc_mixes_pre_rows_workspace_bytes(16385)!=0 || hc_mixes_pre_rows_workspace_bytes(1)!=768 ||
           hc_mixes_pre_rows_workspace_bytes(16384)!=12582912)return 1;
        std::puts("workspace sizes pass");return 0;
    }
    int m=1;size_t n=768;alignas(float) unsigned char storage[768];void* p=storage;
    if(!std::strcmp(argv[1],"m0"))m=0;
    else if(!std::strcmp(argv[1],"mneg"))m=-1;
    else if(!std::strcmp(argv[1],"mhigh"))m=16385;
    else if(!std::strcmp(argv[1],"null"))p=nullptr;
    else if(!std::strcmp(argv[1],"small"))n=767;
    else if(!std::strcmp(argv[1],"align"))p=storage+1;
    else return 2;
    hc_mixes_pre_rows(nullptr,m,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,p,n,0);
    std::puts("ERROR: invalid arguments were accepted");return 1;
}
