// CPU-only CUDA-header conversion/packing probe. No kernel launch or GPU check.
// Compare the control's BF16 pairs with the new raw-word packaging for all
// 256 FP8 encodings and all 256 E8M0 scale bytes in each of eight packet slots.
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>

static unsigned word(__nv_bfloat162 value) {
    const __nv_bfloat162_raw raw = value;
    return static_cast<unsigned>(raw.x) | (static_cast<unsigned>(raw.y) << 16);
}

static void control(unsigned packed, float scale, __nv_bfloat162* out) {
    __nv_fp8x4_e4m3 q;
    q.__x = packed;
    const float4 v = static_cast<float4>(q);
    out[0] = __floats2bfloat162_rn(v.x * scale, v.y * scale);
    out[1] = __floats2bfloat162_rn(v.z * scale, v.w * scale);
}

static unsigned scaled_pair(float x, float y, float scale) {
    const __nv_bfloat162_raw pair = __floats2bfloat162_rn(x * scale, y * scale);
    return static_cast<unsigned>(pair.x) | (static_cast<unsigned>(pair.y) << 16);
}

static void candidate(uint2 packed, float scale, unsigned* out) {
    __nv_fp8x4_e4m3 q0, q1;
    q0.__x = packed.x;
    q1.__x = packed.y;
    const float4 v0 = static_cast<float4>(q0);
    const float4 v1 = static_cast<float4>(q1);
    out[0] = scaled_pair(v0.x, v0.y, scale);
    out[1] = scaled_pair(v0.z, v0.w, scale);
    out[2] = scaled_pair(v1.x, v1.y, scale);
    out[3] = scaled_pair(v1.z, v1.w, scale);
}

int main() {
    unsigned packets = 0;
    for (unsigned byte = 0; byte < 256; ++byte) {
        uint2 packed = make_uint2(0, 0);
        for (unsigned i = 0; i < 8; ++i) {
            const unsigned value = ((byte + 47 * i) & 255u) << (8 * (i % 4));
            if (i < 4) packed.x |= value;
            else packed.y |= value;
        }
        for (unsigned exponent = 0; exponent < 256; ++exponent) {
            const float scale = ldexpf(1.0f, int(exponent) - 127);
            __nv_bfloat162 expected[4];
            unsigned got[4];
            control(packed.x, scale, expected);
            control(packed.y, scale, expected + 2);
            candidate(packed, scale, got);
            for (unsigned pair = 0; pair < 4; ++pair) {
                if (word(expected[pair]) != got[pair]) {
                    std::printf("FAIL byte=%u scale=%u pair=%u expected=%08x got=%08x\n",
                                byte, exponent, pair, word(expected[pair]), got[pair]);
                    return 1;
                }
            }
            ++packets;
        }
    }
    std::printf("PASS CPU CUDA-header conversion: %u packets, %u BF16 values, bit-exact pair order including NaN/zero encodings\n",
                packets, packets * 8);
    std::puts("Host conversion check only; device instruction behavior and GPU numerical acceptance are not established");
}
