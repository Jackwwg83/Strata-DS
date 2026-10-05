// CPU-only validation; uses CUDA's host FP8/BF16 conversions, never a GPU.
// Compile as C++17 with CUDA runtime and CCCL include directories on the path.
#include "packed_decode.cuh"

#include <cstdint>
#include <cstdio>
#include <cstring>

static unsigned short bits(__nv_bfloat16 value) {
    unsigned short result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

static bool check(unsigned packed, unsigned scale_byte) {
    alignas(8) __nv_bfloat16 got[4];
    strata::ds41::kernels::k2_detail::store_four(got, packed, scale_byte);
    const float scale = ldexpf(1.0f, static_cast<int>(scale_byte) - 127);
    for (int lane = 0; lane < 4; ++lane) {
        __nv_fp8_e4m3 q;
        q.__x = static_cast<unsigned char>(packed >> (8 * lane));
        const auto expected = __float2bfloat16_rn(static_cast<float>(q) * scale);
        if (bits(got[lane]) != bits(expected)) {
            std::printf("FAIL packed=%08x scale=%u lane=%d got=%04x expected=%04x\n",
                        packed, scale_byte, lane, bits(got[lane]), bits(expected));
            return false;
        }
    }
    return true;
}

int main() {
    unsigned long long checked = 0;
    unsigned fast_scalar = 0;
    for (unsigned scale = 0; scale < 256; ++scale) {
        for (unsigned byte = 0; byte < 256; ++byte) {
            // Exhaust every FP8 x E8M0 combination in all four byte positions,
            // both alone and mixed with distinct finite normal neighbors.
            const unsigned repeated = byte * 0x01010101u;
            if (!check(repeated, scale)) return 1;
            ++checked;
            fast_scalar += strata::ds41::kernels::k2_detail::normal_four(repeated, scale);
            for (unsigned lane = 0; lane < 4; ++lane) {
                const unsigned shift = lane * 8;
                const unsigned packed = (0xfe087738u & ~(255u << shift)) | (byte << shift);
                if (!check(packed, scale)) return 1;
                ++checked;
            }
        }
    }
    // Exhaust all neighboring byte pairs at safe-range boundaries and the
    // benchmark's five scales; this also checks that pair arithmetic cannot
    // leak sign/exponent carries between adjacent BF16 lanes.
    const unsigned scales[] = {6, 7, 118, 119, 120, 121, 122, 246, 247, 255};
    for (unsigned scale : scales) {
        for (unsigned pair = 0; pair < 65536; ++pair) {
            if (!check(pair | ((pair ^ 0x8080u) << 16), scale)) return 1;
            ++checked;
        }
    }
    std::printf("PASS: all 256 FP8 bytes x 256 E8M0 scales, all four positions; "
                "%llu packed groups / %llu BF16 lanes; %u scalar fast-path combinations\n",
                checked, checked * 4, fast_scalar);
    return 0;
}
