"""Check the production deq body with host models of CUDA scalar types.

This checks scalar arithmetic only. It does not compile or run a CUDA kernel.
"""
from pathlib import Path
import argparse
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
p = argparse.ArgumentParser()
p.add_argument("--source-ref")
a = p.parse_args()
path = "src/ds41/wo_a_fp8.cu"
s = (subprocess.check_output(["git", "show", f"{a.source_ref}:{path}"], text=True)
     if a.source_ref else (ROOT / path).read_text())
begin = s.index("__device__ __forceinline__ float deq(")
end = s.index("\n}\n", begin) + 3
body = s[begin:end]
prefix = r'''
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#define __device__
#define __forceinline__ inline
float __uint_as_float(uint32_t bits) { float f; std::memcpy(&f, &bits, 4); return f; }
uint16_t __float2bfloat16_rn(float f) {
    uint32_t bits; std::memcpy(&bits, &f, 4);
    if (std::isnan(f)) return 0x7FC0;
    return uint16_t((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}
float __bfloat162float(uint16_t bits) { return __uint_as_float(uint32_t(bits) << 16); }
struct __nv_fp8_e4m3 {
    uint8_t __x;
    operator float() const {
        const int e = (__x >> 3) & 15, m = __x & 7;
        if (e == 15 && m == 7) return std::numeric_limits<float>::quiet_NaN();
        const float v = e ? std::ldexp(1.f + m / 8.f, e - 7) : std::ldexp(float(m), -9);
        return (__x & 128) ? -v : v;
    }
};
'''
test = r'''
int main() {
    int failures = 0;
    for (int code : {0, 1, 2, 112, 127, 254, 255}) {
        int bad = 0;
        for (int w = 0; w < 256; ++w) {
            if ((w & 127) == 127) continue;
            __nv_fp8_e4m3 v{uint8_t(w)};
            const float scale = code == 255 ? std::numeric_limits<float>::quiet_NaN() : std::ldexp(1.f, code - 127);
            const float want = __bfloat162float(__float2bfloat16_rn(float(v) * scale));
            const float got = deq(uint8_t(w), uint8_t(code));
            const bool equal = std::isnan(want) ? std::isnan(got) : std::memcmp(&want, &got, 4) == 0;
            bad += !equal;
        }
        std::printf("%s E8M0 code %d: %d mismatches among 254 finite FP8 values\n", bad ? "FAIL" : "PASS", code, bad);
        failures += bad;
    }
    std::printf("RESULT %s scalar dequant (%d failures)\n", failures ? "fail" : "pass", failures);
    return failures ? 1 : 0;
}
'''
with tempfile.TemporaryDirectory(prefix="revfix-e8m0-") as name:
    tmp = Path(name)
    (tmp / "test.cpp").write_text('#include <initializer_list>\n' + prefix + body + test)
    subprocess.run([os.environ.get("CXX", "clang++"), "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    str(tmp / "test.cpp"), "-o", str(tmp / "test")], check=True)
    raise SystemExit(subprocess.run([str(tmp / "test")]).returncode)
