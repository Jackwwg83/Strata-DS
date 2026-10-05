#!/usr/bin/env python3
"""CPU-only model of the actual scaled decoder plus the existing host parity suite.

Run from any directory with Python 3 and a C++17 compiler:
    python3 src/ds41/kernels/fp8_gemv/check_scaled_decode.py [--full]
This extracts the CUDA helper, substitutes it into a temporary copy of the fixed
host lane model, and leaves the fixed header and acceptance tests untouched.
It does not execute CUDA and is not a GPU correctness or performance result.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--full", action="store_true", help="also run every production shape")
args = parser.parse_args()
root = Path(__file__).resolve().parents[4]
kernel = (root / "src/ds41/kernels/fp8_gemv.cu").read_text()
start = kernel.index("__device__ __forceinline__ float decode_scaled_e4m3(")
end = kernel.index("\nvoid check(", start)
helper = kernel[start:end].replace("__device__ __forceinline__", "inline")
parity = (root / "src/ds41/kernels/fp8_gemv_parity.cpp").read_text()

def replace_once(old, new):
    global parity
    assert parity.count(old) == 1, "host model source changed: " + old
    parity = parity.replace(old, new)

replace_once("namespace device_math = strata::ds41::detail;",
             "namespace device_math = strata::ds41::detail;\n"
             "namespace detail = strata::ds41::detail;\n" + helper)
replace_once(
    "const float sw = device_math::decode_e8m0(scales[size_t((row / 32) * (s.k / 32) + col / 32)]);",
    "const uint8_t scale = scales[size_t((row / 32) * (s.k / 32) + col / 32)];\n"
    "                const float sw = device_math::decode_e8m0(scale);\n"
    "                const uint32_t exponent_bias = (uint32_t(scale) - 7u) << 23;\n"
    "                const bool normal_scale = uint32_t(scale) - 7u < 240u;")
replace_once(
    "device_math::decode_e4m3(w[size_t(row * s.k + col + j)]) * sw",
    "decode_scaled_e4m3(w[size_t(row * s.k + col + j)], exponent_bias, normal_scale, sw)")
# Also exercise every m in the ownership tests, beyond their original {1, 2, 8}.
replace_once("for (int m : {1, 2, 8}) {", "for (int m = 1; m <= 8; ++m) {")
exhaustive = r'''
void scaled_decode_exhaustive() {
    size_t fast = 0, fallback = 0;
    for (unsigned scale = 0; scale < 256; ++scale) {
        const float sw = detail::decode_e8m0(uint8_t(scale));
        const uint32_t bias = (uint32_t(scale) - 7u) << 23;
        const bool normal = uint32_t(scale) - 7u < 240u;
        for (unsigned byte = 0; byte < 256; ++byte) {
            const float got = decode_scaled_e4m3(uint8_t(byte), bias, normal, sw);
            const float want = detail::decode_e4m3(uint8_t(byte)) * sw;
            if (detail::float_bits(got) != detail::float_bits(want)) {
                std::fprintf(stderr, "scale=%u byte=%u got=%08x want=%08x\n",
                             scale, byte, detail::float_bits(got), detail::float_bits(want));
                require(false, "scaled decoder bitwise mismatch, including NaN payload/sign");
            }
            const bool common = normal && (byte & 127u) - 8u < 119u;
            (common ? fast : fallback)++;
            if (common) require(std::isnormal(got), "fast result must be normal");
            // Independent arithmetic decoding, including scale 0 and scale 255.
            const float arithmetic_scale = std::ldexp(1.0f, int(scale) - 127);
            const float arithmetic = float(ref_decode(uint8_t(byte))) * arithmetic_scale;
            require(std::isnan(arithmetic) ? std::isnan(got) :
                    detail::float_bits(arithmetic) == detail::float_bits(got),
                    "independent arithmetic scaled decode");
        }
    }
    require(fast == 57120 && fallback == 8416, "exhaustive path counts");
    // Invalid output rows use packed zero and sw=0, unlike any E8M0 byte.
    for (unsigned byte : {0u, 128u})
        require(detail::float_bits(decode_scaled_e4m3(uint8_t(byte), 0, false, 0.0f)) ==
                detail::float_bits(detail::decode_e4m3(uint8_t(byte)) * 0.0f), "masked row zero");
    std::printf("scaled decode: all 65536 byte/scale pairs bit-exact; %zu fast, %zu fallback OK\n",
                fast, fallback);
}
'''
replace_once("void host_selftest(bool full_shapes) {",
             exhaustive + "\nvoid host_selftest(bool full_shapes) {\n    scaled_decode_exhaustive();")
with tempfile.TemporaryDirectory(prefix="k1c-scaled-decode-") as directory:
    source = Path(directory) / "scaled_decode_model.cpp"
    binary = Path(directory) / "scaled_decode_model"
    source.write_text(parity)
    subprocess.run([os.environ.get("CXX", "c++"), "-O2", "-std=c++17", "-fno-fast-math",
                    "-DSTRATA_DS41_HOST_ONLY", "-I" + str(root / "include"), str(source),
                    "-o", str(binary)], check=True)
    subprocess.run([str(binary), "--host-shapes" if args.full else "--host-selftest"], check=True)
