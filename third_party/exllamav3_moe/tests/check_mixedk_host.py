"""Compile and run the unchanged x86 CPU bands on macOS through Rosetta.

This is not a build of the production Linux/Windows CPU layer or thread pool.
Only platform includes, target annotations, and ISA detection are substituted.
"""
from pathlib import Path
import os
import sys
import subprocess
import tempfile

VENDOR = Path(__file__).resolve().parents[1]


def registry():
    full = (VENDOR / "moe_mul1.cpp").read_text()
    begin = full.index("static MoeCpuMatrix make_matrix_raw")
    end = full.index("static const MoeCpuLayer* get_layer(int64_t handle)\n{", begin)
    source = '''#include "moe_mul1.h"
#include <cassert>
#include <cstdio>
#include <mutex>
#include <limits>
#include <memory>
std::vector<MoeCpuLayer*> g_layers;
std::mutex g_layers_mutex;
''' + full[begin:end] + r'''
int main() {
    uint16_t packed[2]{};
    at::Half scale[2]{};
    MoeCpuMatrixDesc g[6], u[6], d[6];
    for (int e = 0; e < 6; ++e) {
        g[e] = {packed, scale, scale, 16, 8, 16 * (e + 1)};
        u[e] = {packed, scale, scale, 16, 8, 16 * ((e + 1) % 6 + 1)};
        d[e] = {packed, scale, scale, 8, 16, 16 * ((e + 5) % 6 + 1)};
    }
    for (int swizzled : {0, 1}) {
        const auto handle = exl3_moe_cpu_make_layer_raw(g, u, d, 6, 0, 10.f, swizzled);
        auto* layer = g_layers[handle];
        for (int e = 0; e < 6; ++e) {
            assert(layer->gates[e].bits == g[e].tile_w / 16);
            assert(layer->ups[e].bits == u[e].tile_w / 16);
            assert(layer->downs[e].bits == d[e].tile_w / 16);
            auto ng = g[e], nu = u[e], nd = d[e];
            ng.trellis = nu.trellis = nd.trellis = packed + 1;
            exl3_moe_cpu_set_expert_raw(handle, e, &ng, &nu, &nd, swizzled);
            assert(layer->gates[e].trellis == packed + 1);
            assert(layer->ups[e].trellis == packed + 1);
            assert(layer->downs[e].trellis == packed + 1);
            assert(layer->gates[e].bits == g[e].tile_w / 16);
            assert(layer->ups[e].bits == u[e].tile_w / 16);
            assert(layer->downs[e].bits == d[e].tile_w / 16);
            // The same expert must keep its rate. Check atomic rejection.
            ng.tile_w = ng.tile_w == 96 ? 16 : ng.tile_w + 16;
            nu.trellis = nd.trellis = packed;
            bool refused = false;
            try { exl3_moe_cpu_set_expert_raw(handle, e, &ng, &nu, &nd, swizzled); }
            catch (const std::runtime_error&) { refused = true; }
            assert(refused);
            assert(layer->ups[e].trellis == packed + 1);
            assert(layer->downs[e].trellis == packed + 1);
        }
        exl3_moe_cpu_free_layer(handle);
    }
    std::puts("PASS extracted raw CPU registry: 6 mixed experts, per-projection K1..K6, relocation, atomic rate rejection, native/swizzled descriptors");
}
'''
    with tempfile.TemporaryDirectory(prefix="mixedk-registry-") as name:
        tmp = Path(name)
        (tmp / "test.cpp").write_text(source)
        subprocess.run([os.environ.get("CXX", "clang++"), "-std=c++17", "-Wall", "-Wextra", "-Werror",
                        "-I", str(VENDOR), str(tmp / "test.cpp"), "-o", str(tmp / "test")], check=True)
        subprocess.run([str(tmp / "test")], check=True)


def main():
    registry()
    if "--registry-only" in sys.argv:
        return
    source = (VENDOR / "moe_mul1.cpp").read_text()
    source = source[:source.index("Isa detect_isa()")]
    start = source.index("#ifdef __linux__\n#include <pthread.h>")
    end = source.index("#endif", start) + len("#endif")
    source = source[:start] + source[end:]
    # Apple clang accepts the same per-function ISA target annotations.
    source = source.replace('#if defined(__GNUC__) && defined(__linux__)\n#define M1_TARGET_AVX2',
                            '#if defined(__GNUC__)\n#define M1_TARGET_AVX2')
    source += '''const Isa g_isa = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")
        ? Isa::Avx2 : Isa::Scalar;\n'''
    full = (VENDOR / "moe_mul1.cpp").read_text()
    source += full[full.index("void run_tiles("):full.index("//   Thread pool")].rsplit("// ---", 1)[0]
    source += "\n} // namespace\n#endif\n"
    with tempfile.TemporaryDirectory(prefix="mixedk-bands-") as name:
        tmp = Path(name)
        (tmp / "mixedk_bands.inc").write_text(source)
        cmd = [os.environ.get("CXX", "clang++"), "-arch", "x86_64", "-std=c++17", "-O0",
               "-DMIXEDK_BANDS_ONLY", "-I", str(tmp), "-I", str(VENDOR),
               str(VENDOR / "tests/strata_mixedk_test.cpp"), "-o", str(tmp / "test")]
        subprocess.run(cmd, check=True)
        print("PASS x86_64 band harness compilation (extracted source; no production pool)", flush=True)
        result = subprocess.run([str(tmp / "test")])
        if result.returncode == 77:
            print("NOT RUN: x86 band numerical checks require AVX2/FMA", flush=True)
            raise SystemExit(77)
        result.check_returncode()


if __name__ == "__main__":
    main()
