#!/usr/bin/env python3
"""K12 host audits. These do NOT compile CUDA or establish GPU correctness."""
from pathlib import Path
import hashlib
import os
import random
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[4]
VENDOR = ROOT / "third_party/exllamav3_gpu"
K12 = ROOT / "src/ds41/kernels/k12"
CU = K12.parent / "k12_exl3_moe_prefill.cu"


def section(text, start, end):
    return text[text.index(start):text.index(end, text.index(start))]


def provenance():
    with tempfile.TemporaryDirectory(prefix="k12-vendor-") as name:
        tmp = Path(name)
        restored = tmp / "third_party/exllamav3_gpu"
        shutil.copytree(VENDOR, restored)
        patch = ["git", "apply", "--unidiff-zero", str(VENDOR / "strata.patch")]
        subprocess.run(patch + ["--reverse"], cwd=tmp, check=True, capture_output=True)
        entries = []
        for line in (VENDOR / "UPSTREAM.sha256").read_text().splitlines():
            digest, path = line.split("  ", 1)
            entries.append(path)
            assert hashlib.sha256((restored / path).read_bytes()).hexdigest() == digest, path
        # The full tile decoder, shuffles and stores, and both wrappers, are unchanged.
        start = "template <int K, int cb, bool HALF = false>"
        pristine = section((restored / "quant/reconstruct.cu").read_text(), start, "// Index cb")
        current = section((VENDOR / "quant/reconstruct.cu").read_text(), start, "namespace strata_exl3")
        assert pristine == current
        subprocess.run(patch, cwd=tmp, check=True, capture_output=True)
        for path in entries:
            assert (restored / path).read_bytes() == (VENDOR / path).read_bytes(), path
    print(f"PASS vendor: {len(entries)} pristine hashes, reverse/forward patch, unchanged reconstruct device code")


def numerics_and_includes():
    k10 = (K12.parent / "k10/pipeline.cuh").read_text()
    k12 = (K12 / "pipeline.cuh").read_text()
    start, end = "    for (int i = 0; i < 128; i += 32)", "    __syncwarp();\n    half* a"
    old = section(k10, start, end)
    new = section(k12, start, "    __syncwarp();\n    had_hf_r_128_inner")
    assert old.replace("weights[slot]", "weights[row]") == new
    start = "    const unsigned bits = __float_as_uint(a);"
    assert section(k10, start, "\n}") == section(k12, start, "\n}")
    assert "constexpr float HAD_SCALE = 0.088388347648f;" in (K12 / "workspace.hpp").read_text()
    seen = set()

    def visit(path):
        path = path.resolve()
        if path in seen:
            return
        assert path.exists(), path
        seen.add(path)
        text = path.read_text()
        assert not re.search(r'#include\s*[<"](?:ATen|c10|torch)/', text), path
        for rel in re.findall(r'^#include\s+"([^"]+)"', text, re.M):
            visit(ROOT / "include" / rel if rel.startswith("strata/") else path.parent / rel)

    visit(CU)
    print(f"PASS source audit: K10 activation/casts and scale rounding preserved; {len(seen)} local includes, no Torch")


def layout():
    # Compile the real layout header and extracted pointer-binding struct.
    # half is replaced by uint16_t for storage only; this is NOT a CUDA compile.
    workspace = section((K12 / "pipeline.cuh").read_text(), "struct Workspace {", "__device__ inline void check_proj")
    source = '''#include <cassert>
#include <climits>
#include <vector>
#include "workspace.hpp"
using half = uint16_t;
namespace strata::ds41::kernels::k12 {
''' + workspace + '''}
int main() {
    using namespace strata::ds41::kernels::k12;
    std::vector<char> backing(Layout(INT_MAX).bytes() + 256);
    for (int limit : {1, 2, 37, 511, 512, 513, 8192, INT_MAX}) {
        Layout l(limit);
        for (int shift = 0; shift < 256; ++shift) {
            char* base = backing.data() + shift;
            Workspace w(base, l);
            auto ptr = [](const void* p) { return reinterpret_cast<uintptr_t>(p); };
            assert(ptr(w.trellis) >= ptr(base) && ptr(w.trellis) - ptr(base) <= 255);
            assert(ptr(w.blas) % 256 == 0);
            assert(ptr(w.matrices) >= ptr(w.trellis + 3));
            assert(w.input == w.matrices + 3 * MATRIX_ELEMENTS);
            assert(ptr(w.gu) == ptr(w.input + size_t(2) * l.rows * H));
            assert(ptr(w.down_input) == ptr(w.gu + size_t(2) * l.rows * F));
            assert(ptr(w.down) == ptr(w.down_input + size_t(l.rows) * F));
            assert(ptr(w.blas) == ptr(w.down + size_t(l.rows) * H));
            assert(ptr(w.blas) + BLAS_BYTES <= ptr(base) + l.bytes());
            for (int count = 1; count <= l.rows; ++count) {
                assert(ptr(w.input + size_t(2) * count * H) <= ptr(w.gu));
                assert(ptr(w.gu + size_t(2) * count * F) <= ptr(w.down_input));
                assert(ptr(w.down_input + size_t(count) * F) <= ptr(w.down));
                assert(ptr(w.down + size_t(count) * H) <= ptr(w.blas));
            }
        }
    }
    assert(Layout(INT_MAX).bytes() == 107741695);
    bool rejected = false;
    try { Layout bad(-1); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}
'''
    with tempfile.TemporaryDirectory(prefix="k12-layout-") as name:
        tmp = Path(name)
        (tmp / "layout.cpp").write_text(source)
        subprocess.run([os.environ.get("CXX", "clang++"), "-std=c++17", "-Wall", "-Wextra", "-Werror",
                        "-I", str(K12), str(tmp / "layout.cpp"), "-o", str(tmp / "layout")], check=True)
        subprocess.run([str(tmp / "layout")], check=True)
    print("PASS native C++17 layout: every base alignment, row-tile tails, disjoint buffers, INT_MAX bounded size")


def routing():
    rng = random.Random(12)
    cases = [[0] * 384, [0, 1, 0, 1, 0], [513, 0, 1025], [512] * 64]
    cases += [[rng.randrange(1100) if rng.random() < .25 else 0 for _ in range(384)] for _ in range(20)]
    for counts in cases:
        offsets = [rng.randrange(1, 5000)]  # always exercise nonzero off[0]
        for n in counts:
            offsets.append(offsets[-1] + n)
        # Model the host launch loop's boundaries; compare with an
        # independent flattened owner list, including empty groups and tails.
        visited = []
        for g, n in enumerate(counts):
            first = offsets[g]
            while first < offsets[g + 1]:
                count = min(offsets[g + 1] - first, 512)
                visited.extend((g, r) for r in range(first, first + count))
                first += count
        expected = [(g, offsets[g] + r) for g, n in enumerate(counts) for r in range(n)]
        assert visited == expected
        # Contributions only touch tokens named by the assignment list.
        tokens = [rng.choice([0, 1, 2, 3, 4, 6, 7]) for _ in visited]
        output = [3.25] * 9
        for token in tokens:
            output[token] += 1
        assert output[5] == output[8] == 3.25
        assert sum(output) == 9 * 3.25 + len(visited)
    print("PASS host routing model: empty groups, nonzero offsets, 512-row tails, duplicate tokens and untouched rows")


def main():
    provenance()
    numerics_and_includes()
    layout()
    routing()
    print("HOST AUDITS PASSED; CUDA compilation, GPU numerical acceptance, graph capture and timing NOT RUN")


if __name__ == "__main__":
    main()
