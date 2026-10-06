#!/usr/bin/env python3
"""Host-only provenance, indexing and scale checks; NOT CUDA/GPU validation.

Run from any directory: python3 src/ds41/kernels/k10/check_host.py
Requires NumPy and git. Temporary files stay outside the repo.
"""
from pathlib import Path
import hashlib
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
VENDOR = ROOT / "third_party/exllamav3_gpu"
CU = ROOT / "src/ds41/kernels/k10_exl3_moe.cu"
PIPELINE = CU.parent / "k10/pipeline.cuh"

# K10-08 combines this exact K10-03 cache substitution with K10-02 scheduling.
# Keep both branches explicit so fallback loads and all address/guard expressions are audited.
WEIGHT_LOAD_UPSTREAM = """            if constexpr (LSTRIDE < 32)
                return lane < LSTRIDE ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
            else
                return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
"""
WEIGHT_LOAD_CANDIDATE = """            // Keep the same scalar uint32 load, alignment, lane guard and address.
            // Narrow integer weights stream through L2 without filling L1; other
            // configurations retain the upstream streaming/evict-first policy.
            if constexpr (CFG == 0 && !HALF)
            {
                if constexpr (LSTRIDE < 32)
                    return lane < LSTRIDE ? __ldcg(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
                else
                    return __ldcg(bp + (size_t) i * slice_stride + l * LSTRIDE);
            }
            else
            {
                if constexpr (LSTRIDE < 32)
                    return lane < LSTRIDE ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
                else
                    return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
            }
"""


def section(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0]


def upstream_schedule(text):
    # Normalize only the explicitly reviewed prefetch scheduling changes. All
    # decode, MMA, fold and reduction code is still compared with pristine
    # upstream below. This is a source audit, not GPU numerical validation.
    replacements = [
        ("    constexpr int PF   = 2;                     // prefetch ring depth, independent of FOLD",
         "    constexpr int PF   = CFG == 0 ? 4 : 2;      // prefetch ring depth"),
        ("    constexpr int FOLD = CFG == 0 ? 4 : 2;      // upstream fp16->fp32 fold cadence",
         "    constexpr int FOLD = CFG == 0 ? 4 : 2;      // fp16->fp32 fold cadence (divides PF)"),
        ('    static_assert(FOLD % PF == 0, "prefetch ring must divide the arithmetic unroll");\n', ""),
        ("        // Keep the upstream arithmetic unroll/fold boundaries. Ring slots repeat\n"
         "        // within each fold group and are refilled before their next use.\n", ""),
        ("for (int ib = 0; ib < myn; ib += FOLD)", "for (int ib = 0; ib < myn; ib += PF)"),
        ("for (int d = 0; d < FOLD; ++d)", "for (int d = 0; d < PF; ++d)"),
        ("bw[l] = pf[d % PF][l];", "bw[l] = pf[d][l];"),
        ("pf[d % PF][l] = ld_b(i + PF, l);", "pf[d][l] = ld_b(i + PF, l);"),
    ]
    for new, old in replacements:
        assert text.count(new) == 1, new
        text = text.replace(new, old)
    return text


def prefetch_schedule():
    def trace(myn, pf, fold):
        ring = [None] * pf
        loads = []
        for d in range(pf):
            if d < myn:
                ring[d] = d
                loads.append(d)
        consumed, folded = [], []
        for ib in range(0, myn, fold):
            for d in range(fold):
                i = ib + d
                if i >= myn:
                    break
                slot = d % pf
                # A slot must contain exactly the weight slice to be decoded.
                assert ring[slot] == i
                consumed.append(ring[slot])
                if i + pf < myn:
                    ring[slot] = i + pf
                    loads.append(i + pf)
                if (d + 1) % fold == 0 or i + 1 == myn:
                    folded.append(i + 1)
        assert sorted(loads) == list(range(myn))  # no extra, missing or repeated loads
        return consumed, folded

    # Include zero work, both legal K10 warp chunks (9 and 20), ring tails,
    # fold tails and larger chunks. Narrow/wide arithmetic order is unchanged.
    for fold in (4, 2):
        for myn in range(258):
            assert trace(myn, 2, fold) == trace(myn, fold, fold)
    print("PASS prefetch schedule: two-slot ring preserves every consumed slice and fold boundary for 516 tail cases")


def k3_source(text):
    # Specialize only the new compile-time generic staging branches away.
    # Every substitution is exact. The upstream comparison below still covers
    # the complete K3 load/decode/MMA/fold/reduction body.
    replacements = [
        ('(bits >= 1 && bits <= 6 && cb == 2)', '(bits == 2 || bits == 3 || bits == 4)'),
        ('Strata GEMV supports integer K1..K6 and half-integer K1..K3 with mul1',
         'exl3_gemv_kernel supports 2, 3 and 4 bpw, and 1.5, 2.5 and 3.5 bpw with mul1'),
        ("""    // Strata: keep K2/K3/K4's register decoder and load schedule unchanged.
    // Other integer rates use the upstream shared-memory dq_dispatch template.
    constexpr bool GENERIC = !HALF && (bits == 1 || bits > 4);
    constexpr int TILE_LOADS = (TWORDS + 31) / 32;
    constexpr int STAGE_STRIDE = GENERIC ? TILE_LOADS * 32 : TWORDS;
""", ''),
        ('GENERIC ? WNT * TILE_LOADS : (TWO_PER_LOAD ? WNT / 2 : WNT)', 'TWO_PER_LOAD ? WNT / 2 : WNT'),
        ('GENERIC ? 32 : TWO_PER_LOAD ?', 'TWO_PER_LOAD ?'),
        ('(SMEM_STAGE || GENERIC) ? WK', 'SMEM_STAGE ? WK'),
        ('(SMEM_STAGE || GENERIC) ? LOADS', 'SMEM_STAGE ? LOADS'),
        ("""            if constexpr (GENERIC)
            {
                const int tile = l / TILE_LOADS;
                const int word = (l % TILE_LOADS) * 32 + lane;
                return word < TWORDS ? __ldcg(bp + (size_t) i * slice_stride + tile * TWORDS + word - lane) : 0;
            }
            else if constexpr (CFG == 0 && !HALF)
""", '            if constexpr (CFG == 0 && !HALF)\n'),
        ('if constexpr (SMEM_STAGE || GENERIC)', 'if constexpr (SMEM_STAGE)'),
        ('t * STAGE_STRIDE', 't * TWORDS'),
        ("""                    if constexpr (GENERIC)
                        dq_dispatch<bits, cb, false>(tp, lane << 3, f0, f1);
                    else if constexpr (HALF)""", '                    if constexpr (HALF)'),
    ]
    for new, old in replacements:
        assert new in text, new
        text = text.replace(new, old)
    return text


def provenance():
    with tempfile.TemporaryDirectory(prefix="k10-pristine-") as name:
        tmp = Path(name)
        restored = tmp / "third_party/exllamav3_gpu"
        shutil.copytree(VENDOR, restored)
        patch = ["git", "apply", "--unidiff-zero", str(VENDOR / "strata.patch")]
        subprocess.run(patch + ["--reverse"], cwd=tmp,
                       check=True, capture_output=True, text=True)
        manifest = {}
        changed = []
        for line in (VENDOR / "UPSTREAM.sha256").read_text().splitlines():
            digest, rel = line.split("  ", 1)
            manifest[rel] = digest
            assert hashlib.sha256((restored / rel).read_bytes()).hexdigest() == digest, rel
            if (restored / rel).read_bytes() != (VENDOR / rel).read_bytes():
                changed.append(rel)
        assert sorted(changed) == ["quant/exl3_gemv.cu", "quant/exl3_gemv.cuh",
                                   "quant/exl3_gemv_kernel.cuh", "quant/reconstruct.cu", "quant/reconstruct.cuh"]
        old = (restored / "quant/exl3_gemv_kernel.cuh").read_text()
        new = (VENDOR / "quant/exl3_gemv_kernel.cuh").read_text()
        new = k3_source(new)
        assert new.count(WEIGHT_LOAD_CANDIDATE) == 1
        assert new.count("__ldcg(") == 2
        new = upstream_schedule(new.replace(WEIGHT_LOAD_CANDIDATE, WEIGHT_LOAD_UPSTREAM))
        assert "__launch_bounds__(512, 2)" in (VENDOR / "quant/exl3_gemv.cu").read_text()
        for start, end in [
            ("namespace exl3_gemv_ns {", "}  // namespace exl3_gemv_ns"),
            ("    static_assert(HALF", "    auto grid = cooperative_groups::this_grid();"),
            ("    const int warp = threadIdx.x / 32;", "    #undef XP"),
        ]:
            old_part = section(old, start, end)
            new_end = "    const int warp = threadIdx.x / 32;" if "auto grid" in end else end
            assert old_part == section(new, start, new_end), start
        subprocess.run(patch, cwd=tmp, check=True, capture_output=True, text=True)
        for rel in manifest:
            assert (restored / rel).read_bytes() == (VENDOR / rel).read_bytes(), rel
        print(f"PASS provenance: {len(manifest)} pristine SHA-256 hashes; reverse/forward patch round trip")
        print("PASS preservation: K3 specialization plus exact cache/schedule substitutions; helpers, constants, decode/MMA/fold/reduction byte-identical")


def include_and_scope_checks():
    seen = set()
    def visit(path):
        path = path.resolve()
        if path in seen:
            return
        assert path.exists(), path
        seen.add(path)
        text = path.read_text()
        assert not re.search(r"#include\s*[<\"](?:ATen|c10|torch)/", text), path
        for rel in re.findall(r'^#include\s+"([^"]+)"', text, re.M):
            child = ROOT / "include" / rel if rel.startswith("strata/") else path.parent / rel
            visit(child)
    visit(CU)
    print(f"PASS unity include closure: {len(seen)} local files; no PyTorch include")
    active = CU.read_text() + PIPELINE.read_text() + (VENDOR / "quant/exl3_gemv.cu").read_text()
    assert not re.search(r"\b(?:cudaMalloc\w*|cudaFree\w*|cudaMemcpy\w*|cudaMemset\w*|"
                         r"cudaDeviceSynchronize|cudaStreamSynchronize|cudaGetDevice\w*|"
                         r"cudaLaunchCooperativeKernel)\s*\(", active)
    assert "grid.sync" not in (VENDOR / "quant/exl3_gemv_kernel.cuh").read_text()
    assert active.count("<<<") == 4  # GEMV wrapper called twice, plus three pipeline kernels
    print("PASS launch source audit: 5 launches; no allocation, host sync, device query or cooperative barrier")
    changed = subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=ROOT, text=True).splitlines()
    changed += subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
    def allowed(p):
        return (p in ("src/ds41/kernels/k10_exl3_moe.cu", "src/ds41/kernels/k12_exl3_moe_prefill.cu",
                      "ds41/ci/make_mixedk_golden.py") or
                p.startswith(("src/ds41/kernels/k10/", "third_party/exllamav3_gpu/",
                              "src/ds41/kernels/k12/", "third_party/exllamav3_moe/",
                              "ds41/tasks/MIXEDK.", "src/ds41/tests/")) or
                p in [f"ds41/tasks/K10.{s}.md" for s in ("QUESTIONS", "PROGRESS", "REPORT", "COMMITS")])
    assert all(allowed(p) for p in changed), [p for p in changed if not allowed(p)]
    print("PASS scope: all changed/untracked files are in the task allowlist")


def indexing():
    # Enumerate the narrow GEMV's actual trellis-load address formulas. Every
    # uint32 word of each shape must be read exactly once, with no OOB loads.
    for k, n in [(5120, 2304), (2304, 5120)]:
        chunk = (k // 16 + 15) // 16
        group, warp, i, load, lane = np.indices((n // 32, 16, chunk, 2, 24))
        ks = warp * chunk + i
        assert np.all(ks < k // 16)
        words = ks * (n // 16) * 24 + group * 48 + load * 24 + lane
        count = (k // 16) * (n // 16) * 24
        assert np.array_equal(np.sort(words.ravel()), np.arange(count))
        # Cache policy keeps uint32 access width. No 8/16-byte vector alignment
        # is assumed; every offset preserves the original 4-byte requirement.
        assert np.all((words * np.dtype(np.uint32).itemsize) % 4 == 0)
        # Only lanes 0..3 read A for MMODE=0; they read the low/high k halves.
        ar = ks[..., 0, 0, None] * 8 + np.arange(4)
        assert ar.min() == 0 and (ar + 4).max() == k // 2 - 1
        # Hadamard's scale uses y*32 + lane in half4 units.
        scales = np.arange(n // 128)[:, None, None] * 128 + np.arange(32)[None, :, None] * 4 + np.arange(4)
        assert np.array_equal(scales.ravel(), np.arange(n))
        print(f"PASS address model k={k} n={n}: {count} trellis words exactly once; input/scales in bounds")
    job_size = 40  # three 64-bit pointers, three int32 fields, padding; verified by host C++ check below
    for m in (1, 4, 8):
        slots = m * 6
        lengths = [2 * slots * job_size, 2 * slots * 5120 * 2,
                   2 * slots * 2304 * 4, slots * 2304 * 2, slots * 5120 * 4]
        offsets = np.cumsum([0] + lengths)
        assert all(offsets % 16 == 0)
        assert offsets[-1] <= 64 << 20
        # Each token/column is owned by one final warp. Slot order is unchanged.
        for token in range(m):
            assert [slot // 6 for slot in range(token * 6, (token + 1) * 6)] == [token] * 6
        print(f"PASS workspace model m={m}: {offsets[-1]} bytes, aligned/disjoint, within 64 MiB")
    print("PASS declared shared array bytes per template: K3 GEMV 2052, K6 GEMV 10240; activation 1280; output 512 (not ptxas measurements)")


def mixed_indexing():
    for bits in range(1, 7):
        words = 8 * bits
        generic = bits in (1, 5, 6)
        loads_per_tile = (words + 31) // 32
        loads = 2 * loads_per_tile if generic else (1 if bits == 2 else 2)
        stride = 32 if generic or bits in (2, 4) else 24
        tile_stride = loads_per_tile * 32 if generic else words
        staged = {}
        addresses = []
        for load in range(loads):
            for lane in range(32):
                if generic:
                    tile = load // loads_per_tile
                    word = (load % loads_per_tile) * 32 + lane
                    address = tile * words + word if word < words else None
                else:
                    address = load * stride + lane if lane < stride else None
                if address is not None:
                    addresses.append(address)
                    staged[load * stride + lane] = address
        assert sorted(addresses) == list(range(2 * words))
        if generic:
            for tile in range(2):
                for word in range(words):
                    assert staged[tile * tile_stride + word] == tile * words + word
        # Slice and output group strides cannot overlap or escape either matrix.
        for k, n in ((5120, 2304), (2304, 5120)):
            last = ((k // 16 - 1) * (n // 16) + (n // 16 - 2)) * words
            assert last + max(addresses) == (k // 16) * (n // 16) * words - 1
        print(f"PASS mixed K={bits}: {2 * words} words per warp, exact load coverage, padded stage bounds")
    dispatch = (VENDOR / 'quant/exl3_gemv.cu').read_text()
    for bits in range(1, 7):
        assert f'exl3_gemv_kernel<{bits}, true, 2, 0, 0, false>' in dispatch
    pipe = PIPELINE.read_text()
    assert 'H, F, p.tile_w / 16}' in pipe and 'F, H, e.w2.tile_w / 16}' in pipe
    assert 'p.tile_w % 16 == 0 && p.tile_w / 16 >= 1 && p.tile_w / 16 <= 6' in pipe


def scale_rounding():
    # Exhaust every finite nonnegative BF16 amax, including zero/subnormals;
    # compare the bit-ceiling implementation with the reference frexp rule.
    values = (np.arange(0x7f80, dtype=np.uint32) << 16).view(np.float32)
    a = np.maximum(values, np.float32(1e-4)) * np.float32(1.0 / 448.0)
    bits = a.view(np.uint32)
    got = ((bits + np.uint32(0x007fffff)) & np.uint32(0x7f800000)).view(np.float32)
    mantissa, exponent = np.frexp(a)
    want = np.ldexp(np.ones_like(a), exponent - (mantissa == np.float32(0.5)))
    assert np.array_equal(got, want)
    print(f"PASS scale rounding: {len(values)} finite nonnegative BF16 maxima match reference frexp/ldexp")


def host_layout():
    # Compile only the actual plain C++ workspace/descriptor definitions.
    # uint16_t stands in for half STORAGE; this does not compile CUDA code.
    job = (VENDOR / "quant/exl3_gemv.cuh").read_text()
    job = job[job.index("namespace strata_exl3 {"):job.index("// K10:")]
    workspace = PIPELINE.read_text()
    workspace = workspace[workspace.index("namespace strata::ds41::kernels::k10 {"):
                          workspace.index("__device__ inline void check_proj")]
    source = """
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <vector>
using half = uint16_t;
""" + job + "}\n" + workspace + "}\n" + r"""
int main() {
    using namespace strata::ds41::kernels::k10;
    static_assert(HAD_SCALE > 0 && HAD_SCALE < 1);
    static_assert(sizeof(Job) == 40);
    static_assert(Workspace::bytes(6) == 384480);
    static_assert(Workspace::bytes(24) == 1537920);
    static_assert(Workspace::bytes(48) == 3075840);
    for (int slots : {6, 24, 48}) {
        std::vector<std::max_align_t> buf((Workspace::bytes(slots) + 15) / sizeof(std::max_align_t) + 1);
        Workspace ws(buf.data(), slots);
        auto off = [&](const void* p) { return size_t(static_cast<const char*>(p) - reinterpret_cast<const char*>(buf.data())); };
        assert(off(ws.jobs) == 0);
        assert(off(ws.input) == 2 * slots * sizeof(Job));
        assert(off(ws.gu) == off(ws.input) + 2 * slots * H * sizeof(half));
        assert(off(ws.down_input) == off(ws.gu) + 2 * slots * F * sizeof(float));
        assert(off(ws.down) == off(ws.down_input) + slots * F * sizeof(half));
        assert(off(ws.down + slots * H) == Workspace::bytes(slots));
        assert(off(ws.input) % 16 == 0 && off(ws.gu) % 16 == 0 &&
               off(ws.down_input) % 16 == 0 && off(ws.down) % 16 == 0);
    }
}
"""
    with tempfile.TemporaryDirectory(prefix="k10-layout-") as directory:
        tmp = Path(directory)
        (tmp / "layout.cpp").write_text(source)
        subprocess.run([os.environ.get("CXX", "clang++"), "-std=c++17", "-Wall", "-Wextra", "-Werror",
                        str(tmp / "layout.cpp"), "-o", str(tmp / "layout")], check=True)
        subprocess.run([str(tmp / "layout")], check=True)
    print("PASS native C++17 workspace/descriptor layout (uint16 half storage substitute; no CUDA code)")


def main():
    provenance()
    prefetch_schedule()
    include_and_scope_checks()
    indexing()
    mixed_indexing()
    scale_rounding()
    host_layout()
    print("HOST CHECKS PASSED (no CUDA compilation, GPU execution, golden parity or timing)")


if __name__ == "__main__":
    main()
