#!/usr/bin/env python3
"""Execute actual upstream/return-only helpers and output kernel on a CPU warp
emulator. This is a host equivalence check, not GPU correctness or speed proof.
Run: python3 src/ds41/kernels/k10/check_output_host.py [--root CHECKOUT]
"""
from pathlib import Path
import argparse
import hashlib
import os
import subprocess
import tempfile

BASE_HASHES = {'src/ds41/kernels/k10_exl3_moe.cu': '6126c759d893a59c1956fd31c7cedbcfcc77b9250dbedb4a35aa28afb305afb2', 'third_party/exllamav3_gpu/quant/exl3_gemv_kernel.cuh': '751c75087dfe1433fa57728d6b2882f3ea357543a0026ea6a376a0a5a9e4dfc9', 'third_party/exllamav3_gpu/quant/exl3_gemv.cu': '295ea33feff62f29ddbd5e28dfa2c70d7c84afac624449c371d9ea17308d07bd', 'third_party/exllamav3_gpu/quant/exl3_gemv.cuh': 'e6b7851f8a1263732582e3e0c4ab44bb62c516014091d5a908af61315f342438', 'third_party/exllamav3_gpu/quant/hadamard_inner.cuh': '8d8e437aced88735e919563301ffac0e4a2aac28cc542ed3f738e1216ea0c36b'}
PREFIX_HASH = "fbd9b6d3328f1b93b19c1ee75bd565a75bb731055cce02b391089f3a4ef5114b"


def extract(text, start):
    pos = text.index(start)
    opening = text.index("{", pos)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[pos:end]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    args = parser.parse_args()
    root = args.root
    for rel, digest in BASE_HASHES.items():
        assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == digest, rel
    pipeline = (root / "src/ds41/kernels/k10/pipeline.cuh").read_text()
    prefix = pipeline.split("// Each token/chunk has one owner.")[0].replace('#include "output_hadamard.cuh"\n', '')
    assert hashlib.sha256(prefix.encode()).hexdigest() == PREFIX_HASH, "upstream pipeline changed"
    upstream = (root / "third_party/exllamav3_gpu/quant/hadamard_inner.cuh").read_text()
    original = extract(upstream, "template <bool pre_scale, bool post_scale>\ninline __device__\nvoid had_ff_r_128_inner")
    adapted = extract((root / "src/ds41/kernels/k10/output_hadamard.cuh").read_text(),
                      "template <bool pre_scale, bool post_scale>")
    expected = original.replace('void had_ff_r_128_inner', 'float4 had_ff_r_128_registers').replace(
        '    float* __restrict__ output_ptr,\n', '').replace(
        '    // Store\n    ((float4*) output_ptr)[t] = v;',
        '    // The caller owns the same lane*4 output coordinates.\n    return v;')
    assert adapted == expected, "Hadamard load/arithmetic/scale body differs from upstream"
    kernel = extract(pipeline, "__global__ void output_had_add")
    for c in "xyzw":
        assert f"acc.{c} = __fadd_rn(acc.{c}, v.{c});" in kernel, "FP32 rounding barrier changed"
    assert "__shared__" not in kernel and "__syncwarp" not in kernel
    assert "const bool vector_out = (reinterpret_cast<uintptr_t>(out) & 15u) == 0;" in kernel, "output alignment guard changed"
    assert "acc = *reinterpret_cast<const float4*>(dst);" in kernel, "vector load changed"
    assert "*reinterpret_cast<float4*>(dst) = acc;" in kernel, "vector store changed"
    for m in range(1, 9):
        for topk in range(1, 32767 // m + 1):
            last_slot = (m - 1) * topk + topk - 1
            assert last_slot == m * topk - 1 < 32767
            assert last_slot * 5120 + 5119 < (1 << 31)
    print("PASS slot index bounds: every host-accepted m/topk pair, including maximum slot/tail")
    print("PASS exact-source preservation: K10-01 GEMV/launcher/upstream pipeline; upstream Hadamard arithmetic")
    shuffle = extract(upstream, "__device__ inline void shuffle_had_f2x32")
    harness = (root / "src/ds41/kernels/k10/host_output_harness.cpp").read_text()
    source = harness.replace("// ACTUAL_HELPERS", shuffle + "\n" + original + "\n" + adapted).replace(
        "// ACTUAL_OUTPUT_KERNEL", kernel)
    with tempfile.TemporaryDirectory(prefix="k10-output-host-") as name:
        d = Path(name)
        (d / "check.cpp").write_text(source)
        subprocess.run([os.environ.get("CXX", "g++"), "-std=c++17", "-O2", "-Wall", "-Wextra",
                        "-Werror", "-Wno-unknown-pragmas", "-ffp-contract=off", "-fno-fast-math",
                        str(d / "check.cpp"), "-o", str(d / "check")], check=True)
        subprocess.run([str(d / "check")], check=True)
    print("HOST OUTPUT CHECKS PASSED: CPU emulation only; GPU parity/graph/timing remain pending")


if __name__ == "__main__":
    main()
