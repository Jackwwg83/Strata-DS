#!/usr/bin/env python3
"""Host checks for paired input Hadamards. No GPU or performance validation.

Source-equivalence checks independently specialize the unchanged vendor helper.
CPU checks compare separate scalar projection transforms with paired vectorized
transforms, including FP16 pre-scale and final rounding, unrelated W1/W3 signs,
finite-half bit patterns, masks, chunk addressing, and stale job descriptors.
"""
from pathlib import Path
import hashlib
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PIPELINE = ROOT / "src/ds41/kernels/k10/pipeline.cuh"
CU = ROOT / "src/ds41/kernels/k10_exl3_moe.cu"
VENDOR = ROOT / "third_party/exllamav3_gpu"
CONTROL = "d42d168c78e755ef1ed85b971795e48ef63666ac"
CONTROL_HASHES = {'third_party/exllamav3_gpu/LICENSE': '27a32b6263fcd96c79d3beeecf221c4366780bdf15ad51986f48650bd7369bff', 'third_party/exllamav3_gpu/UPSTREAM.sha256': '743de967386f5dceacfbcebaafb878e404a59cc18c4d321390b6bc1cdddfb336', 'third_party/exllamav3_gpu/arch.cuh': '763a7389e87cca199e9a2eb1d329e12f2acaee423f7a301e6575b694f8f13090', 'third_party/exllamav3_gpu/compat.cuh': 'dd6038fa6eb8b28df56b184e358e5633421af76eeb40bad4abc522ce6ba54f57', 'third_party/exllamav3_gpu/ptx.cuh': '266eabb4e1e5cded91dcc5e7f68293cefd68df84e9deb079dbfd6b04884bab2f', 'third_party/exllamav3_gpu/quant/codebook.cuh': '0e3c63b323f8d3cc15c6a8f2e2b3816efafd71e149a99997b30b2f806375138a', 'third_party/exllamav3_gpu/quant/exl3_dq.cuh': '4e48a37af4811e8e0f7e00e86c29c9c044fa0b6a8f6a685855a033684e4d6552', 'third_party/exllamav3_gpu/quant/exl3_gemv.cu': '295ea33feff62f29ddbd5e28dfa2c70d7c84afac624449c371d9ea17308d07bd', 'third_party/exllamav3_gpu/quant/exl3_gemv.cuh': 'e6b7851f8a1263732582e3e0c4ab44bb62c516014091d5a908af61315f342438', 'third_party/exllamav3_gpu/quant/exl3_gemv_kernel.cuh': '751c75087dfe1433fa57728d6b2882f3ea357543a0026ea6a376a0a5a9e4dfc9', 'third_party/exllamav3_gpu/quant/exl3_kernel_map.cuh': '68afe01ded6198adf74dd124f0cd151fe536f918c0067ed77844e34e7688b20e', 'third_party/exllamav3_gpu/quant/hadamard_inner.cuh': '8d8e437aced88735e919563301ffac0e4a2aac28cc542ed3f738e1216ea0c36b', 'third_party/exllamav3_gpu/strata.patch': '983177c4c99500b25b517c37da09db2e1434ad54b445e9e0e23c7fe94851135a', 'third_party/exllamav3_gpu/util.cuh': '1907cea115260db7c3b0de540e375e7733b7ffc9019bec190c1d92abea1d964e', 'third_party/exllamav3_gpu/util.h': 'ba89ac6793bf31cfe123d13baa2e34531a18a24bc3b2ff4add8cf4856b75a1c8'}
PREFIX_HASH = "f158e9281d80b7ae360cfdbf4cababa42f2194d1f8efc4622ea46136be1e4299"
SUFFIX_HASH = "394f5851f680e60fe68029264814bd017810d1da5cf64429baafa55b2ca2cc9a"
ENTRY_HASH = "6126c759d893a59c1956fd31c7cedbcfcc77b9250dbedb4a35aa28afb305afb2"


def digest(s):
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def body(s, function):
    start = s.index("{", s.index(function))
    depth = 0
    for i in range(start, len(s)):
        depth += (s[i] == "{") - (s[i] == "}")
        if not depth:
            return s[start + 1:i]
    raise AssertionError("unterminated function")


def tokens(s):
    s = re.sub(r"//[^\n]*|/\*.*?\*/", "", s, flags=re.S)
    return re.findall(r"\w+|[^\s]", s)


def source_equivalence(s):
    original = body((VENDOR / "quant/hadamard_inner.cuh").read_text(), "void had_hf_r_128_inner")
    # Exactly specialize <true,false>, then hoist only the half4 input load.
    original = original.replace("half4 v = ((half4*) input_ptr)[t];", "")
    original = original.replace("if constexpr (pre_scale)", "")
    post = "if constexpr (post_scale)"
    block = body(original, post)
    original = original.replace(post + "\n    {" + block + "}", "")
    assert tokens(body(s, "void input_had_preloaded")) == tokens(original)
    assert digest(s[:s.index("// This is had_hf_r_128_inner")]) == PREFIX_HASH
    assert digest(s[s.index("__device__ inline float round_pow2"):]) == SUFFIX_HASH
    kernel = body(s, "void input_had(")
    for statement in (
        "const int slot = blockIdx.x;", "const int job = 2 * slot;",
        "const int id = sel[slot];", "const int off = blockIdx.y * 128;",
        "if (id < 0) { if (off == 0 && threadIdx.x == 0) { jobs[job] = Job{}; jobs[job + 1] = Job{}; } return; }",
        "const Exl3Proj gate = experts[id].w1;", "const Exl3Proj up = experts[id].w3;",
        "check_proj(gate, H, F);", "check_proj(up, H, F);",
        "half* a_gate = input + size_t(job) * H;", "half* a_up = input + size_t(job + 1) * H;",
        "if (off == 0 && threadIdx.x == 0) { jobs[job] = Job{a_gate, gate.trellis, gu + size_t(job) * F, H, F}; jobs[job + 1] = Job{a_up, up.trellis, gu + size_t(job + 1) * F, H, F}; }",
        "const half4 v = reinterpret_cast<const half4*>(x + size_t(slot / topk) * H + off)[threadIdx.x];",
        "input_had_preloaded(v, a_gate + off, gate.suh, HAD_SCALE);",
        "input_had_preloaded(v, a_up + off, up.suh, HAD_SCALE);",
    ):
        a, b = tokens(statement), tokens(kernel)
        assert any(b[i:i + len(a)] == a for i in range(len(b) - len(a) + 1)), statement
    assert kernel.count("reinterpret_cast<const half4*>(x") == 1
    assert kernel.count("input_had_preloaded(") == 2
    return True


def source_checks():
    for path, want in CONTROL_HASHES.items():
        assert digest((ROOT / path).read_bytes()) == want, path
    entry = CU.read_text()
    assert entry.count("dim3(slots, k10::H / 128)") == 1
    entry = entry.replace("dim3(slots, k10::H / 128)", "dim3(2 * slots, k10::H / 128)")
    assert digest(entry) == ENTRY_HASH
    s = PIPELINE.read_text()
    source_equivalence(s)
    # A checker that misses an incorrect shared scale or changed rounding is not useful.
    for before, after in [
        ("a_up + off, up.suh", "a_up + off, gate.suh"),
        ("jobs[job + 1] = Job{}", "jobs[job] = Job{}"),
        ("v.x = __hmul2(v.x, scales.x);", "v.x = __hadd2(v.x, scales.x);"),
        ("float d0 = v0 - v1;", "float d0 = v1 - v0;"),
    ]:
        assert before in s
        try:
            source_equivalence(s.replace(before, after, 1))
        except AssertionError:
            continue
        raise AssertionError("source mutation escaped: " + before)
    print("PASS source proof: upstream <true,false> arithmetic tokens exact after load hoist; four mutations rejected")
    print("PASS pinned K10-01 vendor/GEMV, workspace, downstream pipeline, and entry except paired grid")


def single_reference(x, scales):
    # Lane-by-lane independent model of the original helper; every elementary
    # sum rounds to float32, and both half conversion boundaries are explicit.
    out = np.empty((32, 4), dtype=np.float32)
    for lane in range(32):
        a, b, c, d = (x[lane * 4:lane * 4 + 4].astype(np.float32) *
                       scales[lane * 4:lane * 4 + 4].astype(np.float32)).astype(np.float16).astype(np.float32)
        p, q = np.float32(a + b), np.float32(a - b)
        r, t = np.float32(c + d), np.float32(c - d)
        out[lane] = [np.float32(p + r), np.float32(q + t),
                     np.float32(p - r), np.float32(q - t)]
    for distance in (1, 2, 4, 8, 16):
        prev = out.copy()
        for lane in range(32):
            for j in range(4):
                v = prev[lane, j]
                if lane & distance:
                    v = np.array(v).view(np.uint32) ^ np.uint32(0x80000000)
                    v = v.view(np.float32)
                out[lane, j] = np.float32(v + prev[lane ^ distance, j])
    return (out.ravel() * np.float32(0.088388347648)).astype(np.float16)


def pair_model(x, scales):
    # Pair dimension is kept separate throughout. Loading x once only changes
    # storage reuse: neither the half pre-multiply nor butterfly can be shared.
    v = (x.astype(np.float32)[None, :] * scales.astype(np.float32)).astype(np.float16).astype(np.float32)
    v = v.reshape(2, 32, 4)
    s0, d0 = v[:, :, 0] + v[:, :, 1], v[:, :, 0] - v[:, :, 1]
    s1, d1 = v[:, :, 2] + v[:, :, 3], v[:, :, 2] - v[:, :, 3]
    v = np.stack((s0 + s1, d0 + d1, s0 - s1, d0 - d1), axis=-1)
    lanes = np.arange(32)
    for distance in (1, 2, 4, 8, 16):
        sign = ((lanes & distance) != 0).astype(np.uint32) << 31
        neg = (v.view(np.uint32) ^ sign[None, :, None]).view(np.float32)
        v = neg + v[:, lanes ^ distance, :]
    return (v.reshape(2, 128) * np.float32(0.088388347648)).astype(np.float16)


def rounding_checks():
    rng = np.random.default_rng(520051)
    bits = np.arange(65536, dtype=np.uint16)
    values = bits[(bits & 0x7c00) != 0x7c00].view(np.float16)
    count = 0
    unrelated = 0
    # Exhaust finite half INPUT encodings including both signed zeros and all
    # subnormals; do not claim all possible 128-vectors or NaN payloads exhausted.
    batches = list(values.reshape(-1, 128))
    batches += [rng.normal(0, 3, 128).astype(np.float16) for _ in range(256)]
    batches += [np.eye(128, dtype=np.float16)[i] for i in range(128)]
    with np.errstate(over="ignore", invalid="ignore"):
        for i, x in enumerate(batches):
            scales = rng.choice(np.array([-1, 1], np.float16), (2, 128))
            if i >= len(values) // 128:
                scales *= rng.choice(np.array([0, 2**-14, 0.33325, 0.9995, 1, 1.001, 2, 16], np.float16), (2, 128))
            want = np.stack([single_reference(x, scales[p]) for p in range(2)])
            got = pair_model(x, scales)
            assert np.array_equal(got.view(np.uint16), want.view(np.uint16)), i
            unrelated += not np.array_equal(got[0].view(np.uint16), got[1].view(np.uint16))
            count += 1
    assert unrelated > count // 2
    # Negative control: postponing FP16 multiplication rounding changes output.
    x = rng.normal(size=128).astype(np.float16)
    scales = rng.uniform(-2, 2, (2, 128)).astype(np.float16)
    rounded = (x.astype(np.float32)[None, :] * scales.astype(np.float32)).astype(np.float16).astype(np.float32)
    unrounded = x.astype(np.float32)[None, :] * scales.astype(np.float32)
    assert np.any(rounded.view(np.uint32) != unrounded.view(np.uint32))
    print(f"PASS CPU rounding: {count} vector pairs, {len(values)} finite FP16 input patterns, independent scales and half boundaries")


def mapping_checks():
    columns = (np.arange(40)[:, None, None] * 128 +
               np.arange(32)[None, :, None] * 4 + np.arange(4))
    assert np.array_equal(columns.ravel(), np.arange(5120))
    assert np.array_equal((np.arange(40)[:, None] * 32 + np.arange(32)).ravel(), np.arange(1280))
    cases = 0
    for m in range(1, 9):
        for topk in (1, 2, 6, 7, 31, 257, 32767 // m):
            slots = m * topk
            slot = np.arange(slots)
            jobs = (slot[:, None] * 2 + np.arange(2)).ravel()
            assert np.array_equal(jobs, np.arange(2 * slots))
            assert np.array_equal(np.repeat(slot // topk, 2), (jobs // 2) // topk)
            # Explicitly model descriptors poisoned by an earlier invocation.
            for mask in (slot < 0, slot >= 0, slot % 2 == 0, slot % 3 == 0):
                old = np.full((slots * 2, 5), -91827, np.int64)
                paired = old.copy()
                separate = old.copy()
                for projection in range(2):
                    job = 2 * slot + projection
                    descriptor = np.column_stack((job * 5120, slot * 3 + projection + 1, job * 2304,
                                                  np.full(slots, 5120), np.full(slots, 2304)))
                    paired[job] = np.where(mask[:, None], descriptor, 0)
                job = np.arange(2 * slots)
                parent = job // 2
                descriptor = np.column_stack((job * 5120, parent * 3 + job % 2 + 1, job * 2304,
                                              np.full(2 * slots, 5120), np.full(2 * slots, 2304)))
                separate[:] = np.where(mask[parent, None], descriptor, 0)
                assert np.array_equal(paired, separate)
                assert not np.any(paired == -91827)
                cases += 1
    print(f"PASS mapping: {cases} mask/shape cases through legal grid bound, 40 chunks, full scales, both stale jobs overwritten")
    for m in (1, 4, 8):
        slots = 6 * m
        print(f"COST m={m}, topk=6, all active: CTAs {80*slots}->{40*slots}; logical x bytes {2*slots*5120*2}->{slots*5120*2}; same transforms/stores")


if __name__ == "__main__":
    source_checks()
    rounding_checks()
    mapping_checks()
    print("PAIRED INPUT HOST CHECKS PASSED; GPU golden parity, graph replay and timing remain untested")
