#!/usr/bin/env python3
"""CPU/source checks for K3-25. No CUDA runtime or GPU claims.

Run at the repository root. The exact K3-11 commit must be available to git.
"""
import hashlib
import pathlib
import random
import re
import subprocess

BASE = "1a4598be4361c03ed818ab80f777f20c73fd4c6d"
PATH = "src/ds41/kernels/k3_sparse_attn.cu"
ROWS, DIM, SLICE, THREADS = 128, 512, 64, 128
WINDOW_BASE, COMP_BASE = 0x100000000, 0x200000000


def old_packet(j, dimension, d):
    if j < 0:
        return WINDOW_BASE, 0
    base, row = (WINDOW_BASE, j) if j < 128 else (COMP_BASE, j - 128)
    return base + (row * DIM + dimension + d) * 2, 16


def prepare_row(j):
    if j < 0:
        return WINDOW_BASE, False
    base, row = (WINDOW_BASE, j) if j < 128 else (COMP_BASE, j - 128)
    return base + row * DIM * 2, True


def new_packet(source, valid, dimension, d):
    return (source + (dimension + d) * 2 if valid else source), (16 if valid else 0)


def check_addresses():
    # Large positive rows are legal if comp has the corresponding extent.
    # n_comp is absent from the interface: neither implementation can validate
    # positive out-of-allocation indices. We verify the entire int32 address
    # arithmetic range symbolically, without pretending to allocate 2 TiB.
    cases = [-(2**31), -1024, -129, -128, -2, -1, 0, 1, 126, 127,
             128, 129, 4223, 2**21 + 128, 2**22 + 128, 2**31 - 1]
    rng = random.Random(2503)
    cases += [rng.randrange(-(2**31), 2**31) for _ in range(4096)]
    checks = 0
    for j in cases:
        source, valid = prepare_row(j)
        assert valid == (j >= 0)
        if not valid:
            assert source == WINDOW_BASE
        for dimension in range(0, DIM, SLICE):
            for d in range(0, SLICE, 8):
                address, size = new_packet(source, valid, dimension, d)
                assert (address, size) == old_packet(j, dimension, d)
                assert address % 16 == 0
                assert 0 <= address < 2**64
                if valid:
                    # Every nonempty packet remains inside its selected row.
                    assert source <= address and address + size <= source + DIM * 2
                else:
                    # No dimension arithmetic on an invalid placeholder.
                    assert address == WINDOW_BASE and size == 0
                checks += 1
    assert prepare_row(2**22 + 128)[0] - COMP_BASE == 2**32
    assert prepare_row(2**31 - 1)[0] - COMP_BASE == (2**31 - 129) * DIM * 2
    print(f"PASS {checks} cached/original packet addresses, negative indices, 64-bit offsets")


def check_tails():
    rng = random.Random(250325)
    pool = [-2**31, -7, -1, 0, 17, 127, 128, 257, 4223, 2**31 - 1]
    checked = 0
    for n_idx in range(1025):
        indices = [rng.choice(pool) for _ in range(n_idx)]
        reads = []
        for first in range(0, n_idx, ROWS):
            for tid in range(THREADS):
                position = first + tid
                j = indices[position] if position < n_idx else -1
                if position < n_idx:
                    reads.append(position)
                source, valid = prepare_row(j)
                assert valid == (j >= 0)
                if position >= n_idx:
                    assert (source, valid) == (WINDOW_BASE, False)
                checked += 1
        assert reads == list(range(n_idx))
    # Query base offsets are unchanged and disjoint at every legal shape.
    for m in range(1, 9):
        for n_idx in (0, 1, 127, 128, 129, 300, 640, 1023, 1024):
            reads = [t * n_idx + p for t in range(m) for p in range(n_idx)]
            assert reads == list(range(m * n_idx))
    print(f"PASS all 1025 n_idx values and {checked} active/tail row publications")


def check_ownership():
    packets = []
    for tid in range(THREADS):
        for i in range(tid, ROWS * SLICE // 8, THREADS):
            r, d = i // (SLICE // 8), (i % (SLICE // 8)) * 8
            offset = r * SLICE + (d ^ ((r & 7) * 8))
            assert offset % 8 == 0
            assert 0 <= offset and offset + 8 <= ROWS * SLICE
            packets.append((r, d, offset))
    assert len({(r, d) for r, d, _ in packets}) == ROWS * SLICE // 8
    assert sorted(off for _, _, off in packets) == list(range(0, ROWS * SLICE, 8))
    for output_group in range(4):
        output_dim = output_group * 128
        first_dim = (output_dim + 128) & 511
        dimensions = [(first_dim + s * SLICE) & 511 for s in range(8)]
        assert sorted(dimensions) == list(range(0, DIM, SLICE))
        assert dimensions[-2:] == [output_dim, output_dim + SLICE]
        logical = {(r, dimension + d) for dimension in dimensions for r, d, _ in packets}
        assert len(logical) == ROWS * DIM // 8
    for m in range(1, 9):
        owners = set()
        for query in range(m):
            for head_group in range(8):
                for output_group in range(4):
                    for tid in range(THREADS):
                        lane, warp = tid & 31, tid >> 5
                        head0 = head_group * 8 + (lane & 3) * 2
                        for v in range(2):
                            d = output_group * 128 + (warp * 2 + v) * 16 + (lane >> 2)
                            for h, dim in ((head0, d), (head0 + 1, d),
                                           (head0, d + 8), (head0 + 1, d + 8)):
                                address = (query * 64 + h) * DIM + dim
                                assert address not in owners
                                owners.add(address)
        assert owners == set(range(m * 64 * DIM))
    print("PASS complete unique gather packets, cyclic slice residency and output ownership for m1..8")


def tokens(source):
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
    return re.findall(r'\w+|[^\s]', source)


def check_source():
    baseline = subprocess.check_output(["git", "show", f"{BASE}:{PATH}"], text=True)
    candidate = pathlib.Path(PATH).read_text()
    # Restore only the intentional metadata/address-cache edits, then require
    # token-for-token equality. Everything else, including all FP32/BF16 math,
    # barriers, MMA instruction text/order and the stream dispatch, is fixed.
    restored = candidate.replace("    const bf16* sources[kRows];\n    int valid[kRows];", "    int indices[kRows];")
    restored = restored.replace("sizeof(TileStorage) == 46816", "sizeof(TileStorage) == 45792")
    restored = restored.replace("                                             int dimension)",
        "                                             const bf16* window, const bf16* comp,\n                                             int dimension)")
    cached = """        const int valid = tile.valid[r];
        const bf16* source = tile.sources[r];
        if (valid) source += dimension + d;"""
    original = """        const int j = tile.indices[r];
        const bf16* source = window;
        if (j >= 0) {
            source = j < kWindow ? window + static_cast<size_t>(j) * kDim
                                 : comp + static_cast<size_t>(j - kWindow) * kDim;
            source += dimension + d;
        }"""
    assert cached in restored
    restored = restored.replace(cached, original).replace('"r"(valid ? 16 : 0)', '"r"(j >= 0 ? 16 : 0)')
    start = restored.index("        const int j = position < n_idx ? indices[position] : -1;")
    end = restored.index("        float score[kFragments][4] = {};", start)
    publish = restored[start:end]
    assert publish.count("__syncthreads()") == 1
    assert "tile.sources[threadIdx.x] = source;" in publish
    assert "tile.valid[threadIdx.x] = j >= 0;" in publish
    restored = restored[:start] + """        tile.indices[threadIdx.x] = position < n_idx ? indices[position] : -1;
        __syncthreads();
""" + restored[end:]
    restored = restored.replace("gather_slice(tile, 0, first_dimension)", "gather_slice(tile, 0, window, comp, first_dimension)")
    restored = restored.replace("gather_slice(tile, buffer ^ 1, next_dimension)", "gather_slice(tile, buffer ^ 1, window, comp, next_dimension)")
    restored = restored.replace("tile.valid[r] ?", "tile.indices[r] >= 0 ?")
    assert tokens(restored) == tokens(baseline), "unexpected arithmetic/schedule/source change"
    # Also constrain the complete metadata initialization instead of deleting
    # unchecked code from that comparison.
    expected_publish = """
        const int j = position < n_idx ? indices[position] : -1;
        const bf16* source = window;
        if (j >= 0) {
            source = j < kWindow ? window + static_cast<size_t>(j) * kDim
                                 : comp + static_cast<size_t>(j - kWindow) * kDim;
        }
        tile.sources[threadIdx.x] = source;
        tile.valid[threadIdx.x] = j >= 0;
        __syncthreads();
    """
    assert tokens(publish) == tokens(expected_publish)
    print("PASS exact K3-11 arithmetic, launch geometry, barriers and stream source comparison")
    print("candidate_sha256=" + hashlib.sha256(candidate.encode()).hexdigest())


if __name__ == "__main__":
    check_addresses()
    check_tails()
    check_ownership()
    check_source()
    print("PASS CPU/source checks only; GPU numerical, graph and timing tests remain required")
