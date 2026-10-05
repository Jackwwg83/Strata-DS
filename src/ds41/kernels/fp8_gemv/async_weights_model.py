#!/usr/bin/env python3
"""CPU-only ownership and happens-before model; no CUDA or third-party modules.

Run: python3 src/ds41/kernels/fp8_gemv/async_weights_model.py
This verifies indexing and the synchronization protocol, not GPU execution.
"""
from pathlib import Path
import random

THREADS, ROWS, TILE_K, VECTORS = 128, 4, 1024, 64


def producer(vector):
    return vector % THREADS


def consumer(vector):
    row, col = divmod(vector, VECTORS)
    return row * 32 + col % 32


def geometry():
    cases = 0
    for n in (1, 2, 3, 4, 5, 31, 32, 33, 63, 65, 4095, 4096, 4097, 262139, 262145):
        grid = min((n + ROWS - 1) // ROWS, 65535)
        outputs = bytearray(n)
        bases = set()
        for block in range(grid):
            for base in range(block * ROWS, n, grid * ROWS):
                for r in range(ROWS):
                    if base + r < n:
                        outputs[base + r] += 1
                if base < 36 or base >= n - 8 or base >= grid * ROWS:
                    bases.add(base)
        assert all(count == 1 for count in outputs), (n, "output ownership")
        for k in (32, 64, 480, 512, 544, 992, 1024, 1056, 1280, 2048, 2304, 3104, 5120, 6144, 8192):
            for base in bases:
                seen = [bytearray(k) for _ in range(min(ROWS, n - base))]
                for start in range(0, k, TILE_K):
                    writes = bytearray(ROWS * VECTORS)
                    for tid in range(THREADS):
                        for vector in range(tid, ROWS * VECTORS, THREADS):
                            assert producer(vector) == tid
                            writes[vector] += 1
                            r, c = divmod(vector, VECTORS)
                            row, col = base + r, start + 16 * c
                            valid = row < n and col < k
                            if valid:
                                assert col + 15 < k
                                assert 0 <= row * k + col <= n * k - 16
                                assert (row * k + col) % 16 == 0
                            # Invalid cp.async uses the allocation base and a
                            # zero source size, never an out-of-range pointer.
                            src, size = (row * k + col, 16) if valid else (0, 0)
                            assert 0 <= src < n * k and size in (0, 16)
                    assert all(count == 1 for count in writes)
                    for warp in range(ROWS):
                        for lane in range(32):
                            tid = 32 * warp + lane
                            for v in range(2):
                                c = lane + v * 32
                                vector = warp * VECTORS + c
                                assert consumer(vector) == tid
                                col, row = start + c * 16, base + warp
                                if row < n and col < k:
                                    scale = (row // 32) * (k // 32) + col // 32
                                    assert 0 <= scale < ((n + 31) // 32) * (k // 32)
                                    assert col // 32 == (col + 15) // 32
                                    for j in range(16):
                                        seen[warp][col + j] += 1
                                        for m in range(1, 9):
                                            assert (m - 1) * k + col + j < m * k
                assert all(all(count == 1 for count in row) for row in seen)
                cases += 1
    return cases


def protocol(tiles, row_iterations):
    """Events follow each CUDA thread's program order plus the CTA barriers.

    A completion is independently delayed from issue; wait depends on every
    copy by that thread. A CTA barrier has all 128 arrival events as parents.
    Random topological orders therefore include producer/consumer warp skew.
    """
    dependencies, actions = {}, {}
    previous = [None] * THREADS
    serial = 0

    def event(parents, action=None):
        nonlocal serial
        serial += 1
        dependencies[serial] = set(p for p in parents if p is not None)
        if action:
            actions[serial] = action
        return serial

    def stage(epoch, tile):
        completions = [[] for _ in range(THREADS)]
        for tid in range(THREADS):
            for vector in range(tid, ROWS * VECTORS, THREADS):
                issue = event([previous[tid]], ("issue", epoch, tile, vector))
                previous[tid] = issue
                completions[tid].append(event([issue], ("complete", epoch, tile, vector)))
        return completions

    def publish(completions):
        waits = [event([previous[t], *completions[t]]) for t in range(THREADS)]
        barrier = event(waits)
        previous[:] = [barrier] * THREADS

    for epoch in range(row_iterations):
        publish(stage(epoch, 0))
        for tile in range(tiles):
            pending = stage(epoch, tile + 1) if tile + 1 < tiles else [[] for _ in range(THREADS)]
            for vector in range(ROWS * VECTORS):
                tid = consumer(vector)
                previous[tid] = event([previous[tid]], ("read", epoch, tile, vector))
            publish(pending)

    # Every legal schedule must order completion -> publish -> read, and
    # read -> release -> the next issue to the same buffer. Check these
    # structural edges without relying on any sampled interleaving.
    action_ids = {value: key for key, value in actions.items()}

    def before(first, last):
        todo, seen = [last], set()
        while todo:
            node = todo.pop()
            if node == first:
                return True
            if node not in seen:
                seen.add(node)
                todo.extend(dependencies[node])
        return False

    prior_reads = {}
    for epoch in range(row_iterations):
        for tile in range(tiles):
            for vector in range(ROWS * VECTORS):
                complete = action_ids[("complete", epoch, tile, vector)]
                read = action_ids[("read", epoch, tile, vector)]
                issue = action_ids[("issue", epoch, tile, vector)]
                assert before(complete, read), "read not ordered after producer completion"
                slot = (tile & 1, vector)
                if slot in prior_reads:
                    assert before(prior_reads[slot], issue), "buffer reuse not ordered after old read"
                prior_reads[slot] = read

    followers = {node: [] for node in dependencies}
    for node, parents in dependencies.items():
        for parent in parents:
            followers[parent].append(node)
    for seed in range(8):
        rng = random.Random(seed)
        degrees = {node: len(parents) for node, parents in dependencies.items()}
        ready = [node for node, degree in degrees.items() if degree == 0]
        memory, unread, in_flight = {}, set(), set()
        visited = 0
        while ready:
            index = rng.randrange(len(ready))
            node = ready[index]
            ready[index] = ready[-1]
            ready.pop()
            visited += 1
            if node in actions:
                kind, epoch, tile, vector = actions[node]
                slot, value = (tile & 1, vector), (epoch, tile)
                if kind == "issue":
                    assert slot not in unread and slot not in in_flight
                    in_flight.add(slot)
                elif kind == "complete":
                    in_flight.remove(slot)
                    memory[slot] = value
                    unread.add(slot)
                else:
                    assert slot not in in_flight and memory[slot] == value
                    unread.remove(slot)
            for after in followers[node]:
                degrees[after] -= 1
                if degrees[after] == 0:
                    ready.append(after)
        assert visited == len(dependencies)
        assert not unread and not in_flight
    return len(dependencies)


def source_contract():
    source = Path(__file__).with_name("async_weights.cuh").read_text()
    assert "ASYNC_ROWS = 4" in source and "ASYNC_K = 1024" in source
    assert "cp.async.cg.shared.global" in source and "cp.async.wait_group 0" in source
    assert source.count("__syncthreads();") == 2
    assert "float acc[M]" in source and "fmaf(" in source
    assert ROWS * TILE_K * 2 == 8192 < 99 * 1024


if __name__ == "__main__":
    source_contract()
    cases = geometry()
    events = sum(protocol(tiles, 2) for tiles in (1, 2, 3, 4))
    print(f"PASS: {cases} N/K tail geometries, unique vector/byte/output ownership")
    print(f"PASS: {events} pipeline events, full happens-before proofs and 32 random schedules")
    print("PASS: zero-fill bounds, grid-stride reuse, scale blocks, m=1..8 indexing, 8192-byte shared tile")
