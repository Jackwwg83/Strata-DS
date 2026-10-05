"""Tests for tools/ds41/make_profile.py: ranking order, fill, and the STRP file layout."""
import os
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_profile as MP  # noqa: E402


def routes_with(pairs_per_token):
    """[T, 40, 6] routes: layer 0 slots from the given lists, every other layer routes experts 0..5."""
    r = np.tile(np.arange(6), (len(pairs_per_token), MP.N_LAYER, 1))
    for t, ids in enumerate(pairs_per_token):
        r[t, 0] = ids
    return r


def test_rank_most_frequent_first_then_fill():
    r = routes_with([[10, 11, 12, 13, 14, 15], [10, 11, 20, 21, 22, 23], [10, 30, 31, 32, 33, 34]])
    ranked = MP.rank(MP.count_pairs(r))
    assert len(ranked) == MP.N_LAYER * MP.N_EXPERT
    assert len(set(ranked)) == len(ranked)
    # every layer >= 1 routed experts 0..5 three times: count 3, ties by (layer, expert); (0, 10) also count 3
    assert ranked[0] == (0, 10)
    assert ranked[1] == (1, 0)
    # (0, 11) has count 2: after every count-3 pair (1 + 39 * 6)
    assert ranked[1 + 39 * 6] == (0, 11)
    # the fill starts after the traced pairs, interleaved across layers: expert e of every layer, then e + 1
    n_traced = len(MP.count_pairs(r))
    assert ranked[n_traced] == (0, 0)
    assert ranked.index((1, 6)) == ranked.index((0, 6)) + 1


def test_file_round_trip_and_layout():
    r = routes_with([[1, 2, 3, 4, 5, 6]] * 4)
    ranked = MP.rank(MP.count_pairs(r))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "p.bin")
        MP.write_profile(p, ranked)
        assert MP.read_profile(p) == ranked
        blob = open(p, "rb").read()
        n = MP.N_LAYER * MP.N_EXPERT
        assert len(blob) == 24 + 4 * n + 4 * n
        assert struct.unpack_from("<5I", blob, 4) == (1, MP.N_LAYER, MP.N_EXPERT, n, n)
        table = np.frombuffer(blob, "<i4", n, 24 + 4 * n).reshape(MP.N_LAYER, MP.N_EXPERT)
        for slot in (0, 7, n - 1):
            l, e = ranked[slot]
            assert table[l, e] == slot


def test_trace_shape_checked():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "bad.npz")
        np.savez(p, routes=np.zeros((3, 39, 6), np.int64))
        try:
            MP.trace_routes(p)
        except SystemExit:
            return
        raise AssertionError("a wrong shape must be rejected")
