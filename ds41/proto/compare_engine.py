"""Compare a ds41_generate --dump file with the oracle from dump_oracle.py.

Usage: python compare_engine.py --oracle oracle/zh_moe.npz --dump steps.bin [--out result.json]

The engine must have been run with --force-ids on the oracle's token sequence, so both saw the same tokens.
Reports, per fed token: per-layer relative L2 error of the hc stream (first H tokens), routing agreement
(same expert set per layer), and whether the greedy next token agrees.
"""
import argparse
import json

import numpy as np

L, HC, D, K = 40, 4, 5120, 6
REC = 8 + L * HC * D * 2 + L * K * 4 * 2 + 8 * 4 * 2


def read_dump(path):
    raw = np.fromfile(path, dtype=np.uint8)
    n = len(raw) // REC
    if n * REC != len(raw):
        raise SystemExit(f"{path}: size {len(raw)} is not a multiple of the record size {REC}")
    steps = []
    for i in range(n):
        r = raw[i * REC:(i + 1) * REC]
        o = 0
        tok, nxt = np.frombuffer(r[o:o + 8].tobytes(), np.int32); o += 8
        hid = np.frombuffer(r[o:o + L * HC * D * 2].tobytes(), np.uint16).reshape(L, HC, D); o += L * HC * D * 2
        routes = np.frombuffer(r[o:o + L * K * 4].tobytes(), np.int32).reshape(L, K); o += L * K * 4
        weights = np.frombuffer(r[o:o + L * K * 4].tobytes(), np.float32).reshape(L, K); o += L * K * 4
        top = np.frombuffer(r[o:o + 32].tobytes(), np.int32); o += 32
        topv = np.frombuffer(r[o:o + 32].tobytes(), np.float32)
        hidden = (hid.astype(np.uint32) << 16).view(np.float32)
        steps.append(dict(tok=int(tok), next=int(nxt), hidden=hidden, routes=routes, weights=weights, top=top, topv=topv))
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    z = np.load(a.oracle)
    steps = read_dump(a.dump)
    n = min(len(steps), len(z["ids"]))
    per_layer = np.zeros((min(n, len(z["hidden"])), L))
    route_agree = np.zeros((n, L))
    next_agree = []
    for i in range(n):
        s = steps[i]
        if s["tok"] != int(z["ids"][i]):
            raise SystemExit(f"step {i}: engine fed {s['tok']}, oracle fed {z['ids'][i]}: run with --force-ids")
        if i < len(z["hidden"]):
            ref = z["hidden"][i].astype(np.float32)
            for l in range(L):
                per_layer[i, l] = np.linalg.norm(s["hidden"][l] - ref[l]) / max(np.linalg.norm(ref[l]), 1e-30)
        for l in range(L):
            route_agree[i, l] = set(s["routes"][l].tolist()) == set(z["routes"][i, l].tolist())
        next_agree.append(s["next"] == int(z["logits_top"][i, 0]))
    res = {
        "steps": n,
        "hidden_rel_l2_median_by_layer": [float(x) for x in np.median(per_layer, axis=0)],
        "hidden_rel_l2_max": float(per_layer.max()),
        "first_layer_over_1pct": int(np.argmax(np.median(per_layer, axis=0) > 0.01)) if (np.median(per_layer, axis=0) > 0.01).any() else None,
        "route_agreement": float(route_agree.mean()),
        "route_agreement_layer0": float(route_agree[:, 0].mean()),
        "next_token_agreement": float(np.mean(next_agree)),
    }
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
