"""Compare two dump_oracle.py outputs of the same token sequence (for example the prototype with two
summation orders). This measures how far two equally valid implementations drift apart, the baseline that
an engine-vs-prototype difference is judged against.

Usage: python compare_oracles.py A.npz B.npz
"""
import json
import sys

import numpy as np


def f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32) if bits.dtype == np.uint16 else bits.astype(np.float32)


def main():
    a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
    if not np.array_equal(a["ids"], b["ids"]):
        raise SystemExit("the two runs fed different tokens")
    ha, hb = f32(a["hidden"]), f32(b["hidden"])
    n = min(len(ha), len(hb))
    rel = np.array([[np.linalg.norm(ha[i, l] - hb[i, l]) / max(np.linalg.norm(hb[i, l]), 1e-30)
                     for l in range(ha.shape[1])] for i in range(n)])
    ra = np.array([[set(a["routes"][i, l].tolist()) == set(b["routes"][i, l].tolist())
                    for l in range(a["routes"].shape[1])] for i in range(len(a["ids"]))])
    res = {"steps": int(len(a["ids"])),
           "hidden_rel_l2_median_by_layer": [round(float(x), 5) for x in np.median(rel, axis=0)],
           "route_agreement": float(ra.mean()),
           "next_token_agreement": float(np.mean(a["logits_top"][:, 0] == b["logits_top"][:, 0]))}
    if len(a["nll"]) and len(b["nll"]):
        res.update(ppl_a=float(np.exp(a["nll"].mean())), ppl_b=float(np.exp(b["nll"].mean())))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
