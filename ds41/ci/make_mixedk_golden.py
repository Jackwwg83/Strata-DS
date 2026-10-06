"""Synthetic K1..K6 fixtures for K10 and K11, using the K10 FP16 reference.

EXL3_INT8_GEMV=0 python ds41/ci/make_mixedk_golden.py --out /workspace/ci/golden/mixedk
Use --weights-only for a host-only fixture format check. It writes no reference outputs.
"""
import argparse
import os
from pathlib import Path

import numpy as np

H, F = 5120, 2304
RATES = [(1, 2, 6), (2, 3, 1), (3, 4, 2), (4, 5, 3), (5, 6, 4), (6, 1, 5)]


def weights(out):
    rng = np.random.default_rng(410159)
    for e, rates in enumerate(RATES):
        for name, bits in zip(("w1", "w3", "w2"), rates):
            k, n = (F, H) if name == "w2" else (H, F)
            prefix = out / f"{e}_{name}"
            rng.integers(0, 65536, (k // 16, n // 16, 16 * bits), dtype=np.uint16).tofile(str(prefix) + ".trellis")
            for part, dim, scale in (("suh", k, 1.0), ("svh", n, 0.05 if name == "w2" else 0.02)):
                sign = 2 * rng.integers(0, 2, dim) - 1
                (sign * scale * (0.5 + rng.random(dim))).astype(np.float16).tofile(str(prefix) + "." + part)
    (out / "rates.txt").write_text("".join(" ".join(map(str, row)) + "\n" for row in RATES))
    print("Synthetic weights: 6 experts, 18 projections, each projection covers K1..K6")


def golden(out):
    if os.environ.get("EXL3_INT8_GEMV") != "0":
        raise SystemExit("Set EXL3_INT8_GEMV=0 for the independent FP16 reference")
    import torch
    from exllamav3.modules.quant.exl3 import LinearEXL3
    from make_k10_golden import TK

    torch.set_default_dtype(torch.bfloat16)
    lin = []
    for e, rates in enumerate(RATES):
        expert = {}
        for name, bits in zip(("w1", "w3", "w2"), rates):
            k, n = (F, H) if name == "w2" else (H, F)
            prefix = str(out / f"{e}_{name}")
            trellis = torch.from_numpy(np.fromfile(prefix + ".trellis", np.int16).reshape(k // 16, n // 16, 16 * bits)).cuda()
            suh = torch.from_numpy(np.fromfile(prefix + ".suh", np.float16)).cuda()
            svh = torch.from_numpy(np.fromfile(prefix + ".svh", np.float16)).cuda()
            expert[name] = LinearEXL3(None, k, n, suh=suh, svh=svh, trellis=trellis,
                                      mul1=torch.tensor(1, device="cuda", dtype=torch.int32), key=name)
        lin.append(expert)
    gen = torch.Generator(device="cuda").manual_seed(410159)
    for m in (1, 4, 8):
        x = torch.randn(m, H, device="cuda", generator=gen).bfloat16()
        TK.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, True)
        xh = x.half()
        # Each token selects all six rates in a different order. Keep one empty slot.
        sel = torch.tensor([[(t + j) % 6 for j in range(6)] for t in range(m)], device="cuda", dtype=torch.int32)
        if m == 8:
            sel[3, 5] = -1
        # Dyadic weights are exact in both CPU half and GPU float storage.
        w = torch.tensor([[((t + j) % 6 + 1) / 32 for j in range(6)] for t in range(m)],
                         device="cuda", dtype=torch.float32)
        result = torch.zeros(m, H, device="cuda", dtype=torch.float32)
        for t in range(m):
            for j in range(6):
                e = int(sel[t, j])
                if e < 0:
                    continue
                layer = lin[e]
                xt = xh[t:t + 1].contiguous()
                gate = layer["w1"].forward(xt, {}, torch.float).clamp(max=10.0)
                up = layer["w3"].forward(xt, {}, torch.float).clamp(-10.0, 10.0)
                hidden = (torch.nn.functional.silu(gate) * up * w[t, j]).to(torch.bfloat16)
                TK.act_quant(hidden, 32, "ue8m0", torch.float8_e8m0fnu, True)
                result[t] += layer["w2"].forward(hidden.half().contiguous(), {}, torch.float)[0]
        for name, tensor in (("x", xh), ("sel", sel), ("w", w), ("out", result)):
            tensor.cpu().numpy().tofile(out / f"{name}_{m}.bin")
        print(f"mixed K m={m}: FP16 LinearEXL3 reference rms={result.square().mean().sqrt().item():.6g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    weights(args.out)
    if not args.weights_only:
        golden(args.out)


if __name__ == "__main__":
    main()
