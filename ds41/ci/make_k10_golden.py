"""Golden data for task K10 (GPU EXL3 MoE): exllamav3's own LinearEXL3 on real experts from the ds41 pack,
with the prototype's GPU expert math (ds41/proto/ds41_proto.py moe_forward, GPU path).

Usage (on the GPU box with exllamav3 installed):
  EXL3_INT8_GEMV=0 python make_k10_golden.py --pack /workspace/pack-3bpw --out /workspace/ci/golden/k10

EXL3_INT8_GEMV=0 is required. exllamav3 v1.5.4 defaults to mode 2: its small-row path quantizes the activations
to int8 (about 0.9% output deviation per projection, 2.1-2.4% on a whole expert). The golden must be the FP16
path, the exact math; the script refuses to run otherwise.

Writes, per case m in {1, 4, 8}: x_<m>.bin (fp16 [m][5120], already FP8-quantized), sel_<m>.bin (int32 [m][6],
slot indices into the 32 experts), w_<m>.bin (f32 [m][6]), out_<m>.bin (f32 [m][5120]); and experts.txt (the 32
(layer, expert) pairs in slot order).
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools", "ds41"))
sys.path.insert(0, os.path.join(HERE, "..", "proto"))
import pack as PK  # noqa: E402
import torch_kernels as TK  # noqa: E402

H, F, LIM = 5120, 2304, 10.0
SHAPE = {"w1": (H, F), "w3": (H, F), "w2": (F, H)}


def linears(blob, row, device="cuda"):
    from exllamav3.modules.quant.exl3 import LinearEXL3
    out = {}
    for w in ("w1", "w3", "w2"):
        k, n = SHAPE[w]
        t = {}
        for p in ("trellis", "suh", "svh", "mul1"):
            off, nb = row["components"][f"{w}.{p}"]
            raw = torch.from_numpy(np.ascontiguousarray(blob[row["offset"] + off: row["offset"] + off + nb]))
            dt = {"trellis": torch.int16, "suh": torch.float16, "svh": torch.float16, "mul1": torch.int32}[p]
            t[p] = raw.view(dt).to(device)
        bits = row["bits"][w]
        tw = int(round(bits * 16)) if float(bits).is_integer() else int((bits - 0.5) * 16 + 8)
        t["trellis"] = t["trellis"].view(k // 16, n // 16, tw)
        t["mul1"] = t["mul1"].view(())
        out[w] = LinearEXL3(None, k, n, suh=t["suh"], svh=t["svh"], trellis=t["trellis"], mul1=t["mul1"], key=w)
    return out


def main():
    if os.environ.get("EXL3_INT8_GEMV") != "0":
        raise SystemExit("set EXL3_INT8_GEMV=0: the golden is exllamav3's FP16 path, not its int8-activation path")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_default_dtype(torch.bfloat16)
    rows = PK.read_experts_txt(os.path.join(a.pack, "experts.txt"))
    blob = np.memmap(os.path.join(a.pack, "experts.bin"), dtype=np.uint8, mode="r")
    ids = list(range(0, 384, 12))[:32]                       # 32 experts of one layer
    lin = [linears(blob, rows[(a.layer, e)]) for e in ids]
    with open(os.path.join(a.out, "experts.txt"), "w") as f:
        for e in ids:
            f.write(f"{a.layer} {e}\n")
    g = torch.Generator(device="cuda").manual_seed(0)
    for m in (1, 4, 8):
        x = torch.randn(m, H, device="cuda", generator=g).bfloat16()
        TK.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu, True)
        xh = x.half()
        sel = torch.stack([torch.randperm(32, device="cuda", generator=g)[:6] for _ in range(m)]).int()
        if m == 8:
            sel[3, 5] = -1                                   # an empty slot must be skipped
        w = torch.rand(m, 6, device="cuda", generator=g) * 0.4
        out = torch.zeros(m, H, device="cuda", dtype=torch.float32)
        for t in range(m):
            for j in range(6):
                s = int(sel[t, j])
                if s < 0:
                    continue
                L = lin[s]
                xt = xh[t:t + 1].contiguous()
                gate = L["w1"].forward(xt, {}, torch.float).clamp(max=LIM)
                up = L["w3"].forward(xt, {}, torch.float).clamp(-LIM, LIM)
                h = (torch.nn.functional.silu(gate) * up * w[t, j]).to(torch.bfloat16)
                TK.act_quant(h, 32, "ue8m0", torch.float8_e8m0fnu, True)
                out[t] += L["w2"].forward(h.half().contiguous(), {}, torch.float)[0]
        xh.cpu().numpy().tofile(os.path.join(a.out, f"x_{m}.bin"))
        sel.cpu().numpy().astype(np.int32).tofile(os.path.join(a.out, f"sel_{m}.bin"))
        w.float().cpu().numpy().tofile(os.path.join(a.out, f"w_{m}.bin"))
        out.cpu().numpy().tofile(os.path.join(a.out, f"out_{m}.bin"))
        print(f"m={m}: out rms {float(out.pow(2).mean().sqrt()):.4f}")


if __name__ == "__main__":
    main()
