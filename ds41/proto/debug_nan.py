"""Find the first kernel call that returns non-finite values, for several prompt lengths.

Usage: python debug_nan.py --model-dir /workspace/model --doc corpus/docs.jsonl --doc-id code_py_0
"""
import argparse
import json
import sys

import torch

import ds41_proto as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--doc-id", default="code_py_0")
    ap.add_argument("--lengths", default="64,200,600,1500,3844")
    ap.add_argument("--kernels", default="auto")
    a = ap.parse_args()
    model, tok, args, info = P.build_model(a.model_dir, 4200, a.kernels)
    import model as M
    doc = next(json.loads(l) for l in open(a.doc) if json.loads(l)["id"] == a.doc_id)
    ids = tok.encode(doc["text"])

    first = {}

    def wrap(name, fn):
        def inner(*args, **kw):
            out = fn(*args, **kw)
            outs = out if isinstance(out, tuple) else (out,)
            for i, o in enumerate(outs):
                if torch.is_tensor(o) and o.is_floating_point() and not first:
                    bad = ~torch.isfinite(o.float())
                    if bad.any():
                        ins = [(tuple(t.shape), str(t.dtype), bool((~torch.isfinite(t.float())).any()))
                               for t in args if torch.is_tensor(t) and t.is_floating_point()]
                        first.update(name=name, out_index=i, shape=tuple(o.shape), n_bad=int(bad.sum()),
                                     inputs=ins, layer=CUR["layer"])
            return out
        return inner

    for n in ("act_quant", "fp4_act_quant", "fp8_gemm", "sparse_attn", "hc_split_sinkhorn"):
        setattr(M, n, wrap(n, getattr(M, n)))
    CUR = {"layer": None}
    for i, layer in enumerate(model.layers):
        layer.register_forward_pre_hook(lambda m, inp, i=i: CUR.__setitem__("layer", i))
        for sub in ("attn", "ffn", "attn_norm", "ffn_norm"):
            def hook(m, inp, out, i=i, sub=sub):
                if not first and torch.is_tensor(out) and not torch.isfinite(out.float()).all():
                    first.update(name=f"module {sub}", layer=i, shape=tuple(out.shape))
                return None                       # a non-None return would replace the output
            getattr(layer, sub).register_forward_hook(hook)
    for n in map(int, a.lengths.split(",")):
        first.clear()
        P.STATE["routes"].reset()
        x = torch.tensor([ids[:n]], device="cuda")
        model(x, 0)
        print(json.dumps({"len": n, "first_nonfinite": first}, default=str), flush=True)


if __name__ == "__main__":
    sys.exit(main())
