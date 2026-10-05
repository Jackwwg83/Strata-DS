"""Export the reference the C++ engine is checked against: per-layer hidden states, logits and routing,
one token at a time (token 0 through the prefill path with one token, every later token through the decode
path), which is how the M1 engine runs.

Usage:
  python dump_oracle.py --model-dir /workspace/model --out oracle/ --prompts corpus/prompts.jsonl \
      --gen-tokens 32 --kernels torch

Writes oracle/<id>.npz with:
  ids        int32 [T]        prompt ids followed by the greedy continuation (the fed tokens)
  hidden     uint16 [H, 40, 4, 5120]    bf16 bits of the hc stream after each block, for the first H fed tokens
  dbg        uint16 [H, 2, 5 parts]     layers 1-2: block input, attn input, attn output, ffn input, ffn output
  logits_top int32 [T, 8], logits_val float32 [T, 8]  top 8 of the last-position logits
  routes     int16 [T, 40, 6], weights float32 [T, 40, 6]
  nll        float32 [T-1]   -log p(ids[t+1]) at step t (teacher forcing; meaningful for --ids-file runs)
"""
import argparse
import json
import os

import numpy as np
import torch

import ds41_proto as P


@torch.inference_mode()
def run(model, ids_prompt, gen_tokens, hidden_tokens, forced=False):
    hidden, dbg = [], []
    hooks = []
    cur = {}

    for i, layer in enumerate(model.layers):
        def hook(m, inp, out, i=i):
            cur.setdefault("h", {})[i] = out[0][0, -1].contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
            return None
        hooks.append(layer.register_forward_hook(hook))
        if i in (1, 2):
            def bits(t):
                return t[0, -1].contiguous().view(torch.int16).cpu().numpy().view(np.uint16).reshape(-1)

            def pre_block(m, args, i=i):
                cur.setdefault("dbg", {})[(i, 0)] = bits(args[0])
            def attn_hook(m, args, out, i=i):
                cur["dbg"][(i, 1)] = bits(args[0]); cur["dbg"][(i, 2)] = bits(out)
            def ffn_hook(m, args, out, i=i):
                cur["dbg"][(i, 3)] = bits(args[0]); cur["dbg"][(i, 4)] = bits(out)
            hooks.append(layer.register_forward_pre_hook(pre_block))
            hooks.append(layer.attn.register_forward_hook(attn_hook))
            hooks.append(layer.ffn.register_forward_hook(ffn_hook))

    fed, tops, vals, routes, weights, nll = [], [], [], [], [], []
    nxt = None
    total = len(ids_prompt) + gen_tokens
    for pos in range(total):
        tok = ids_prompt[pos] if pos < len(ids_prompt) else int(nxt)
        fed.append(tok)
        P.STATE["routes"].reset()
        cur.clear()
        out_ids, logits, _ = model(torch.tensor([[tok]], device="cuda"), pos)
        nxt = int(out_ids[0])
        lv, li = logits[0].float().topk(8)
        if forced and pos + 1 < len(ids_prompt):
            lp = torch.log_softmax(logits[0].float(), -1)
            nll.append(float(-lp[ids_prompt[pos + 1]]))
        tops.append(li.cpu().numpy().astype(np.int32))
        vals.append(lv.cpu().numpy())
        if pos < hidden_tokens:
            hidden.append(np.stack([cur["h"][i] for i in range(len(model.layers))]))
            dbg.append(np.concatenate([cur["dbg"][(l, k)] for l in (1, 2) for k in range(5)]))
        r, w = P.STATE["routes"].stacked()
        routes.append(r[0])
        weights.append(w[0].astype(np.float32))
    for h in hooks:
        h.remove()
    return {"ids": np.array(fed, np.int32), "hidden": np.stack(hidden), "dbg": np.stack(dbg),
            "logits_top": np.stack(tops), "logits_val": np.stack(vals),
            "routes": np.stack(routes).astype(np.int16), "weights": np.stack(weights),
            "nll": np.array(nll, np.float32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts")
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--kernels", default="torch")
    ap.add_argument("--hidden-tokens", type=int, default=16)
    ap.add_argument("--docs", help="docs.jsonl: teacher-force the first --doc-tokens tokens of each doc instead")
    ap.add_argument("--doc-ids", default="code_py_0,zh_0")
    ap.add_argument("--doc-tokens", type=int, default=200)
    ap.add_argument("--experts", default="cpu", choices=["cpu", "gpu"],
                    help="cpu: exllamav3 moe_mul1, the kernel the C++ engine uses (default)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model, tok, args, info = P.build_model(a.model_dir, 1024, a.kernels)
    if a.experts == "cpu":
        P.STATE["cpu_experts"] = P.CpuExperts(P.STATE["ckpt"])
    if a.docs:
        want = a.doc_ids.split(",")
        for line in open(a.docs):
            d = json.loads(line)
            if d["id"] not in want:
                continue
            ids = tok.encode(d["text"])[: a.doc_tokens]
            r = run(model, ids, 0, a.hidden_tokens, forced=True)
            np.savez_compressed(os.path.join(a.out, f"{d['id']}.npz"), **r)
            print(json.dumps({"id": d["id"], "tokens": len(ids), "mean_nll": float(r["nll"].mean()),
                              "ppl": float(np.exp(r["nll"].mean()))}), flush=True)
        return
    for line in open(a.prompts):
        p = json.loads(line)
        ids = tok.encode(P.chat_prompt(p["messages"], p.get("thinking_mode", "chat")))
        r = run(model, ids, a.gen_tokens, a.hidden_tokens)
        np.savez_compressed(os.path.join(a.out, f"{p['id']}.npz"), **r)
        print(json.dumps({"id": p["id"], "tokens": int(len(r["ids"])),
                          "text": tok.decode(r["ids"][len(ids):].tolist())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
