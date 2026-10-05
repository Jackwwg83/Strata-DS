"""Export the reference the C++ engine is checked against: per-layer hidden states, logits and routing,
one token at a time (token 0 through the prefill path with one token, every later token through the decode
path), which is how the M1 engine runs.

Usage:
  python dump_oracle.py --model-dir /workspace/model --out oracle/ --prompts corpus/prompts.jsonl \
      --gen-tokens 32 --kernels torch

Writes oracle/<id>.npz with:
  ids        int32 [T]        prompt ids followed by the greedy continuation (the fed tokens)
  hidden     float16 [H, 40, 4, 5120]   the hc stream after each block, for the first H fed tokens
  logits_top int32 [T, 8], logits_val float32 [T, 8]  top 8 of the last-position logits
  routes     int16 [T, 40, 6], weights float32 [T, 40, 6]
"""
import argparse
import json
import os

import numpy as np
import torch

import ds41_proto as P


@torch.inference_mode()
def run(model, ids_prompt, gen_tokens, hidden_tokens):
    hidden = []
    hooks = []
    cur = {}

    for i, layer in enumerate(model.layers):
        def hook(m, inp, out, i=i):
            cur.setdefault("h", {})[i] = out[0][0, -1].float().cpu().numpy()   # [4, 5120] of the last token
            return None
        hooks.append(layer.register_forward_hook(hook))

    fed, tops, vals, routes, weights = [], [], [], [], []
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
        tops.append(li.cpu().numpy().astype(np.int32))
        vals.append(lv.cpu().numpy())
        if pos < hidden_tokens:
            hidden.append(np.stack([cur["h"][i] for i in range(len(model.layers))]).astype(np.float16))
        r, w = P.STATE["routes"].stacked()
        routes.append(r[0])
        weights.append(w[0].astype(np.float32))
    for h in hooks:
        h.remove()
    return {"ids": np.array(fed, np.int32), "hidden": np.stack(hidden),
            "logits_top": np.stack(tops), "logits_val": np.stack(vals),
            "routes": np.stack(routes).astype(np.int16), "weights": np.stack(weights)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--kernels", default="torch")
    ap.add_argument("--hidden-tokens", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    model, tok, args, info = P.build_model(a.model_dir, 1024, a.kernels)
    for line in open(a.prompts):
        p = json.loads(line)
        ids = tok.encode(P.chat_prompt(p["messages"], p.get("thinking_mode", "chat")))
        r = run(model, ids, a.gen_tokens, a.hidden_tokens)
        np.savez_compressed(os.path.join(a.out, f"{p['id']}.npz"), **r)
        print(json.dumps({"id": p["id"], "tokens": int(len(r["ids"])),
                          "text": tok.decode(r["ids"][len(ids):].tolist())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
