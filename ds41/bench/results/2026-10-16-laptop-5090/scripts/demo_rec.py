#!/usr/bin/env python3
"""A live streamed request for the demo video: every streamed piece with its arrival time (seconds after the request)
and kind (reasoning or answer). Thinking on, the server's defaults. Usage: demo_rec.py URL PROMPT_FILE MAX_TOKENS OUT.json [nothink]"""
import json, sys, time, urllib.request

url, pf, max_tokens, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
prompt = open(pf, encoding="utf-8").read()
body = json.dumps({"model": "deepseek-v4.1-flash", "stream": True, "max_tokens": max_tokens,
                   "messages": [{"role": "user", "content": prompt}],
                   **({"chat_template_kwargs": {"enable_thinking": False}} if sys.argv[5:] == ["nothink"] else {})}).encode()
req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
t0 = time.time(); pieces = []; usage = None
with urllib.request.urlopen(req, timeout=3600) as r:
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        obj = json.loads(line[5:])
        for ch in obj.get("choices", []):
            d = ch.get("delta") or {}
            for kind in ("reasoning_content", "content"):
                if d.get(kind):
                    pieces.append([round(time.time() - t0, 4), "r" if kind == "reasoning_content" else "a", d[kind]])
        usage = obj.get("usage") or usage
json.dump({"started": t0, "prompt": prompt, "usage": usage, "pieces": pieces}, open(out, "w"), ensure_ascii=False)
print(len(pieces), "pieces", usage)
