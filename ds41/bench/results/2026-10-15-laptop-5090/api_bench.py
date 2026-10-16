#!/usr/bin/env python3
"""One streamed chat request against the OpenAI API; prints one JSON line: time to first token, client-side decode
tok/s ((completion_tokens - 1) / (last token - first token)), and the server's own usage and timings."""
import json, sys, time, urllib.request
url, label, prompt_file, max_tokens = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
prompt = open(prompt_file, encoding="utf-8").read()
body = json.dumps({"model": "deepseek-v4.1-flash", "stream": True, "max_tokens": max_tokens,
                   "messages": [{"role": "user", "content": prompt}]}).encode()
req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
t0 = time.time(); first = last = None; deltas = 0; usage = timings = None; text = []
with urllib.request.urlopen(req, timeout=3600) as r:
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        obj = json.loads(line[5:])
        for ch in obj.get("choices", []):
            d = ch.get("delta") or {}
            piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
            if piece:
                now = time.time(); first = first or now; last = now; deltas += 1; text.append(piece)
        usage = obj.get("usage") or usage
        timings = obj.get("timings") or timings
n = (usage or {}).get("completion_tokens", deltas)
out = {"label": label, "prompt_tokens": (usage or {}).get("prompt_tokens"), "completion_tokens": n,
       "ttft_s": round(first - t0, 2) if first else None,
       "client_decode_tok_s": round((n - 1) / (last - first), 2) if first and last > first else None,
       "server_timings": timings, "text_head": "".join(text)[:160]}
print(json.dumps(out, ensure_ascii=False), flush=True)
