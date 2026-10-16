#!/usr/bin/env python3
"""Streamed chat requests through the OpenAI API, timed by the client. Per request one JSON line: prompt and completion
tokens, time to first token, decode tokens/s ((n - 1) / (last - first)), the gaps between streamed pieces (ms) and the
server's timings and the whole text (reasoning, then answer). Thinking on (the server's default). Two modes:
  api_stream.py URL OUT.jsonl LABEL PROMPT_FILE MAX_TOKENS          one request
  api_stream.py URL OUT.jsonl --sustain SECONDS MAX_TOKENS FILE...  requests back to back, cycling the files, until
                                                                      SECONDS have passed"""
import json, sys, time, urllib.request


def one(url, label, prompt, max_tokens):
    body = json.dumps({"model": "deepseek-v4.1-flash", "stream": True, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time(); stamps = []; usage = timings = None; text = []
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            obj = json.loads(line[5:])
            for ch in obj.get("choices", []):
                d = ch.get("delta") or {}
                piece = (d.get("reasoning_content") or "") + (d.get("content") or "")
                if piece:
                    stamps.append(time.time()); text.append(piece)
            usage = obj.get("usage") or usage
            timings = obj.get("timings") or timings
    n = (usage or {}).get("completion_tokens", len(stamps))
    gaps = [round((b - a) * 1000, 2) for a, b in zip(stamps, stamps[1:])]
    return {"label": label, "start": round(t0, 2), "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": n, "pieces": len(stamps), "ttft_s": round(stamps[0] - t0, 3) if stamps else None,
            "decode_tok_s": round((n - 1) / (stamps[-1] - stamps[0]), 2) if len(stamps) > 1 else None,
            "gaps_ms": gaps, "server_timings": timings, "text": "".join(text)}


url, out = sys.argv[1], sys.argv[2]
with open(out, "a") as f:
    if sys.argv[3] == "--sustain":
        seconds, max_tokens, files = float(sys.argv[4]), int(sys.argv[5]), sys.argv[6:]
        end, i = time.time() + seconds, 0
        while time.time() < end:
            p = files[i % len(files)]
            r = one(url, f"sustain-{i}-{p.rsplit('/', 1)[-1]}", open(p, encoding="utf-8").read(), max_tokens)
            f.write(json.dumps(r) + "\n"); f.flush()
            print(r["label"], r["completion_tokens"], r["decode_tok_s"], flush=True)
            i += 1
    else:
        r = one(url, sys.argv[3], open(sys.argv[4], encoding="utf-8").read(), int(sys.argv[5]))
        f.write(json.dumps(r) + "\n")
        print(r["label"], r["prompt_tokens"], r["completion_tokens"], r["ttft_s"], r["decode_tok_s"], flush=True)
