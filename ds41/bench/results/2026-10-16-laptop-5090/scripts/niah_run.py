#!/usr/bin/env python3
"""Needle-in-a-haystack through the OpenAI API: greedy, thinking off, up to 32 tokens. One JSON line per trial with
the answer and whether it contains the 7-digit code. Resumes. Usage: niah_run.py URL niah60.jsonl OUT"""
import json, sys, time, urllib.request
url, data, out = sys.argv[1:4]
done = set()
try:
    done = {(r["length"], r["depth"], r["trial"]) for r in map(json.loads, open(out))}
except FileNotFoundError:
    pass
with open(out, "a") as f:
    for x in map(json.loads, open(data)):
        key = (x["length"], x["depth"], x["trial"])
        if key in done:
            continue
        body = json.dumps({"model": "deepseek-v4.1-flash", "max_tokens": 32, "temperature": 0,
                           "chat_template_kwargs": {"enable_thinking": False},
                           "messages": [{"role": "user", "content": x["text"]}]}).encode()
        req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=7200) as r:
            res = json.load(r)
        ans = res["choices"][0]["message"].get("content") or ""
        rec = {"length": x["length"], "depth": x["depth"], "trial": x["trial"], "code": x["code"], "answer": ans,
               "correct": x["code"] in ans, "usage": res.get("usage"), "timings": res.get("timings"),
               "seconds": round(time.time() - t0, 1)}
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(*key, rec["correct"], repr(ans[:40]), rec["seconds"], flush=True)
