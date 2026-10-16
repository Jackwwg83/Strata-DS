#!/usr/bin/env python3
"""Task evaluation through the OpenAI API: HumanEval+ (164) or GSM8K (the first 200 of test.jsonl), greedy
(temperature 0), thinking off (chat_template_kwargs enable_thinking false). One JSON line per item: the prompt, the
whole answer, token counts, seconds. Resumes: items already in OUT are skipped.
Usage: eval_run.py URL humaneval|gsm8k DATASET OUT [MAX_TOKENS]"""
import json, sys, time, urllib.request

url, task, data, out = sys.argv[1:5]
max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else (768 if task == "humaneval" else 512)
items = [json.loads(l) for l in open(data)]
if task == "gsm8k":
    items = [{"task_id": f"gsm8k/{i}", **x} for i, x in enumerate(items[:200])]
done = set()
try:
    done = {json.loads(l)["task_id"] for l in open(out)}
except FileNotFoundError:
    pass


def prompt_of(x):
    if task == "humaneval":
        return ("Complete the following Python function. Reply with the whole function (including the imports and "
                "the signature) in one ```python code block.\n\n```python\n" + x["prompt"] + "```")
    return (x["question"] + "\n\nSolve this step by step. End your reply with a line of the form "
            "\"The answer is N.\" where N is a number.")


with open(out, "a") as f:
    for x in items:
        if x["task_id"] in done:
            continue
        p = prompt_of(x)
        body = json.dumps({"model": "deepseek-v4.1-flash", "max_tokens": max_tokens, "temperature": 0,
                           "chat_template_kwargs": {"enable_thinking": False},
                           "messages": [{"role": "user", "content": p}]}).encode()
        req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as r:
            res = json.load(r)
        msg = res["choices"][0]["message"]
        rec = {"task_id": x["task_id"], "prompt": p, "answer": msg.get("content") or "",
               "reasoning": msg.get("reasoning_content") or "", "finish": res["choices"][0].get("finish_reason"),
               "usage": res.get("usage"), "timings": res.get("timings"), "seconds": round(time.time() - t0, 2)}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        print(x["task_id"], rec["finish"], (res.get("usage") or {}).get("completion_tokens"), rec["seconds"], flush=True)
