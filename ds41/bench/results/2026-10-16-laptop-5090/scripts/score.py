#!/usr/bin/env python3
"""Scoring.
  score.py gsm GOLD.jsonl RUN.jsonl            -> per-item correct (the last "The answer is N", else the last number)
  score.py he-samples RUN.jsonl SAMPLES.jsonl  -> evalplus samples (task_id, solution: the first python code block)
  score.py pair NAME A.json C.json             -> paired comparison of two per-item result maps {task_id: bool}:
                                                 accuracies, difference with a 95% bootstrap interval (10,000
                                                 resamples, seed 1) and the exact McNemar p-value"""
import json, math, random, re, sys


def last_number(s):
    m = re.findall(r"answer is\s*\$?\s*(-?[\d,]*\.?\d+)", s, re.I) or re.findall(r"-?[\d,]*\.?\d+", s)
    return m[-1].replace(",", "").rstrip(".") if m else None


def num_eq(a, b):
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


cmd = sys.argv[1]
if cmd == "gsm":
    gold = [json.loads(l) for l in open(sys.argv[2])][:200]
    res = {}
    for r in map(json.loads, open(sys.argv[3])):
        i = int(r["task_id"].split("/")[1])
        want = gold[i]["answer"].split("####")[-1].strip().replace(",", "")
        res[r["task_id"]] = num_eq(last_number(r["answer"]), want)
    print(json.dumps(res))
elif cmd == "he-samples":
    with open(sys.argv[3], "w") as out:
        for r in map(json.loads, open(sys.argv[2])):
            m = re.search(r"```(?:python)?\n(.*?)```", r["answer"], re.S)
            out.write(json.dumps({"task_id": r["task_id"], "solution": m.group(1) if m else r["answer"]}) + "\n")
elif cmd == "pair":
    name, a, c = sys.argv[2], json.load(open(sys.argv[3])), json.load(open(sys.argv[4]))
    ids = sorted(set(a) & set(c))
    xa, xc = [int(a[i]) for i in ids], [int(c[i]) for i in ids]
    n = len(ids)
    rng = random.Random(1)
    diffs = []
    for _ in range(10000):
        s = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(xc[i] - xa[i] for i in s) / n)
    diffs.sort()
    b = sum(1 for i in range(n) if xa[i] and not xc[i])   # right without skips, wrong with
    cc = sum(1 for i in range(n) if xc[i] and not xa[i])
    k, m = min(b, cc), b + cc
    p = min(1.0, 2 * sum(math.comb(m, j) for j in range(k + 1)) / 2 ** m) if m else 1.0
    print(json.dumps({"task": name, "n": n, "acc_no_skip": sum(xa) / n, "acc_default": sum(xc) / n,
                      "diff": (sum(xc) - sum(xa)) / n, "diff_ci95": [diffs[250], diffs[9750]],
                      "only_no_skip_right": b, "only_default_right": cc, "mcnemar_p": p}))
