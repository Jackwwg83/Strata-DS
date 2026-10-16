#!/usr/bin/env python3
"""Needle-in-a-haystack prompts: lengths 32K/64K/128K/256K tokens x depth 10/50/90% x 5 trials (seed 20261015).
Haystack: this repository's docs (*.md) and Python sources in a fixed order; each trial starts at its own offset.
Needle: "The vault code of the Pelican Observatory is NNNNNNN." inserted at a line boundary at the depth; question at the
end. The needle's key words must not occur in the haystack. Writes one JSON line per trial (text, code, length,
depth, measured token count). Usage: make_niah.py REPO TOKENIZER_DIR OUT.jsonl"""
import json, random, sys
from pathlib import Path

repo, out = Path(sys.argv[1]), sys.argv[3]
sys.path[:0] = [str(repo), str(repo / "tools")]
from serve.deepseek import DeepSeekTemplate, DeepSeekTokenizer

tok, tpl = DeepSeekTokenizer(sys.argv[2]), DeepSeekTemplate()
files = sorted(repo.glob("docs/*.md")) + sorted(repo.glob("serve/*.py")) + sorted(repo.glob("tools/*.py")) + \
    [repo / "setup.py"] + sorted(repo.glob("src/ds41/*.cu")) + sorted(repo.glob("src/ds41/*.cpp"))
hay = "\n\n".join(f"===== {f.relative_to(repo)} =====\n{f.read_text(errors='replace')}" for f in files)
assert "pelican" not in hay.lower() and "observatory" not in hay.lower() and "vault code" not in hay.lower(), "needle words occur in the haystack"
ids = tok.encode(hay)
QUESTION = ("\n\nWhat is the vault code of the Pelican Observatory mentioned in the text above? "
            "Reply with the vault code only.")
rng = random.Random(20261015)
rows = []
for length in (32768, 65536, 131072, 262144):
    for depth in (0.1, 0.5, 0.9):
        for trial in range(5):
            code = str(rng.randrange(1_000_000, 10_000_000))
            needle = f"\nThe vault code of the Pelican Observatory is {code}.\n"
            budget = length - 120          # template, needle and question
            start = rng.randrange(0, max(1, len(ids) - budget))
            body = tok.decode(ids[start:start + budget])
            cut = body.find("\n", int(len(body) * depth))
            cut = len(body) if cut < 0 else cut
            text = body[:cut] + needle + body[cut:] + QUESTION
            n = len(tok.encode(tpl.render([{"role": "user", "content": text}], enable_thinking=False)))
            rows.append({"length": length, "depth": depth, "trial": trial, "code": code, "start": start,
                         "prompt_tokens": n, "text": text})
            print(length, depth, trial, n, flush=True)
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("haystack tokens", len(ids), "trials", len(rows))
