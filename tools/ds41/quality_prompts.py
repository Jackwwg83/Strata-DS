"""Fixed chat prompts for the quality comparison between packs (3bpw against SAGE 1.59bpw), and decoding of the
generated ids back to text.

Usage:
  python quality_prompts.py ids --model /workspace/model --out DIR     writes DIR/<name>.ids (chat template applied)
  python quality_prompts.py text --model /workspace/model --log FILE   prints the text of a 'generated:' line
Same prompts, same ids on every machine: the answers of two packs can be put side by side.
"""
import argparse
import os

from transformers import AutoTokenizer

PROMPTS = {
    "zh_explain": "用三句话解释为什么天空是蓝色的。",
    "en_history": "Write a short paragraph about the history of the printing press.",
    "code_fib": "Write a Python function that returns the n-th Fibonacci number using memoization.",
    "math_speed": "A train travels 120 km in 1.5 hours, then 80 km in 1 hour. What is its average speed for the "
                  "whole trip in km/h? Show your steps.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ids", "text"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out")
    ap.add_argument("--log")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    if a.mode == "ids":
        os.makedirs(a.out, exist_ok=True)
        for name, text in PROMPTS.items():
            ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True)
            with open(os.path.join(a.out, name + ".ids"), "w") as f:
                f.write(",".join(map(str, ids)))
        return
    with open(a.log) as f:
        line = next((l for l in f if l.startswith("generated:")), "")
    print(tok.decode([int(v) for v in line.split()[1:]]))


if __name__ == "__main__":
    main()
