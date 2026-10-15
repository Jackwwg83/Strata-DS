"""Usage: make_len_ids.py PACK_DIR OUT_DIR 4096,32768,65536,131072,262144 2

Upstream's length-tiered speed prompts (bench/results/2026-10-06-community-rtx-5090-laptop/benchmark.py), rendered
with serve's DeepSeek template (thinking off, as reasoning_effort none) and tokenizer. Writes <target>-run<r>.ids."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]   # the repo
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from serve.deepseek import DeepSeekTemplate, DeepSeekTokenizer

tok, tpl = DeepSeekTokenizer(sys.argv[1]), DeepSeekTemplate()
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
targets = [int(t) for t in sys.argv[3].split(",")]
runs = int(sys.argv[4])
filler = "\n".join(f"def task_{i:05d}(value: int) -> int: return (value * {(i % 97) + 1} + {i}) % 100003"
                   for i in range(30000))
ending = ("\n\nWrite a detailed explanation of the code above. Discuss deterministic integer transforms, "
          "modulo arithmetic, testing, naming, complexity, and maintainability. Write at least 600 words.")
def ids(content):
    return tok.encode(tpl.render([{"role": "user", "content": content}], enable_thinking=False))
for target in targets:
    for run in range(1, runs + 1):
        prefix = f"Benchmark nonce: series-{target}-trial-{run}.\nReview this synthetic Python module:\n"
        lo, hi = 0, len(filler)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(ids(prefix + filler[:mid] + ending)) <= target: lo = mid
            else: hi = mid - 1
        v = ids(prefix + filler[:lo] + ending)
        assert target - 20 <= len(v) <= target, (target, len(v))
        (out / f"{target}-run{run}.ids").write_text(",".join(map(str, v)))
        print(target, run, len(v), flush=True)
