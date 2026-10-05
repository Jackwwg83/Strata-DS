"""Build the routing-trace corpus from local text: proto/corpus/docs.jsonl and prompts.jsonl.

Mix (what a coding/agent user of a local model sends): Python, C++/CUDA, Chinese technical prose,
English technical prose, and chat-formatted code questions. Every source is public code or this
project's own docs. Each doc holds about 4K tokens of text; the run truncates to --max-tokens.

Usage: python build_corpus.py --exllamav3 DIR --strata DIR --kit DIR --paper paper.txt
"""
import argparse
import glob
import json
import os

CHARS = {"code_py": 14000, "code_cpp": 14000, "zh": 6500, "en": 15000}
PER_KIND = 6


def chunks_from_files(paths, budget, n):
    """Concatenate files in order and cut the stream into n docs of about `budget` characters."""
    docs, cur = [], ""
    for p in paths:
        try:
            txt = open(p, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        cur += f"\n# ===== {os.path.basename(p)} =====\n" + txt
        while len(cur) >= budget and len(docs) < n:
            docs.append(cur[:budget])
            cur = cur[budget:]
        if len(docs) >= n:
            break
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exllamav3", required=True)
    ap.add_argument("--strata", required=True)
    ap.add_argument("--kit", required=True)
    ap.add_argument("--paper", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    py = sorted(glob.glob(f"{a.exllamav3}/exllamav3/modules/*.py")) + sorted(glob.glob(f"{a.exllamav3}/exllamav3/model/*.py"))
    cpp = sorted(glob.glob(f"{a.strata}/src/core/*.cpp")) + sorted(glob.glob(f"{a.strata}/src/kernels/cuda/*.cu"))
    zh = sorted(glob.glob(f"{a.kit}/docs/*.zh-CN.md")) + [f"{a.kit}/PREFILL_COMPLETE_GUIDE.zh-CN.md"]
    en = [f"{a.strata}/docs/{n}" for n in ("DETAILS.md", "HOW_IT_WORKS.md", "BATCHING.md", "MULTI_GPU.md",
                                           "MODELS.md", "TROUBLESHOOTING.md")] + [a.paper]
    docs = []
    for kind, paths in (("code_py", py), ("code_cpp", cpp), ("zh", zh), ("en", en)):
        for i, text in enumerate(chunks_from_files(paths, CHARS[kind], PER_KIND)):
            docs.append({"id": f"{kind}_{i}", "kind": kind, "text": text})

    # Chat-formatted code questions: the prompt shape a coding agent sends
    questions = ["这段代码有没有并发问题？请指出具体行。", "Explain what this module does and list its public API.",
                 "把这段代码里的错误处理改得更稳健，给出修改后的版本。", "Find the performance bottleneck in this code."]
    snippets = chunks_from_files(py[::-1], 9000, 2) + chunks_from_files(cpp[::-1], 9000, 2)
    for i, (q, code) in enumerate(zip(questions, snippets)):
        docs.append({"id": f"chat_{i}", "kind": "chat", "thinking_mode": "chat",
                     "messages": [{"role": "user", "content": f"{q}\n\n```\n{code}\n```"}]})

    with open(os.path.join(a.out, "docs.jsonl"), "w") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    prompts = [
        ("zh_moe", "用三段话解释什么是混合专家模型（MoE），以及它为什么适合在显存小的电脑上运行。"),
        ("zh_mail", "帮我写一封简短的邮件，通知团队下周三的产品评审改到周四下午三点。"),
        ("zh_pcie", "比较 PCIe 4.0 和 PCIe 5.0 的带宽，并说明它们对本地大模型推理有什么影响。"),
        ("code_st", "Write a Python function that reads a safetensors file header and returns a dict "
                    "mapping tensor name to (dtype, shape). Include error handling."),
        ("code_lru", "Implement a thread-safe LRU cache template in C++ with get and put, then explain "
                     "the locking strategy in two sentences."),
        ("code_zsh", "下面的脚本在 zsh 里运行时 for 循环只执行一次，为什么？怎么修？\n```bash\nPKGS=\"a b c\"\n"
                     "for p in $PKGS; do echo $p; done\n```"),
    ]
    with open(os.path.join(a.out, "prompts.jsonl"), "w") as f:
        for pid, text in prompts:
            f.write(json.dumps({"id": pid, "thinking_mode": "chat",
                                "messages": [{"role": "user", "content": text}]}, ensure_ascii=False) + "\n")
    kinds = {}
    for d in docs:
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    print(f"{len(docs)} docs {kinds}, {len(prompts)} prompts -> {a.out}")


if __name__ == "__main__":
    main()
