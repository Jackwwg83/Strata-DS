"""Token ids of a long, fixed text for the context benchmark (M3): this repository's documents and sources,
concatenated in a fixed order and tokenized with the model's tokenizer. Same files, same ids on every machine.

Usage: python make_long_ids.py --model /workspace/model --repo /workspace/Strata-DS --tokens 33000 --out long.ids
Past the end of the file list the text starts again, so any length works (for speed runs).
Writes comma-separated ids (BOS first), as ds41_generate --ids / --force-ids read them.
"""
import argparse
import glob
import os

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tokens", type=int, default=33000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    patterns = ["README.md", "docs/*.md", "ds41/tasks/*.md", "src/ds41/*.cu", "src/ds41/*.cpp", "include/strata/ds41/*.hpp",
                "src/ds41/kernels/*.cu", "tools/ds41/*.py", "ds41/proto/ref/*.py", "src/**/*.cpp"]
    files = []
    for p in patterns:
        for f in sorted(glob.glob(os.path.join(a.repo, p), recursive=True)):
            if f not in files:
                files.append(f)
    ids = [tok.bos_token_id] if tok.bos_token_id is not None else []
    if not files:
        raise SystemExit("no files")
    # past the end of the file list the text starts again (a long-context speed run only needs the length)
    rounds = 0
    while len(ids) < a.tokens:
        for f in files:
            with open(f, encoding="utf-8", errors="replace") as fh:
                text = f"\n\n# file: {os.path.relpath(f, a.repo)}\n\n" + fh.read()
            ids += tok.encode(text, add_special_tokens=False)
            if len(ids) >= a.tokens:
                break
        rounds += 1
    ids = ids[: a.tokens]
    with open(a.out, "w") as fh:
        fh.write(",".join(map(str, ids)))
    print(f"{len(ids)} tokens from {len(files)} files ({rounds} pass(es) over them) -> {a.out}")


if __name__ == "__main__":
    main()
