"""tools/ds41/nll_ids.py - fixed texts as token ids, for a teacher-forced nll (ds41_generate --force-ids FILE).

    python tools/ds41/nll_ids.py --tokenizer PACK_DIR --out DIR [--tokens 768]

Writes DIR/<name>.ids (comma separated, BOS first): the first N tokens of a Chinese document, an English document and
Python source of this repository, tokenized as serve/server.py does it (serve/deepseek.py). Same files, same ids on
every machine of the same commit. Used to measure what an opt-in speed switch (DS41_SKIP_MISS) costs in quality.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from ds41.proto.ref import encoding as ref  # noqa: E402
from serve.deepseek import DeepSeekTokenizer  # noqa: E402

TEXTS = {"zh_doc": "README.zh-CN.md", "en_doc": "docs/DETAILS.md", "code": "serve/frontend.py"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="the pack (or model) directory with tokenizer.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=768)
    a = ap.parse_args()
    if a.tokens < 2:
        ap.error("--tokens must be at least 2")
    tok = DeepSeekTokenizer(Path(a.tokenizer))
    bos = tok.special_tokens[ref.bos_token]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rel in TEXTS.items():
        text = (ROOT / rel).read_text(encoding="utf-8")
        ids = [bos] + tok.encode(text, parse_special=False)[:a.tokens - 1]
        if len(ids) < a.tokens:
            raise SystemExit(f"{rel} has only {len(ids)} tokens")
        (out / f"{name}.ids").write_text(",".join(map(str, ids)))
        print(f"{name}: {len(ids)} tokens from {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
