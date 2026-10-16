"""200 segments of 256 tokens (BOS + 255) from ds41/proto/corpus/docs.jsonl: 40 per kind (code_py, code_cpp, zh, en,
chat: the conversation in the DeepSeek template), consecutive non-overlapping chunks taken round-robin over each kind's documents. One id list per line."""
import json, sys
sys.path[:0] = [".", "tools"]
from serve.deepseek import DeepSeekTemplate, DeepSeekTokenizer
tok, tpl = DeepSeekTokenizer(sys.argv[1]), DeepSeekTemplate()
docs = [json.loads(l) for l in open("ds41/proto/corpus/docs.jsonl")]
L, PER = 255, 40
out, meta = [], []
for kind in ("code_py", "code_cpp", "zh", "en", "chat"):
    chunks = []
    for d in (d for d in docs if d["kind"] == kind):
        ids = (tok.encode(d["text"]) if "text" in d else   # chat: the rendered conversation (thinking off)
               tok.encode(tpl.render(d["messages"], enable_thinking=False))[1:])
        chunks.append([(d["id"], i, ids[i:i + L]) for i in range(0, len(ids) - L + 1, L)])
    picked, r = [], 0
    while len(picked) < PER and any(r < len(c) for c in chunks):
        for c in chunks:
            if r < len(c) and len(picked) < PER: picked.append(c[r])
        r += 1
    assert len(picked) == PER, (kind, len(picked))
    for doc, at, ids in picked:
        out.append([0] + ids); meta.append({"kind": kind, "doc": doc, "offset": at})
open(sys.argv[2], "w").write("".join(",".join(map(str, s)) + "\n" for s in out))
json.dump(meta, open(sys.argv[2] + ".meta.json", "w"))
print(len(out), "segments", sum(len(s) for s in out), "tokens")
