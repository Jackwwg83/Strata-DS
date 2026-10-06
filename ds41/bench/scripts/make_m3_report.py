"""M3 test report: prefill and decode speed by context length on several machines, plus the prefill checks.
One self-contained HTML file (inline CSS and SVG charts, no network).

Usage: python make_m3_report.py --config report.json --out ds41/docs/m3-report.html

report.json:
{
  "title": "...", "date": "2026-10-06", "summary": ["...", "..."],      (summary: one sentence per item)
  "machines": [
    {"label": "7950X + RTX 4090", "machine_json": "path/machine.json", "context_tsv": "path/m3_context_X.tsv",
     "verify_log": "path/m3_verify.log", "notes": ["..."]},
    {"label": "i9-14900K + RTX 4090 (earlier)", "static": {"cpu": "...", "gpu": "...", "ram": "..."},
     "decode_only": [[512, 2.69]], "notes": ["..."]}
  ],
  "notes": ["..."]
}
context_tsv columns (m3_context.sh): machine context prefill_ms prefill_tok_s decode_ms_per_token decode_tok_s chunk
streamed ssd stream_wait_ms
"""
import argparse
import html
import json
import math
import re

COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]


def read_tsv(path):
    rows = []
    with open(path) as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 6 or not c[1].isdigit():
                continue
            num = lambda s: float(s) if s not in ("", None) else None
            rows.append({"context": int(c[1]), "prefill_ms": num(c[2]), "prefill_tok_s": num(c[3]),
                         "decode_ms": num(c[4]), "decode_tok_s": num(c[5]),
                         "chunk": c[6] if len(c) > 6 else "", "streamed": c[7] if len(c) > 7 else "",
                         "ssd": c[8] if len(c) > 8 else "", "wait_ms": num(c[9]) if len(c) > 9 else None})
    return sorted(rows, key=lambda r: r["context"])


def read_verify(path):
    out = {"nll": [], "long": None, "gen": []}
    with open(path) as f:
        for line in f:
            m = re.match(r"NLL (\S+) tokens (\d+) \| prototype_fp16 (\S+) \| step (\S*) \| prefill (\S*) \| chunk61 (\S*)",
                         line)
            if m:
                out["nll"].append(m.groups())
            m = re.match(r"LONG (\d+) tokens \| one chunk (\S*) \| chunks of 999 (\S*)", line)
            if m:
                out["long"] = m.groups()
            if line.startswith("GEN "):
                out["gen"].append(line.strip())
    return out


def fmt(v, digits=1):
    if v is None:
        return "–"
    if v >= 100:
        return f"{v:,.0f}"
    return f"{v:.{digits}f}"


def chart(series, key, title, unit):
    """Line chart: x = context (log 2), y = metric. series: [(label, color, [(x, y)])]"""
    pts = [(x, y) for _, _, s in series for x, y in s if y is not None]
    if not pts:
        return f'<div class="chart-empty">{html.escape(title)}: no data yet</div>'
    W, H, L, R, T, B = 720, 360, 64, 20, 30, 50
    xs = sorted({x for x, _ in pts})
    x0, x1 = math.log2(min(xs)), math.log2(max(xs))
    if x1 == x0:
        x1 = x0 + 1
    ymax = max(y for _, y in pts) * 1.12
    step = 10 ** math.floor(math.log10(ymax / 5)) if ymax > 0 else 1
    for m in (1, 2, 2.5, 5, 10):
        if ymax / (step * m) <= 6:
            step *= m
            break
    ymax = math.ceil(ymax / step) * step or 1
    X = lambda x: L + (math.log2(x) - x0) / (x1 - x0) * (W - L - R)
    Y = lambda y: H - B - y / ymax * (H - T - B)
    g = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">']
    y = 0.0
    while y <= ymax + 1e-9:
        g.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(y):.1f}" y2="{Y(y):.1f}"/>')
        g.append(f'<text class="tick" x="{L - 8}" y="{Y(y) + 4:.1f}" text-anchor="end">{fmt(y, 1 if step < 1 else 0)}</text>')
        y += step
    for x in xs:
        lab = f"{x // 1024}K" if x >= 1024 else str(x)
        g.append(f'<line class="grid v" x1="{X(x):.1f}" x2="{X(x):.1f}" y1="{T}" y2="{H - B}"/>')
        g.append(f'<text class="tick" x="{X(x):.1f}" y="{H - B + 18}" text-anchor="middle">{lab}</text>')
    g.append(f'<text class="axis" x="{(L + W - R) / 2}" y="{H - 8}" text-anchor="middle">上下文长度（token）</text>')
    g.append(f'<text class="axis" x="14" y="{(T + H - B) / 2}" text-anchor="middle" '
             f'transform="rotate(-90 14 {(T + H - B) / 2})">{html.escape(unit)}</text>')
    for label, color, s in series:
        s = [(x, y) for x, y in s if y is not None]
        if not s:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(s))
        g.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in s:
            g.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" fill="{color}"><title>{html.escape(label)}: '
                     f'{x} token, {fmt(y, 2)} {html.escape(unit)}</title></circle>')
            g.append(f'<text class="val" x="{X(x):.1f}" y="{Y(y) - 9:.1f}" text-anchor="middle" fill="{color}">'
                     f'{fmt(y, 1)}</text>')
    g.append("</svg>")
    legend = "".join(f'<span class="key"><i style="background:{c}"></i>{html.escape(l)}</span>' for l, c, _ in series)
    return f'<figure><figcaption>{html.escape(title)}</figcaption><div class="legend">{legend}</div>{"".join(g)}</figure>'


CSS = """
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#57534e;--line:#e7e5e4;--card:#ffffff;--accent:#2563eb;--warn:#b45309;
--ok:#047857;--bad:#b91c1c;--grid:#e7e5e4;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0c0a09;--fg:#e7e5e4;--muted:#a8a29e;
--line:#292524;--card:#1c1917;--accent:#60a5fa;--warn:#fbbf24;--ok:#34d399;--bad:#f87171;--grid:#292524;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0c0a09;--fg:#e7e5e4;--muted:#a8a29e;--line:#292524;--card:#1c1917;--accent:#60a5fa;
--warn:#fbbf24;--ok:#34d399;--bad:#f87171;--grid:#292524;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",
"PingFang SC","Hiragino Sans GB","Microsoft YaHei","Noto Sans CJK SC",Roboto,Helvetica,Arial,sans-serif}
main{max-width:960px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 10px;padding-top:8px;border-top:1px solid var(--line)}
.meta{color:var(--muted);font-size:13px}
.summary{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin:18px 0}
.summary li{margin:4px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card h3{margin:0 0 6px;font-size:15px}.card .big{font-size:22px;font-weight:650}.card .sub{color:var(--muted);font-size:12px}
figure{margin:18px 0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
figcaption{font-weight:600;margin-bottom:4px}
svg{width:100%;height:auto;display:block}
svg .grid{stroke:var(--grid);stroke-width:1}svg .tick,svg .axis{fill:var(--muted);font-size:12px}
svg .val{font-size:11px;font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:var(--muted)}
.key i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px;background:var(--card)}
th,td{border:1px solid var(--line);padding:6px 8px;text-align:right;white-space:nowrap}
th{background:var(--bg);text-align:center}td:first-child,th:first-child{text-align:left}
.tag{display:inline-block;font-size:11px;padding:1px 7px;border-radius:9px;border:1px solid currentColor}
.real{color:var(--ok)}.est{color:var(--warn)}.miss{color:var(--bad)}
.chart-empty{color:var(--muted);padding:12px;border:1px dashed var(--line);border-radius:10px;margin:12px 0}
ul{padding-left:20px}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    machines = []
    for i, m in enumerate(cfg["machines"]):
        info = {}
        if m.get("machine_json"):
            info = json.load(open(m["machine_json"]))
        rows = read_tsv(m["context_tsv"]) if m.get("context_tsv") else []
        ver = read_verify(m["verify_log"]) if m.get("verify_log") else None
        machines.append({"cfg": m, "info": info, "rows": rows, "verify": ver, "color": COLORS[i % len(COLORS)]})

    def spec(mm):
        if mm["cfg"].get("static"):
            s = mm["cfg"]["static"]
            return s.get("cpu", ""), s.get("gpu", ""), s.get("ram", "")
        i = mm["info"]
        ram = f'{i.get("mem_total_gib", 0):.0f} GiB'
        if i.get("cgroup_memory_max") and i["cgroup_memory_max"].strip().isdigit():
            ram += f' (容器上限 {int(i["cgroup_memory_max"]) / 2**30:.0f} GiB)'
        ram = mm["cfg"].get("ram", ram)   # the config overrides it (container limits machine.json cannot see)
        return i.get("cpu_model", ""), f'{i.get("gpu_name", "")} {i.get("gpu_mem_gib", 0):.0f} GB', ram

    out = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width,initial-scale=1">',
           f'<title>{html.escape(cfg.get("short_title", "M3 测试报告"))}</title><style>{CSS}</style></head><body><main>',
           f'<h1>{html.escape(cfg["title"])}</h1><div class="meta">{html.escape(cfg.get("date", ""))}'
           f' · {html.escape(cfg.get("subtitle", ""))}</div>']
    if cfg.get("summary"):
        out.append('<div class="summary"><strong>一页结论</strong><ul>' +
                   "".join(f"<li>{s}</li>" for s in cfg["summary"]) + "</ul></div>")
    # cards
    out.append('<h2>机器</h2><div class="cards">')
    for mm in machines:
        cpu, gpu, ram = spec(mm)
        rows = mm["rows"]
        best = max((r["prefill_tok_s"] or 0 for r in rows), default=0)
        dec = [r["decode_tok_s"] for r in rows if r["decode_tok_s"]]
        if not rows and mm["cfg"].get("decode_only"):
            dec = [d for _, d in mm["cfg"]["decode_only"]]
        med = sorted(dec)[len(dec) // 2] if dec else None
        out.append(f'<div class="card" style="border-top:3px solid {mm["color"]}"><h3>{html.escape(mm["cfg"]["label"])}</h3>'
                   f'<div class="sub">{html.escape(cpu)}<br>{html.escape(gpu)} · 内存 {html.escape(ram)}</div>'
                   f'<div style="margin-top:8px"><span class="big">{fmt(best) if best else "–"}</span> tok/s prefill 峰值'
                   f'<br><span class="big">{fmt(med, 2) if med else "–"}</span> tok/s decode（各上下文的中位数）</div></div>')
    out.append("</div>")
    # charts
    out.append("<h2>不同上下文下的速度</h2>")
    out.append(chart([(mm["cfg"]["label"], mm["color"], [(r["context"], r["prefill_tok_s"]) for r in mm["rows"]])
                      for mm in machines], "prefill_tok_s", "Prefill 速度（越高越好）", "token / 秒"))
    out.append(chart([(mm["cfg"]["label"], mm["color"],
                       [(r["context"], r["decode_tok_s"]) for r in mm["rows"]] or
                       [(x, y) for x, y in mm["cfg"].get("decode_only", [])]) for mm in machines],
                     "decode_tok_s", "Decode 速度（紧接 prefill 生成 64 个 token，越高越好）", "token / 秒"))
    out.append(chart([(mm["cfg"]["label"], mm["color"],
                       [(r["context"], (r["prefill_ms"] or 0) / 1000 or None) for r in mm["rows"]]) for mm in machines],
                     "prefill_s", "处理完整个提示词所需时间（越低越好）", "秒"))
    # tables
    out.append("<h2>全部数字</h2>")
    for mm in machines:
        if not mm["rows"]:
            continue
        out.append(f'<h3>{html.escape(mm["cfg"]["label"])}</h3><div class="tablewrap"><table><tr><th>上下文</th>'
                   '<th>prefill 用时 (s)</th><th>prefill tok/s</th><th>decode ms/token</th><th>decode tok/s</th>'
                   '<th>每段 token</th><th>流式搬运的专家</th><th>其中读 SSD</th><th>等待搬运 (s)</th></tr>')
        for r in mm["rows"]:
            out.append(f'<tr><td>{r["context"]:,}</td><td>{fmt((r["prefill_ms"] or 0) / 1000, 1)}</td>'
                       f'<td>{fmt(r["prefill_tok_s"])}</td><td>{fmt(r["decode_ms"])}</td><td>{fmt(r["decode_tok_s"], 2)}</td>'
                       f'<td>{r["chunk"]}</td><td>{r["streamed"]}</td><td>{r["ssd"]}</td>'
                       f'<td>{fmt((r["wait_ms"] or 0) / 1000, 1)}</td></tr>')
        out.append("</table></div>")
        for n in mm["cfg"].get("notes", []):
            out.append(f'<p class="meta">{n}</p>')
    # verification
    vs = [mm for mm in machines if mm["verify"]]
    if vs:
        out.append("<h2>正确性检查</h2>")
        for mm in vs:
            v = mm["verify"]
            out.append(f'<p>{html.escape(mm["cfg"]["label"])}：5 篇文档的平均 nll（越低越好）。参照是 DeepSeek 官方 model.py '
                       '在 FP16 专家上的结果。「逐 token」是 decode 路径（CPU 专家用 int8 激活）；「prefill」是一次处理'
                       '整篇；「61 一段」故意把文档切成 61 个 token 一段，检查段与段之间的状态衔接。</p>')
            out.append('<div class="tablewrap"><table><tr><th>文档</th><th>token</th><th>原型 FP16</th><th>逐 token</th>'
                       '<th>prefill</th><th>61 一段</th></tr>')
            for doc, n, ref, st, pf, ck in v["nll"]:
                out.append(f"<tr><td>{doc}</td><td>{n}</td><td>{ref}</td><td>{st}</td><td>{pf}</td><td>{ck}</td></tr>")
            out.append("</table></div>")
            if v["long"]:
                n, one, ck = v["long"]
                out.append(f"<p>长文本 {n} token：一段处理 nll {one}，每 999 个 token 一段 nll {ck}。</p>")
            seqs = [re.findall(r"generated:((?: \d+)+)", g) for g in v["gen"]]
            if len(seqs) == 2 and all(seqs):
                a_ids, b_ids = seqs[0][0].split(), seqs[1][0].split()
                same = next((i for i, (x, y) in enumerate(zip(a_ids, b_ids)) if x != y), min(len(a_ids), len(b_ids)))
                out.append(f'<p>生成：同一个 300 token 的提示词，逐 token 处理后生成和 prefill 后生成，前 {same} 个 token 完全相同'
                           f'（共 {len(b_ids)} 个），之后因舍入差异分叉。</p>')
    if cfg.get("notes"):
        out.append("<h2>说明</h2><ul>" + "".join(f"<li>{n}</li>" for n in cfg["notes"]) + "</ul>")
    out.append("</main></body></html>")
    with open(a.out, "w") as f:
        f.write("\n".join(out))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
