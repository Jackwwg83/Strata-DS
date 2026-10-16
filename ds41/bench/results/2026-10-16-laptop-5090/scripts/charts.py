#!/usr/bin/env python3
"""SVG charts for the laptop recipe (stdlib only). Input: a JSON file with the measured numbers.
Usage: charts.py data.json OUTDIR LANG   (LANG = en | zh)"""
import json, sys, html

data, out, lang = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
ZH = lang == "zh"
FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', "
        "'Noto Sans CJK SC', Roboto, Helvetica, Arial, sans-serif")
INK, MUTED, GRID, BG = "#1f2328", "#59636e", "#d8dee4", "#ffffff"
C1, C2, C3 = "#2f6fde", "#e8833a", "#8a63d2"


def esc(s):
    return html.escape(str(s), quote=True)


def svg(w, h, title, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="{esc(title)}"><title>{esc(title)}</title>'
            f'<rect width="{w}" height="{h}" fill="{BG}"/>'
            f'<g font-family="{esc(FONT)}" fill="{INK}">{body}</g></svg>\n')


def text(x, y, s, size=14, anchor="start", weight=400, fill=INK):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" '
            f'fill="{fill}">{esc(s)}</text>')


def bars(x0, y0, w, h, labels, series, vmax, step, unit_fmt, colors, legend):
    """Grouped vertical bars. series: list of value lists (None = no bar)."""
    o = []
    for v in range(0, int(vmax) + 1, step):
        y = y0 + h - h * v / vmax
        o.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(text(x0 - 8, y + 4, v, 12, "end", fill=MUTED))
    n, k = len(labels), len(series)
    gw = w / n
    bw = min(56, gw * 0.7 / k)
    for i, lab in enumerate(labels):
        cx = x0 + gw * (i + 0.5)
        for j, s in enumerate(series):
            v = s[i]
            if v is None:
                continue
            bx = cx - bw * k / 2 + bw * j
            bh = h * v / vmax
            o.append(f'<rect x="{bx + 2:.1f}" y="{y0 + h - bh:.1f}" width="{bw - 4:.1f}" height="{bh:.1f}" rx="3" '
                     f'fill="{colors[j]}"/>')
            o.append(text(bx + bw / 2, y0 + h - bh - 6, unit_fmt(v), 13, "middle", 600))
        o.append(text(cx, y0 + h + 20, lab, 13, "middle", fill=MUTED))
    lx = x0
    for j, name in enumerate(legend):
        o.append(f'<rect x="{lx}" y="{y0 - 34}" width="12" height="12" rx="2" fill="{colors[j]}"/>')
        o.append(text(lx + 18, y0 - 24, name, 13))
        lx += 18 + 9 * len(name) * (1.6 if ZH else 1.0) + 24
    return "".join(o)


# 1. decode and prefill by context length
ctx = data["context"]
labels = [c["label"] for c in ctx]
dec = [c["decode_tok_s"] for c in ctx]
pre = [c["prefill_tok_s"] for c in ctx]
t1 = "上下文越长，生成多快、读入多快" if ZH else "Speed by context length"
sub = (data["machine_zh"] if ZH else data["machine_en"])
body = text(40, 44, t1, 22, weight=700) + text(40, 70, sub, 13, fill=MUTED)
body += text(40, 112, "生成（decode），token/s" if ZH else "Writing the answer (decode), tokens/s", 15, weight=600)
body += bars(80, 160, 380, 250, labels, [dec], max(35, int(max(dec) / 5 + 1) * 5), 5, lambda v: f"{v:.1f}", [C1],
             [])
body += text(520, 112, "读入提示词（prefill），token/s" if ZH else "Reading the prompt (prefill), tokens/s", 15,
             weight=600)
body += bars(560, 160, 380, 250, labels, [pre], max(1000, int(max(pre) / 200 + 1) * 200), 200, lambda v: f"{v:.0f}",
             [C2], [])
note = ("每个长度一条合成提示词，生成 256 个 token；生成速度含所有步" if ZH else
        "One synthetic prompt per length, 256 tokens written; decode includes every step")
body += text(40, 470, note, 12, fill=MUTED)
open(f"{out}/deepseek-laptop-context{'-zh' if ZH else ''}.svg", "w").write(svg(980, 490, t1, body))

# 2. the week: short-prompt speed after each change
wk = data["week"]
t2 = "一周内，同一台笔记本：短提示词生成速度" if ZH else "One laptop, one week: short-prompt decode speed"
body = text(40, 44, t2, 22, weight=700) + text(40, 70, "code 提示词，256 个 token，token/s" if ZH else
                                                "code prompt, 256 tokens, tokens/s", 13, fill=MUTED)
x0, y0, w, h = 60, 110, 880, 250
vmax = 35
o = []
for v in range(0, vmax + 1, 5):
    y = y0 + h - h * v / vmax
    o.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}"/>')
    o.append(text(x0 - 8, y + 4, v, 12, "end", fill=MUTED))
gw = w / len(wk)
for i, s in enumerate(wk):
    cx = x0 + gw * (i + 0.5)
    bh = h * s["tok_s"] / vmax
    col = C3 if s.get("system") else C1
    o.append(f'<rect x="{cx - 34:.1f}" y="{y0 + h - bh:.1f}" width="68" height="{bh:.1f}" rx="3" fill="{col}"/>')
    o.append(text(cx, y0 + h - bh - 8, f'{s["tok_s"]:.1f}', 15, "middle", 700))
    for k, line in enumerate((s["zh"] if ZH else s["en"]).split("\n")):
        o.append(text(cx, y0 + h + 22 + 17 * k, line, 12, "middle", fill=MUTED))
body += "".join(o)
body += f'<rect x="60" y="440" width="12" height="12" rx="2" fill="{C1}"/>' + text(78, 450, "引擎改动" if ZH else
                                                                                   "engine change", 12)
body += f'<rect x="200" y="440" width="12" height="12" rx="2" fill="{C3}"/>' + text(218, 450, "系统设置（Dynamic Boost）"
                                                                                    if ZH else
                                                                                    "system setting (Dynamic Boost)",
                                                                                    12)
open(f"{out}/deepseek-laptop-week{'-zh' if ZH else ''}.svg", "w").write(svg(980, 470, t2, body))
print("ok", lang)
