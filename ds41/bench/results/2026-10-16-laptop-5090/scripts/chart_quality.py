#!/usr/bin/env python3
"""The quality chart: task accuracy without and with the tail skips, and KL divergence per text kind (noise floor =
two runs without skips vs each other; default vs no skip). Usage: chart_quality.py data.json OUTDIR en|zh"""
import json, sys, html

d, out, lang = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
ZH = lang == "zh"
FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', "
        "'Noto Sans CJK SC', Roboto, Helvetica, Arial, sans-serif")
INK, MUTED, GRID = "#1f2328", "#59636e", "#d8dee4"
C_A, C_C, C_N = "#8c959f", "#2f6fde", "#c9d1d9"
e = lambda s: html.escape(str(s), quote=True)


def text(x, y, s, size=13, anchor="start", weight=400, fill=INK):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">{e(s)}</text>'


def group(x0, y0, w, h, labels, a, c, vmax, ticks, fmt, names, colors, lsize=11):
    o = []
    for v in ticks:
        y = y0 + h - h * v / vmax
        o.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}"/>')
        o.append(text(x0 - 8, y + 4, fmt(v, True), 11, "end", fill=MUTED))
    gw = w / len(labels)
    bw = min(34, gw * 0.42)
    for i, lab in enumerate(labels):
        cx = x0 + gw * (i + 0.5)
        for j, (vals, col) in enumerate(((a, colors[0]), (c, colors[1]))):
            v = vals[i]
            if v is None:
                continue
            bx = cx - bw + bw * j
            bh = h * v / vmax
            o.append(f'<rect x="{bx + 1:.1f}" y="{y0 + h - bh:.1f}" width="{bw - 2:.1f}" height="{bh:.1f}" rx="2" fill="{col}"/>')
            o.append(text(bx + bw / 2, y0 + h - bh - 5, fmt(v, False), lsize, "middle", 600))
        for k, line in enumerate(lab.split("\n")):
            o.append(text(cx, y0 + h + 18 + 15 * k, line, 12, "middle", fill=MUTED))
    lx = x0
    for name, col in zip(names, colors):
        o.append(f'<rect x="{lx}" y="{y0 - 30}" width="12" height="12" rx="2" fill="{col}"/>' + text(lx + 18, y0 - 20, name, 12))
        lx += 18 + len(name) * (13 if ZH else 7) + 22
    return "".join(o)


t = d["tasks"]
title = "跳过专家的代价：任务准确率几乎不变，文本分布有可测的偏移" if ZH else \
    "What the expert skips cost: task accuracy holds, the token distribution shifts measurably"
body = text(40, 40, title, 19, weight=700)
body += text(40, 64, d["sub_zh"] if ZH else d["sub_en"], 12, fill=MUTED)
body += text(40, 104, "任务准确率（%），同一批题配对比较" if ZH else "Task accuracy (%), the same items paired", 14, weight=600)
labels = [x["zh" if ZH else "en"] for x in t]
body += group(70, 150, 400, 230, labels, [x["a"] for x in t], [x["c"] for x in t], 100, range(0, 101, 20),
              lambda v, tick: f"{v:.0f}" if tick else f"{v:.1f}", ["不跳过" if ZH else "no skip", "默认（跳过）" if ZH else "default (skips)"],
              [C_A, C_C])
k = d["kl"]
body += text(530, 104, "KL 散度（相对不跳过），按文本类别" if ZH else "KL divergence from no skip, by text kind", 14, weight=600)
klabels = [x["zh" if ZH else "en"] for x in k]
vmax = 0.06
body += group(580, 150, 360, 230, klabels, [x["floor"] for x in k], [x["c"] for x in k], vmax,
              [0, 0.02, 0.04, 0.06], lambda v, tick: f"{v:.2f}" if tick else f"{v:.3f}",
              ["噪声底线（两次不跳过）" if ZH else "noise floor (no skip twice)", "默认" if ZH else "default"], [C_N, C_C], 9.5)
body += text(40, 452, d["note_zh"] if ZH else d["note_en"], 11, fill=MUTED)
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 470" width="980" height="470" role="img" '
       f'aria-label="{e(title)}"><title>{e(title)}</title><rect width="980" height="470" fill="#fff"/>'
       f'<g font-family="{e(FONT)}" fill="{INK}">{body}</g></svg>\n')
open(f"{out}/deepseek-laptop-quality{'-zh' if ZH else ''}.svg", "w").write(svg)
print("ok", lang)
