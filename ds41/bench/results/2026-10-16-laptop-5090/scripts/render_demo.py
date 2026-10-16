#!/usr/bin/env python3
"""A real-speed video of a recorded live run: every piece appears at the time it arrived (demo_rec.py), with a
tokens/s counter from the same timestamps and the GPU's power and temperature from the 1-second samples (tele.py).
Nothing is sped up or cut. Usage: render_demo.py DEMO.json TELE.tsv OUT.mp4 en|zh [SECONDS]"""
import json, subprocess, sys, bisect
from PIL import Image, ImageDraw, ImageFont

demo, tele_path, out, lang = sys.argv[1:5]
SECONDS = float(sys.argv[5]) if len(sys.argv) > 5 else 60
ZH = lang == "zh"
W, H, FPS = 1280, 720, 30
d = json.load(open(demo))
t0 = d["started"]
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"
f_text, f_small, f_big, f_mid = (ImageFont.truetype(FONT, s) for s in (19, 15, 46, 22))
BG, INK, DIM, ACC, LINE = (13, 17, 23), (230, 237, 243), (125, 133, 144), (121, 192, 255), (48, 54, 61)

tele = [l.rstrip("\n").split("\t") for l in open(tele_path)]
th, trows = tele[0], tele[1:]
ti = {c: i for i, c in enumerate(th)}
tt = [float(r[ti["t"]]) for r in trows]


def hw(at):
    j = max(0, min(len(trows) - 1, bisect.bisect_right(tt, at) - 1))
    r = trows[j]
    return float(r[ti["gpu_w"]]), float(r[ti["gpu_c"]])


# the text as shown: the question, then the reasoning (dim), then the answer; per char its color and arrival time
user = ("你：" if ZH else "You: ") + d["prompt"].strip()
chars, colors, times = [], [], []
for ch in user:
    chars.append(ch); colors.append(ACC); times.append(0.0)
for ch in "\n\n":
    chars.append(ch); colors.append(DIM); times.append(0.0)
answer_started = False
for t, kind, piece in d["pieces"]:
    if kind == "a" and not answer_started:
        answer_started = True
        for ch in "\n\n":
            chars.append(ch); colors.append(INK); times.append(t)
    for ch in piece:
        chars.append(ch); colors.append(DIM if kind == "r" else INK); times.append(t)

# wrap once: line = list of char indices
TX, TY, TW, TH, LH = 40, 96, 860, 560, 27
lines, cur, width = [], [], 0
for i, ch in enumerate(chars):
    if ch == "\n":
        lines.append(cur); cur, width = [], 0
        continue
    w = f_text.getlength(ch)
    if width + w > TW and cur:
        # English: break after the last space of the line; CJK and long words: break here
        sp = max((j for j, c in enumerate(cur) if chars[c] == " "), default=-1)
        if sp > len(cur) // 2:
            lines.append(cur[:sp + 1]); cur = cur[sp + 1:]
        else:
            lines.append(cur); cur = []
        width = sum(f_text.getlength(chars[c]) for c in cur)
    cur.append(i); width += w
lines.append(cur)
first_char_line = [l[0] if l else None for l in lines]
pt = [p[0] for p in d["pieces"]]
title = ("DeepSeek V4.1 Flash（7630 亿参数）· RTX 5090 笔记本 24 GB · 64 GB 内存" if ZH else
         "DeepSeek V4.1 Flash (763B) · RTX 5090 Laptop 24 GB · 64 GB RAM")
foot = ("根据 2026-10-16 一次实时运行（关闭思考）的 token 时间戳按真实速度渲染，未加速、未剪辑。" if ZH else
        "Rendered at real speed from the token timestamps of a live run on 2026-10-16 (thinking off). "
        "Not sped up, not cut.")
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                       "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out],
                      stdin=subprocess.PIPE)
for frame in range(int(SECONDS * FPS)):
    now = frame / FPS
    k = bisect.bisect_right(times, now)            # chars visible
    n = bisect.bisect_right(pt, now)                # pieces (about one token each) so far
    img = Image.new("RGB", (W, H), BG)
    g = ImageDraw.Draw(img)
    g.text((40, 28), title, font=f_mid, fill=INK)
    g.line([(40, 72), (W - 40, 72)], fill=LINE, width=1)
    vis = [ln for ln in lines if ln and ln[0] < k or not ln]
    # lines up to the last visible char
    last = max([j for j, ln in enumerate(lines) if ln and ln[0] < k] or [0])
    rows = lines[:last + 1]
    rows = rows[-(TH // LH):]
    y = TY
    for ln in rows:
        x = TX
        for i in ln:
            if i >= k:
                break
            g.text((x, y), chars[i], font=f_text, fill=colors[i])
            x += f_text.getlength(chars[i])
        y += LH
    if k < len(chars) and int(now * 2) % 2 == 0:  # cursor
        g.rectangle([x + 2, y - LH + 4, x + 11, y - 4], fill=ACC)
    # the panel
    px = 940
    g.line([(px - 20, 96), (px - 20, 656)], fill=LINE, width=1)
    recent = (n - bisect.bisect_right(pt, now - 2.0)) / 2.0 if n else 0.0
    avg = (n - 1) / (pt[n - 1] - pt[0]) if n > 1 and pt[n - 1] > pt[0] else 0.0
    w, c = hw(t0 + now)
    items = [("tokens/s" if not ZH else "token/s（最近 2 秒）", f"{recent:.0f}"),
             ("average tokens/s" if not ZH else "平均 token/s", f"{avg:.1f}"),
             ("tokens" if not ZH else "已生成 token", f"{n}"),
             ("seconds" if not ZH else "已用时间（秒）", f"{now:.1f}"),
             ("GPU", f"{w:.0f} W · {c:.0f} C")]
    y = 100
    for label, val in items:
        g.text((px, y), label, font=f_small, fill=DIM)
        g.text((px, y + 20), val, font=f_big if label != "GPU" else f_mid, fill=ACC if y == 100 else INK)
        y += 108 if label != "GPU" else 60
    g.text((40, 676), foot, font=f_small, fill=DIM)
    ff.stdin.write(img.tobytes())
ff.stdin.close()
ff.wait()
print("wrote", out)
