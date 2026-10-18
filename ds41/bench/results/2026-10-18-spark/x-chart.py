"""The X chart: DeepSeek-V4.1-Flash-1017 on one DGX Spark. Every number comes from data.json (measured runs)."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(sys.argv[1]))
out = sys.argv[2]

BG, FG, MUTED, ACC, ACC2, GRID = "#0d1117", "#e6edf3", "#8b949e", "#76b900", "#3fb6ff", "#21262d"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})

fig = plt.figure(figsize=(16, 9), dpi=150, facecolor=BG)
fig.text(0.04, 0.925, "DeepSeek-V4.1-Flash-1017 on a single NVIDIA DGX Spark", fontsize=30, fontweight="bold")
fig.text(0.04, 0.875, d["subtitle"], fontsize=15, color=MUTED)

# left: decode speed by context length
ax = fig.add_axes([0.05, 0.17, 0.37, 0.6], facecolor=BG)
labels = [c["label"] for c in d["decode"]]
vals = [c["tok_s"] for c in d["decode"]]
bars = ax.bar(labels, vals, color=ACC, width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=17, fontweight="bold")
ax.set_ylim(0, max(vals) * 1.22)
ax.set_title("Decode speed (tokens/s) vs context length", fontsize=17, loc="left", pad=14)
ax.set_xlabel("context (tokens)", fontsize=13, color=MUTED)
ax.grid(axis="y", color=GRID)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="x", labelsize=14)

# right: time to first token of an agent turn
ax2 = fig.add_axes([0.645, 0.17, 0.315, 0.6], facecolor=BG)
rows = d["ttft"]
names = [r["label"] for r in rows][::-1]
secs = [r["s"] for r in rows][::-1]
colors = [ACC2 if r.get("highlight") else "#2f81f7" for r in rows][::-1]
hb = ax2.barh(names, secs, color=colors, height=0.6)
for b, v in zip(hb, secs):
    ax2.text(v + max(secs) * 0.02, b.get_y() + b.get_height() / 2, f"{v:.2f} s" if v < 1 else f"{v:.1f} s",
             va="center", fontsize=15, fontweight="bold")
ax2.set_xlim(0, max(secs) * 1.25)
ax2.set_title("Time to first token, next agent turn", fontsize=17, loc="left", pad=14)
ax2.set_xlabel("seconds", fontsize=13, color=MUTED)
ax2.grid(axis="x", color=GRID)
ax2.set_axisbelow(True)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.tick_params(axis="y", labelsize=13, colors=FG)

fig.text(0.04, 0.065, d["footer1"], fontsize=12.5, color=MUTED)
fig.text(0.04, 0.035, d["footer2"], fontsize=12.5, color=MUTED)
fig.savefig(out, facecolor=BG)
print("wrote", out)
