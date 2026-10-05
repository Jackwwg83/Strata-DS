"""Render the session-2 report (routing + correctness) as one self-contained HTML file.

Usage: python make_routing_report.py <run_dir> <out.html>
Numbers come from <run_dir>/analysis.json, docs.jsonl and generations.jsonl.
"""
import html
import json
import os
import sys
from collections import defaultdict

CSS = """
:root{--bg:#fbfbf9;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--card:#fff;--acc:#2f5d8a;--warn:#9a5b00}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161615;--fg:#e9e8e3;--mut:#9b9a94;--line:#33332f;--card:#1f1f1d;--acc:#8db7e0;--warn:#e0b060}}
:root[data-theme=dark]{--bg:#161615;--fg:#e9e8e3;--mut:#9b9a94;--line:#33332f;--card:#1f1f1d;--acc:#8db7e0;--warn:#e0b060}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei","Noto Sans CJK SC",sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 8px;color:var(--acc)}h3{font-size:15px;margin:18px 0 6px}
.sub,.note{color:var(--mut);font-size:13px}.warn{color:var(--warn)}
.tw{overflow-x:auto;margin:8px 0;border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;background:color-mix(in srgb,var(--acc) 8%,transparent);white-space:nowrap}
tr:last-child td{border-bottom:0}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:8px 0}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px}
"""


def table(headers, rows):
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def pct(x):
    return f"{x * 100:.1f}%"


def main():
    run, out_path = sys.argv[1], sys.argv[2]
    a = json.load(open(os.path.join(run, "analysis.json")))
    docs = [json.loads(l) for l in open(os.path.join(run, "docs.jsonl"))]
    gens = [json.loads(l) for l in open(os.path.join(run, "generations.jsonl"))]
    parts = []

    by_kind = defaultdict(list)
    for d in docs:
        by_kind[d["kind"]].append(d)
    kind_name = {"code_py": "Python 代码", "code_cpp": "C++/CUDA 代码", "zh": "中文技术文档",
                 "en": "英文技术文档", "chat": "聊天格式的代码提问"}
    rows = []
    for k, ds in by_kind.items():
        p = [d["ppl"] for d in ds]
        rows.append([kind_name.get(k, k), len(ds), sum(d["tokens"] for d in ds),
                     f"{min(p):.2f} – {max(p):.2f}", f"{sum(d['seconds'] for d in ds) / len(ds):.1f}"])
    parts.append("<h2>1. 正确性：没有对照模型时的两条证据</h2>")
    parts.append("<h3>逐 token 困惑度（teacher forcing，每篇最多 4096 token）</h3>")
    parts.append(table(["类型", "篇数", "token 数", "困惑度范围", "每篇平均秒数（原型）"], rows))
    parts.append('<p class="note">代码约 2–3.4、英文约 4.5–6.8：同类模型在这类文本上的正常水平。中文 12–28 偏高，'
                 "原因判断为：中文 token 粒度更粗，且这些文档是本项目自己的、充满数字和缩写的设计文档；"
                 "中文生成输出通顺，可排除中文路径的 bug。这些是合理性检查，不是与原版模型的对比。</p>")
    rows = [[html.escape(g["id"]), g["prompt_tokens"], g["new_tokens"],
             html.escape(g["output"][:160]).replace("\n", " ⏎ ") + ("…" if len(g["output"]) > 160 else "")]
            for g in gens]
    parts.append("<h3>贪心生成（chat 模式，最多 128 token）</h3>")
    parts.append(table(["任务", "prompt token", "生成 token", "输出开头"], rows))

    parts.append("<h2>2. 专家热度：偏斜明显</h2>")
    s = a["skew_all"]
    parts.append(table(["最热的 (层, 专家) 对", "占全部访问"],
                       [["前 1%", pct(s["top_1pct_pairs_share"])], ["前 5%", pct(s["top_5pct_pairs_share"])],
                        ["前 10%", pct(s["top_10pct_pairs_share"])], ["前 20%", pct(s["top_20pct_pairs_share"])],
                        ["前 50%", pct(s["top_50pct_pairs_share"])]]))
    parts.append(f'<p class="note">共 {a["tokens_total"]:,} 个 token、{len(a["sequences"])} 条序列；'
                 f'15,360 个专家对里有 {a["pairs_never_used"]} 个一次都没被用到。每层前 10% 专家的访问占比中位数 '
                 f'{pct(a["skew_per_layer_top10pct_median"])}。均匀分布时前 10% 只占 10%。</p>')

    c = a["static_hit_curve"]
    ad = a["adaptive_vs_static"]
    rows = [[k, pct(v)] for k, v in c.items()]
    parts.append("<h3>静态热度档案的命中率（留一法：用其余序列建档案，测剩下那条）</h3>")
    parts.append(table(["显存里放的专家数", "命中率"], rows))
    parts.append("<h3>上游式自适应缓存 vs 静态（衰减计数 0.7、每 4 token 换一次、每次最多换 96 个）</h3>")
    parts.append(table(["显存专家数", "静态", "自适应"],
                       [[k, pct(v["static"]), pct(v["adaptive"])] for k, v in ad.items()]))
    parts.append('<p class="note">上游 Strata 在 Qwen 上的数字：4,500 个专家（占 18%）时静态 50%、自适应 72%。'
                 "DeepSeek 在 850 个专家（占 5.5%）时就有 34% / 53%，热度集中程度比预想的好。</p>")

    parts.append("<h2>3. 128 GB 内存下的三层分配与 decode 速度推算</h2>")
    parts.append('<p class="note">放置规则：最热的专家放显存，其次放内存（专家区约 95 GiB），最冷的放 SSD。'
                 "速度公式：GPU（稠密 8.9 ms + 显存专家）与 CPU（内存专家）并行，取慢者，再加 SSD 读取。"
                 "常数来自第一段实测（7950X：CPU 专家 46 GB/s；NVMe 6.8 GB/s；4090 显存 954 GB/s；"
                 "GPU 单专家 40 µs，未融合）。<b>这是推算，不含投机解码，也没算注意力计算和调度开销。</b></p>")
    rows = []
    for key in a["tiers_128gb"]:
        st, at = a["tiers_128gb"][key], a["tiers_128gb_adaptive"][key]
        es, ea = a["decode_estimate_no_spec"][key], a["decode_estimate_no_spec_adaptive"][key]
        bpw, gpu = key.split("bpw_")
        rows.append([bpw, gpu, f'{st["vram_experts"]:,}', pct(at["vram_share"]), pct(at["ram_share"]),
                     pct(at["ssd_share"]), f'{ea["cpu_ms"]} / {ea["gpu_ms"]} / {ea["ssd_ms"]}',
                     f'<b>{ea["tok_s"]}</b>', es["tok_s"]])
    parts.append(table(["位宽", "显卡", "显存放得下的专家", "显存命中", "内存", "SSD",
                        "CPU / GPU / SSD 毫秒", "推算 tok/s（自适应）", "（静态）"], rows))

    w = a["window_distinct_ratio"]
    parts.append("<h2>4. 投机解码窗口里的专家复用</h2>")
    parts.append(table(["窗口 token 数", "每层不同专家数 ÷（6 × 窗口）"], [[k, f"{v:.2f}"] for k, v in w.items()]))
    parts.append('<p class="note">窗口 6 个 token（DSpark 一次验证 6 个位置）时，CPU 要算的专家是单 token 的约 '
                 f'{6 * float(w.get("6", w.get(6, 0))):.1f} 倍。若平均每次只接受 3 个 token，CPU 侧每 token 的成本反而上升。'
                 "所以在 CPU 受限的配置上，投机解码未必提速；收益主要在显存命中高、GPU 受限的配置。</p>")

    parts.append("<h2>5. 局限</h2><ul>"
                 "<li>语料小：28 篇文档（约 10 万 token，teacher forcing）+ 6 个生成任务（约 700 个 decode token）。"
                 "大部分路由来自 prefill，用来近似逐 token 的 decode 顺序。</li>"
                 "<li>留一法里，其余文档和被测文档同类，静态档案的命中率偏乐观；真实用户的混合使用会低一些。</li>"
                 "<li>路由来自 EXL3 3bpw 模型；2bpw / 1.59bpw 模型的路由会略有不同，这里假设热度结构相同。</li>"
                 "<li>GPU 专家耗时用的是未融合的 40 µs；融合 kernel + CUDA graph 后 GPU 侧会更快。</li>"
                 "<li>第一轮采集用了官方 TileLang kernel，act_quant 产生 NaN，数据已作废（保留在 full_tilelang_nan/）。"
                 "本报告全部数据来自 PyTorch 版 kernel 的重跑。</li></ul>")

    doc = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Strata-DS 路由实测</title>
<link rel="icon" href="data:,"><style>{CSS}</style></head><body><main>
<h1>Strata-DS 路由实测：DeepSeek V4.1 Flash 单卡 4090</h1>
<p class="sub">2026-10-05 · Vast 实例 54270567（RTX 4090 24 GB）· 模型 coolbho3k EXL3 3bpw <code>650cae2c</code> ·
官方 <code>inference/model.py</code> 原样导入 · 原始数据在 <code>full/routes/</code>，分析在 <code>full/analysis.json</code></p>
{''.join(parts)}
</main></body></html>"""
    open(out_path, "w").write(doc)
    print(out_path)


if __name__ == "__main__":
    main()
