"""Build a self-contained HTML report from one results directory.

Usage: python make_report.py <results_dir> [--notes notes.html]
Writes <results_dir>/report.html. Every number in the tables comes from the JSON files in the
directory; the optional notes file holds the hand-written summary and derived estimates.
"""
import argparse
import glob
import html
import json
import os


def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def table(headers, rows):
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def section(title, body, note=""):
    n = f'<p class="note">{note}</p>' if note else ""
    return f"<section><h2>{html.escape(title)}</h2>{n}{body}</section>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--notes")
    a = ap.parse_args()
    d = a.dir
    parts = []

    m = load(os.path.join(d, "machine.json")) or {}
    parts.append(section("机器", table(["项目", "值"], [[k, v] for k, v in m.items()]),
                         "来自 machine_info.sh；完整原始输出见 machine.txt。"))

    rows = []
    for line in open(os.path.join(d, "membw.jsonl")) if os.path.exists(os.path.join(d, "membw.jsonl")) else []:
        try:
            r = json.loads(line)
            rows.append([r["threads"], r["read_gbps"], r["copy_gbps_rw"]])
        except Exception:
            pass
    parts.append(section("内存带宽（8 GiB 数组，取 5 次最好）",
                         table(["线程", "读 GB/s", "拷贝 GB/s（读+写）"], rows)))

    g = load(os.path.join(d, "gpu_pcie.json")) or {}
    if g:
        rows = [[f'{r["bytes"] / 2**20:.1f} MiB', r["h2d_pinned_gbps"], r["d2h_pinned_gbps"]]
                for r in g.get("pcie_by_size", [])]
        body = table(["单次大小", "H2D pinned GB/s", "D2H pinned GB/s"], rows)
        body += table(["指标", "值"], [
            ["H2D pageable 256 MiB GB/s", g.get("h2d_pageable_256m_gbps")],
            ["40 个专家大小的连续 H2D 拷贝 GB/s", g.get("h2d_40_expert_copies_gbps")],
            ["显存拷贝 GB/s（读+写）", g.get("gpu_copy_gbps_rw")],
            ["显存读（fp16 求和）GB/s", g.get("gpu_read_sum_gbps")],
            ["空闲时可用显存 GiB", g.get("gpu_free_gib_idle")],
            ["ulimit -l", g.get("ulimit_memlock")]])
        body += table(["pinned 申请 GiB", "结果", "耗时 s / 错误"],
                      [[r["gib"], "跳过" if "skipped" in r else ("成功" if r.get("ok") else "失败"),
                        r.get("alloc_s", r.get("error", r.get("skipped", "")))] for r in g.get("pinned_alloc", [])])
        parts.append(section("PCIe 与显存", body))

    n = load(os.path.join(d, "nvme_summary.json")) or {}
    if n:
        parts.append(section("NVMe（fio，O_DIRECT，libaio，每项 20 s）",
                             table(["测试", "GB/s", "IOPS", "平均延迟 µs", "p99 延迟 µs"],
                                   [[k, v.get("gbps", v.get("error")), v.get("iops", ""), v.get("lat_mean_us", ""),
                                     v.get("lat_p99_us", "")] for k, v in n.items()]),
                             "13m = 13 MiB 一次（约一个 3bpw 专家），4k = Engram 行所在的页。磁盘是 Vast 容器的 overlay，"
                             "底层设备见 machine.txt。"))

    p = load(os.path.join(d, "expert_parity.json"))
    if p:
        rows = [[k, round(v["rel_l2"], 5), round(v["max_abs"], 5), round(v["cos"], 6), round(v["ref_rms"], 5)]
                for k, v in p.items() if isinstance(v, dict)]
        parts.append(section("CPU kernel 与 GPU 参考的一致性（第 %s 层，4 token × 6 专家）" % p.get("layer"),
                             table(["对比", "相对 L2 误差", "最大绝对误差", "余弦相似度", "参考 RMS"], rows),
                             "两个参考只差路由权重乘在 w2 前还是后（DeepSeek 原版是乘在 w2 前）。两种参考之间的差：%.2e。"
                             % p.get("ref_before_vs_after_rel_l2", 0)))

    gp = load(os.path.join(d, "expert_gpu.json"))
    if gp:
        rows = [[r["path"], r["proj"], r["rows"], r["us"], r.get("weight_gbps", ""), r.get("tflops", "")]
                for r in gp["rows"]]
        parts.append(section("GPU EXL3 kernel（单个投影）",
                             table(["路径", "投影", "行数", "µs", "权重 GB/s", "TFLOPS"], rows),
                             "单个专家、单 token 完整计算：%s µs（%s GB/s）。" %
                             (gp.get("one_expert_one_token_us"), gp.get("one_expert_one_token_gbps"))))

    for f in sorted(glob.glob(os.path.join(d, "expert_cpu*.json"))):
        c = load(f)
        if not c:
            continue
        rows = [[r["threads"], r["tokens"], r["experts_per_token"], r["mode"], r["unique_experts"], r["ms"],
                 r["expert_gbps"], r["us_per_unique_expert"]] for r in c["rows"]]
        name = os.path.basename(f)[:-5]
        note = f'ISA 上限：{c.get("isa_cap")}；swizzle：{c.get("swizzled")}；CPU 标志：{c.get("isa_flags")}'
        if c.get("bg_h2d_gbps") is not None:
            note += f'；后台 H2D 同时跑出 {c["bg_h2d_gbps"]} GB/s'
        parts.append(section(f"CPU 专家 kernel：{name}",
                             table(["线程", "token", "每 token 专家", "模式", "不同专家数", "ms", "专家权重 GB/s",
                                    "每专家 µs"], rows), note))

    notes = open(a.notes).read() if a.notes and os.path.exists(a.notes) else ""
    title = os.path.basename(os.path.abspath(d))
    doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Strata-DS 硬件基线</title><link rel="icon" href="data:,">
<style>
:root{{--bg:#fbfbf9;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--card:#fff;--acc:#2f5d8a}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#161615;--fg:#e9e8e3;--mut:#9b9a94;--line:#33332f;--card:#1f1f1d;--acc:#8db7e0}}}}
:root[data-theme=dark]{{--bg:#161615;--fg:#e9e8e3;--mut:#9b9a94;--line:#33332f;--card:#1f1f1d;--acc:#8db7e0}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei","Noto Sans CJK SC",sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:24px 16px 64px}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:18px;margin:32px 0 8px;color:var(--acc)}}
.sub,.note{{color:var(--mut);font-size:13px}}
.tw{{overflow-x:auto;margin:8px 0;border:1px solid var(--line);border-radius:8px;background:var(--card)}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}}
th,td{{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
th{{font-weight:600;background:color-mix(in srgb,var(--acc) 8%,transparent)}}
tr:last-child td{{border-bottom:0}}
.summary td{{white-space:normal;min-width:7em}}
h3{{font-size:15px;margin:20px 0 6px}}
.summary{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px}}
</style></head><body><main>
<h1>Strata-DS 硬件基线：{html.escape(title)}</h1>
<p class="sub">全部表格由 <code>make_report.py</code> 从本目录的 JSON 生成。原始日志在 <code>logs/</code>。</p>
{notes}
{''.join(parts)}
</main></body></html>"""
    open(os.path.join(d, "report.html"), "w").write(doc)
    print(os.path.join(d, "report.html"))


if __name__ == "__main__":
    main()
