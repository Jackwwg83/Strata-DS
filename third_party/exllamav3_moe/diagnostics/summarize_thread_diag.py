#!/usr/bin/env python3
"""Summarize raw diagnostic logs without treating instrumentation as an acceptance score."""
import argparse
import hashlib
import json
import math
import re
import statistics
import struct
from collections import defaultdict
from pathlib import Path


def med(values):
    return statistics.median(values) if values else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--golden", type=Path)
    a = p.parse_args()
    manifest_path = a.directory / "run-manifest.json"
    status_path = a.directory / "run-status.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    statuses = {r["stem"]: r["status"] for r in json.loads(status_path.read_text())} if status_path.exists() else {}
    threads = manifest.get("threads", [])
    repeats = manifest.get("repeats", 0)
    pool_only = manifest.get("pool_only", False)
    expected = {f"pool-t{n}" for n in threads}
    fixed_expected = set() if pool_only else {f"fixed-r{r}-t{n}" for r in range(repeats) for n in threads}
    expected |= fixed_expected
    if not pool_only:
        expected |= {f"instrumented-t{n}" for n in threads}
    complete = bool(manifest and expected) and all(statuses.get(stem) == 0 for stem in expected)
    result = {"warning": "Only uninstrumented fixed-test results are acceptance timings. Instrumented worker timings contain counters/timers. Cgroup deltas may include other processes.",
              "run_complete": complete, "expected_threads": threads,
              "command_statuses": statuses,
              "missing_or_failed_commands": {stem: statuses.get(stem, "missing") for stem in sorted(expected) if statuses.get(stem) != 0},
              "missing_run_manifest": not bool(manifest),
              "fixed_test": {}, "profiles": {}, "pool": {}, "outputs": {}}
    times = defaultdict(list)
    for path in sorted(a.directory.glob("fixed-r*-t*.log")):
        text = path.read_text()
        lines = [line for line in text.splitlines() if line.startswith("RESULT ")]
        match = re.search(r"-t(\d+)\.log$", path.name)
        n = int(match[1])
        result["fixed_test"][path.name] = {"result_lines_verbatim": lines,
            "command_status": statuses.get(path.stem, "missing"),
            "error_lines_verbatim": [line for line in text.splitlines() if "rel_l2 vs FP16 golden" in line]}
        if path.stem in fixed_expected and statuses.get(path.stem) == 0 and lines and lines[-1].startswith("RESULT pass"):
            fields = {k: float(v) for k, v in re.findall(r"(\w+)=([0-9.eE+-]+)", lines[-1])}
            times[n].append(fields)
    fixed_complete = bool(fixed_expected) and all(
        len(times[n]) == repeats for n in threads) and all(statuses.get(stem) == 0 for stem in fixed_expected)
    result["fixed_sweep_complete"] = fixed_complete
    if not pool_only and not fixed_complete:
        result["run_complete"] = False
    result["uninstrumented_repeat_medians"] = {
        n: {key: med([r[key] for r in rows]) for key in ("us_m1", "us_m8", "score_us")}
        for n, rows in times.items()} if fixed_complete else {}
    for path in sorted(a.directory.glob("instrumented-t*.jsonl")):
        try:
            records = [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")]
        except json.JSONDecodeError:
            result["profiles"][path.name] = {"error": "Incomplete JSONL; inspect exact log/status"}
            result["run_complete"] = False
            continue
        incoming = next((r["cpus"] for r in records if r["kind"] == "incoming_affinity"), [])
        counts = {m: sum(r.get("kind") == "forward" and r.get("m") == m for r in records) for m in (1, 4, 8)}
        trace_complete = counts == {1: 25, 4: 1, 8: 25}
        if not trace_complete:
            result["run_complete"] = False
        groups = defaultdict(list)
        hashes = defaultdict(set)
        for record in records:
            if record["kind"] != "forward":
                continue
            hashes[record["m"]].add(record["output_fnv1a64"])
            # Fixed test: first forward checks accuracy; next three are warmups.
            # Retain m4's one accuracy call separately; never label it a timed median.
            if record["m"] != 4 and record["sample_for_m"] < 4:
                continue
            for phase in record["phases"]:
                groups[record["m"], phase["phase"]].append((record, phase))
        profiles = {}
        for (m, phase_id), rows in groups.items():
            walls = [r[1]["wall_us"] for r in rows]
            workers = defaultdict(list)
            for _, phase in rows:
                for w in phase["workers"]:
                    workers[w["worker"]].append(w)
            summary = {"samples": len(rows), "m4_accuracy_call_only": m == 4,
                       "phase_wall_us_median": med(walls),
                       "start_spread_us_median": med([max(w["start_us"] for w in ph["workers"])-min(w["start_us"] for w in ph["workers"]) for _, ph in rows]),
                       "finish_spread_us_median": med([max(w["end_us"] for w in ph["workers"])-min(w["end_us"] for w in ph["workers"]) for _, ph in rows]),
                       "last_finish_to_return_us_median": med([ph["wall_us"]-max(w["end_us"] for w in ph["workers"]) for _, ph in rows]),
                       "workers": {}}
            for worker, samples in workers.items():
                summary["workers"][worker] = {
                    "cpu_start_observed": sorted({w["cpu_start"] for w in samples}),
                    "cpu_end_observed": sorted({w["cpu_end"] for w in samples}),
                    "core_types": sorted({w["core_type"] for w in samples}),
                    "pin_targets": sorted({w["pin_target"] for w in samples}),
                    "pin_errno": sorted({w["pin_errno"] for w in samples}),
                    "outside_incoming_affinity": any(w["cpu_start"] not in incoming or w["cpu_end"] not in incoming for w in samples),
                    "cpu_us_median": med([w["cpu_us"] for w in samples]),
                    "wall_us_median": med([w["end_us"]-w["start_us"] for w in samples]),
                    "start_us_median": med([w["start_us"] for w in samples]),
                    "end_us_median": med([w["end_us"] for w in samples]),
                    "phase_completion_wait_us_median": med([
                        ph["wall_us"]-next(w["end_us"] for w in ph["workers"] if w["worker"] == worker)
                        for _, ph in rows]),
                    "bands_median": med([w["bands"] for w in samples]),
                    "row_bands_median": med([w["row_bands"] for w in samples]),
                    "weight_bytes_median": med([w["weight_bytes"] for w in samples]),
                    "cpu_us_per_band_median": med([w["cpu_us"]/w["bands"] for w in samples if w["bands"]]),
                }
            profiles[f"m{m}-phase{phase_id}"] = summary
        result["profiles"][path.name] = {"incoming_affinity": incoming, "groups": profiles,
            "expected_forward_counts_present": trace_complete, "forward_counts": counts,
            "command_status": statuses.get(path.stem, "missing"),
            "repeated_forward_output_hashes_by_m": {m: sorted(v) for m, v in hashes.items()}}
    for path in sorted(a.directory.glob("pool-t*.jsonl")):
        try:
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")]
        except json.JSONDecodeError:
            result["pool"][path.name] = {"error": "Incomplete JSONL; inspect exact log/status"}
            result["run_complete"] = False
            continue
        samples = [r["us_per_five"] for r in rows if r["kind"] == "noop_five_dispatch_burst"]
        if len(samples) != 11:
            result["run_complete"] = False
        result["pool"][path.name] = {"median_burst_average_us_per_five": med(samples),
                                   "command_status": statuses.get(path.stem, "missing"),
                                   "workers": [r for r in rows if r["kind"] == "worker"]}
    by_m = defaultdict(list)
    for path in sorted(a.directory.glob("instrumented-t*-m*.f32")):
        raw = path.read_bytes()
        m = int(re.search(r"-m(\d+)\.f32$", path.name)[1])
        item = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        values = [v[0] for v in struct.iter_unpack("=f", raw)]
        item["finite"] = all(map(math.isfinite, values))
        if not item["finite"] or len(raw) != m*5120*4:
            result["run_complete"] = False
        if a.golden:
            gold = (a.golden / f"out_{m}.bin").read_bytes()
            if len(gold) != len(raw):
                raise SystemExit(f"Golden size mismatch for {path}")
            wants = [v[0] for v in struct.iter_unpack("=f", gold)]
            item["relative_l2_full_precision"] = math.sqrt(sum((v-w)**2 for v, w in zip(values, wants))/max(sum(w*w for w in wants), 1e-300))
        result["outputs"][path.name] = item
        by_m[m].append(item["sha256"])
    equality = {}
    for m in (() if pool_only else (1, 4, 8)):
        required = [a.directory / f"instrumented-t{n}-m{m}.f32" for n in threads]
        available = len(threads) >= 2 and all(path.exists() for path in required) and all(
            statuses.get(f"instrumented-t{n}") == 0 for n in threads)
        equality[m] = len({hashlib.sha256(path.read_bytes()).hexdigest() for path in required}) == 1 if available else None
        if not available:
            result["run_complete"] = False
    result["instrumented_output_bitwise_equal_across_threads"] = equality
    required_logs = {stem + (".jsonl" if stem.startswith(("pool-", "instrumented-")) else ".log") for stem in expected}
    result["missing_expected_logs"] = sorted(name for name in required_logs if not (a.directory / name).exists())
    if result["missing_expected_logs"]:
        result["run_complete"] = False
    dest = a.directory / "summary.json"
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("run_complete", "fixed_sweep_complete", "uninstrumented_repeat_medians", "instrumented_output_bitwise_equal_across_threads")}, indent=2))
    print(dest)
    if not result["run_complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
