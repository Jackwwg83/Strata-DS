#!/usr/bin/env python3
"""Build and run an unchanged K11 test plus a separate timing-instrumented copy.

Linux only. All generated sources, objects, binaries and reports live in --out.
Default sweep is 8/12/16/24/32 requested workers. No thread-count cap is applied.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import re
import shlex
import shutil
import subprocess
import sys
import time

from make_thread_diag import generate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VENDOR = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cgroup_directory():
    """Resolve unified-cgroup membership against its actual mount, including namespaces."""
    try:
        member = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                      if line.startswith("0::"))
        def unescape(value):
            return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            left, right = line.split(" - ", 1)
            if right.split()[0] != "cgroup2":
                continue
            fields = left.split()
            root, mount = unescape(fields[3]), Path(unescape(fields[4]))
            if member == root:
                relative = ""
            elif member.startswith(root.rstrip("/") + "/"):
                relative = member[len(root):].lstrip("/")
            elif member == "/":
                # In a cgroup namespace, '/' names the mounted namespace root.
                relative = ""
            else:
                continue
            directory = mount / relative
            if (directory / "cpu.stat").exists():
                return directory, mount
        raise ValueError("No readable cgroup-v2 CPU controller matching current membership")
    except (OSError, ValueError, StopIteration) as e:
        return None, str(e)


def cgroup_stats():
    directory, detail = cgroup_directory()
    if directory is None:
        return {"unavailable": detail}
    try:
        result = {k: int(v) for k, v in (line.split() for line in (directory / "cpu.stat").read_text().splitlines())}
        result["resolved_directory"] = str(directory)
        result["visible_ancestor_stats"] = {}
        while directory != detail and directory.parent != directory:
            directory = directory.parent
            try:
                result["visible_ancestor_stats"][str(directory)] = {
                    k: int(v) for k, v in (line.split() for line in (directory / "cpu.stat").read_text().splitlines())}
            except (OSError, ValueError) as e:
                result["visible_ancestor_stats"][str(directory)] = {"unavailable": str(e)}
        return result
    except (OSError, ValueError) as e:
        return {"unavailable": str(e), "resolved_directory": str(directory)}


def run(command, log, env=None, timeout=None):
    print(shlex.join(map(str, command)), flush=True)
    before = cgroup_stats()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    rc = 0
    launch_error = None
    with log.open("w") as f:
        try:
            p = subprocess.run(list(map(str, command)), stdout=f, stderr=subprocess.STDOUT,
                               env=env, timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            f.write("\nK11_DIAG_RUNNER timeout\n")
            rc = 124
        except OSError as e:
            launch_error = str(e)
            f.write("\nK11_DIAG_RUNNER launch failed: " + launch_error + "\n")
            rc = 127
    after = cgroup_stats()
    usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    telemetry = {"status": rc, "wall_seconds": time.monotonic()-start,
                 "launch_error": launch_error,
                 "command": list(map(str, command)),
                 "effective_test_env": {k: v for k, v in (env or os.environ).items() if k.startswith(("K11_", "EXL3_MOE_"))},
                 "child_user_seconds": usage_end.ru_utime-usage.ru_utime,
                 "child_system_seconds": usage_end.ru_stime-usage.ru_stime,
                 "cgroup_cpu_stat_before": before, "cgroup_cpu_stat_after": after,
                 "cgroup_cpu_stat_delta": {k: v-before[k] for k, v in after.items()
                    if isinstance(v, int) and isinstance(before.get(k), int)},
                 "visible_ancestor_cpu_stat_deltas": {path: {k: v-before.get("visible_ancestor_stats", {}).get(path, {}).get(k, v)
                     for k, v in values.items() if isinstance(v, int)}
                     for path, values in after.get("visible_ancestor_stats", {}).items()}}
    # Cgroup counters can include unrelated sibling processes; they are evidence,
    # not exclusive attribution of throttling to the tested kernel.
    log.with_suffix(log.suffix + ".telemetry.json").write_text(json.dumps(telemetry, indent=2) + "\n")
    return rc


def build(a):
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"Use a new/empty build directory; refusing stale artifacts: {a.out}")
    a.out.mkdir(parents=True, exist_ok=True)
    receipt = a.out / "build-status.json"
    receipt.write_text(json.dumps({"complete": False}) + "\n")
    root = a.source_root or ROOT
    vendor = root / "third_party/exllamav3_moe"
    source = vendor / "moe_mul1.cpp"
    fixed = root / "src/ds41/tests/k11_cpu_moe_test.cpp"
    generated = a.out / "moe_mul1_diag.cpp"
    generate(source, generated)
    raw = generated.read_text()
    pool = raw[raw.index("typedef void (*PoolFn)"):raw.index("//   Pool self-test hook")]
    includes = raw[:raw.index("// CPU MoE expert GEMM")]
    pool_source = a.out / "pool_probe.cpp"
    pool_source.write_text(includes + '#include "thread_diag.h"\ninline void cpu_pause(){__builtin_ia32_pause();}\n'
                           + pool + (HERE / "pool_probe_main.inc").read_text()
                           + '\n#endif\n')
    flags = [a.cxx, "-O3", "-std=c++17", "-pthread", "-I", vendor, "-I", HERE, "-I", root / "include"]
    commands = [
        (flags + [pool_source, "-o", a.out / "pool_probe"], "build-pool.log"),
        (flags + ["-c", source, "-o", a.out / "vendor.o"], "build-vendor.log"),
        (flags + ["-c", generated, "-o", a.out / "vendor_diag.o"], "build-diag.log"),
    ]
    if not a.pool_only:
        if not a.cuda_include or not a.cudart:
            raise SystemExit("For the fixed test, provide --cuda-include DIR (repeatable) and --cudart FILE; or --pool-only")
        cuda = [item for p in a.cuda_include for item in ("-I", p)]
        commands += [(flags + cuda + ["-c", root / "src/ds41/pack.cpp", "-o", a.out / "pack.o"], "build-pack.log")]
        for tag, obj in (("fixed", "vendor.o"), ("instrumented", "vendor_diag.o")):
            commands += [(flags + [fixed, a.out / "pack.o", a.out / obj, a.cudart,
                                  "-Wl,-rpath," + str(a.cudart.parent), "-o", a.out / tag], "build-" + tag + ".log")]
    else:
        commands = commands[:1]
    manifest = {"source_commit": subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip(),
                "source_sha256": sha(source), "fixed_test_sha256": sha(fixed),
                "header_sha256": sha(vendor / "moe_mul1.h"),
                "input_sha256": {str(p.relative_to(root)): sha(p) for p in
                    sorted([p for p in vendor.iterdir() if p.suffix in (".h", ".cpp")]
                        + [root / "src/ds41/pack.cpp", root / "include/strata/ds41/pack.hpp",
                           root / "include/strata/ds41/config.hpp", fixed])},
                "diagnostics_sha256": {p.name: sha(p) for p in HERE.iterdir() if p.is_file()},
                "commands": [[str(x) for x in c] for c, _ in commands]}
    (a.out / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for command, log in commands:
        rc = run(command, a.out / log)
        if rc:
            raise SystemExit(f"Build failed ({rc}): {a.out / log}")
    receipt.write_text(json.dumps({"complete": True, "pool_only": a.pool_only}) + "\n")


def environment(env=None):
    result = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "platform": platform.platform(), "python": sys.version,
              "incoming_affinity": sorted(os.sched_getaffinity(0)),
              "effective_test_env": {k: v for k, v in (env or os.environ).items() if k.startswith(("K11_", "EXL3_MOE_"))}}
    for path in ("/proc/cpuinfo", "/proc/self/status", "/proc/self/cgroup"):
        try:
            result[path] = Path(path).read_text()
        except OSError as e:
            result[path] = str(e)
    directory, mount = cgroup_directory()
    result["cgroup_cpu_stat"] = cgroup_stats()
    result["cgroup_cpu_limits"] = {}
    if directory is not None:
        while True:
            values = {}
            for field in ("cpu.max", "cpuset.cpus.effective"):
                try:
                    values[field] = (directory / field).read_text().strip()
                except OSError as e:
                    values[field] = str(e)
            result["cgroup_cpu_limits"][str(directory)] = values
            if directory == mount or directory.parent == directory:
                break
            directory = directory.parent
    topology = {}
    for cpu in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
        fields = {}
        for field in ("topology/core_id", "topology/physical_package_id", "topology/thread_siblings_list", "topology/core_type", "cpu_capacity", "cpufreq/cpuinfo_max_freq"):
            try:
                fields[field] = (cpu / field).read_text().strip()
            except OSError:
                pass
        topology[cpu.name] = fields
    result["topology"] = topology
    return result


def sweep(a):
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"Use a new/empty sweep directory; refusing stale results: {a.out}")
    receipt_path = a.build / "build-status.json"
    if not receipt_path.exists():
        raise SystemExit(f"Missing completed-build receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    if not receipt.get("complete") or (receipt.get("pool_only") and not a.pool_only):
        raise SystemExit("Build incomplete or pool-only; cannot run this sweep")
    a.out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["EXL3_MOE_CPU_PIN"] = "1" if a.pin == "on" else "0"
    if a.isa:
        env["EXL3_MOE_CPU_MAX_ISA"] = a.isa
    (a.out / "environment.json").write_text(json.dumps(environment(env), indent=2) + "\n")
    manifest = {"threads": a.threads, "repeats": a.repeats, "pool_only": a.pool_only,
                "pin": a.pin, "isa": env.get("EXL3_MOE_CPU_MAX_ISA", "auto"),
                "build_manifest": json.loads((a.build / "build-manifest.json").read_text())}
    (a.out / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    statuses = []
    def record_status(stem, rc):
        statuses.append({"stem": stem, "status": rc})
        (a.out / "run-status.json").write_text(json.dumps(statuses, indent=2) + "\n")
    # Fresh process at each worker count: no leftover helper threads from a larger pool.
    # Uninstrumented timing first; never use the instrumented RESULT as a score.
    if not a.pool_only:
        for repeat in range(a.repeats):
            for n in a.threads[::1 if repeat % 2 == 0 else -1]:
                env["K11_THREADS"] = str(n)
                stem = f"fixed-r{repeat}-t{n}"
                rc = run([a.build / "fixed"], a.out / (stem + ".log"), env, a.timeout)
                record_status(stem, rc)
        for n in a.threads:
            env["K11_THREADS"] = str(n)
            stem = f"instrumented-t{n}"
            env["K11_DIAG_JSONL"] = str(a.out / (stem + ".jsonl"))
            env["K11_DIAG_DUMP_PREFIX"] = str(a.out / stem)
            rc = run([a.build / "instrumented"], a.out / (stem + ".log"), env, a.timeout)
            record_status(stem, rc)
    for n in a.threads:
        stem = f"pool-t{n}"
        rc = run([a.build / "pool_probe", n, a.pool_iterations], a.out / (stem + ".jsonl"), env, a.timeout)
        record_status(stem, rc)
    (a.out / "run-status.json").write_text(json.dumps(statuses, indent=2) + "\n")
    print("Reports:", a.out, flush=True)
    if any(s["status"] for s in statuses):
        raise SystemExit("Some commands failed; see run-status.json and the exact unfiltered logs")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--source-root", type=Path, help="Build another checkout with this same diagnostic harness")
    b.add_argument("--cxx", default="g++")
    b.add_argument("--cuda-include", type=Path, action="append")
    b.add_argument("--cudart", type=Path)
    b.add_argument("--pool-only", action="store_true")
    s = sub.add_parser("sweep")
    s.add_argument("--build", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--threads", type=int, nargs="+", default=[8, 12, 16, 24, 32])
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--pin", choices=["on", "off"], default="on")
    s.add_argument("--isa", choices=["scalar", "avx2", "avxvnni", "bw", "vnni", "vbmi"])
    s.add_argument("--pool-only", action="store_true")
    s.add_argument("--pool-iterations", type=int, default=20)
    s.add_argument("--timeout", type=int, default=900)
    a = p.parse_args()
    for name in ("out", "build", "cudart", "source_root"):
        if getattr(a, name, None):
            setattr(a, name, getattr(a, name).resolve())
    if a.mode == "build":
        build(a)
    else:
        if len(set(a.threads)) != len(a.threads) or any(n < 1 or n > 512 for n in a.threads) or a.repeats < 1:
            p.error("diagnostic thread counts must be 1..512")
        sweep(a)


if __name__ == "__main__":
    main()
