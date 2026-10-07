"""Compile production host control paths with CUDA fault injection. No GPU is used.

Run with --sanitize=thread on Linux to check host synchronization.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CASES = "valid early_slot component_overflow dense_overflow slot_overflow shape_overflow negative_index missing_index zero_prime bad_index large_index short_array layer_mismatch path_spaces engram_retry vram_cleanup stream_cleanup vram_race host_race thread_cleanup".split()


def writer_test(root, baseline_ref=None):
    # Load the production writer without the unrelated tensor libraries.
    code = (subprocess.check_output(["git", "show", f"{baseline_ref}:tools/ds41/pack.py"], cwd=ROOT, text=True)
            if baseline_ref else (ROOT / "tools/ds41/pack.py").read_text())
    source = ast.parse(code)
    fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "write_engram")
    ns = dict(os=os, json=json, struct=struct)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "pack.py", "exec"), ns)
    for suffix in (" with spaces", "\nnewline", "\rcarriage"):
        src = root / ("source" + suffix)
        src.mkdir()
        out = root / "out"
        out.mkdir(exist_ok=True)
        (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layers.1.engram.embed.weight": "table.safetensors"}}))
        hdr = json.dumps({"layers.1.engram.embed.weight": {"shape": [100, 256], "data_offsets": [0, 25600]},
                          "layers.1.engram.embed.scale": {"data_offsets": [25600, 26400]}}).encode()
        (src / "table.safetensors").write_bytes(struct.pack("<Q", len(hdr)) + hdr)
        try:
            ns["write_engram"](str(src), str(out))
        except (ValueError, SystemExit):
            if suffix == " with spaces":
                raise AssertionError("writer rejected spaces")
        else:
            if suffix != " with spaces":
                raise AssertionError("writer accepted a newline in the path")
            assert str(src / "table.safetensors") in (out / "engram.txt").read_text()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", help="Compile production files from this git revision")
    parser.add_argument("--sanitize", choices=["address", "thread", "undefined"])
    parser.add_argument("--cases", nargs="*", default=CASES)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="ds41-revfix-") as temp:
        work = Path(temp)
        compat = work / "compat.h"
        compat.write_text('''#ifdef __APPLE__
#include <sys/mman.h>
#define O_DIRECT 0
#define POSIX_FADV_DONTNEED 0
inline int posix_fadvise(int, long long, long long, int) { return 0; }
#define mincore(a,b,c) mincore(a,b,reinterpret_cast<char*>(c))
#endif
''')
        cmd = [os.environ.get("CXX", "c++"), "-std=c++17", "-O1", "-g", "-pthread", "-include", str(compat),
               "-Isrc/ds41/tests/revfix_stubs", "-Iinclude", "-Ithird_party/exllamav3_moe"]
        if args.sanitize:
            cmd += ["-fsanitize=" + args.sanitize, "-fno-omit-frame-pointer"]
        cmd += ["-x", "c++", "src/ds41/tests/revfix_test.cpp", "src/ds41/pack.cpp", "src/ds41/vram_experts.cu",
                "src/ds41/host_experts.cpp", "src/ds41/expert_stream.cpp", "src/ds41/lookahead.cpp",
                "src/ds41/engram_rows.cpp", "src/platform/direct_file.cpp", "src/ds41/tests/revfix_stubs/allocation_failure.cpp", "-o", str(work / "test")]
        if sys.platform != "darwin":
            cmd += ["-ldl"]
        if args.baseline_ref:
            old = work / "baseline"
            paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", args.baseline_ref,
                                             "include/strata/ds41"], cwd=ROOT, text=True).splitlines()
            paths += [p for p in cmd if p.startswith("src/") and "/tests/" not in p]
            for path in paths:
                dest = old / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(subprocess.check_output(["git", "show", f"{args.baseline_ref}:{path}"], cwd=ROOT))
            cmd = [str(old / p) if p in paths else p for p in cmd]
            cmd.insert(cmd.index("-Iinclude"), "-I" + str(old / "include"))
            print("BASELINE", args.baseline_ref, flush=True)
        print("BUILD", shlex.join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        print("MODE: CUDA fault injection; " + ("O_DIRECT emulated with buffered pread on macOS" if sys.platform == "darwin" else "native O_DIRECT"), flush=True)
        failures = 0
        for case in args.cases:
            result = subprocess.run([str(work / "test"), case, str(work / case)], capture_output=True, text=True, timeout=30)
            print(result.stdout + result.stderr, end="", flush=True)
            if result.returncode:
                print(f"EXIT {case}: {result.returncode}", flush=True)
                failures += 1
        try:
            writer_test(work, args.baseline_ref)
            print("PASS writer_paths")
        except AssertionError as e:
            print("FAIL writer_paths:", e)
            failures += 1
        print(f"RESULT {len(args.cases) + 1 - failures} passed; {failures} failed", flush=True)
        return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
