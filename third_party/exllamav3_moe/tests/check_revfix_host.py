"""Run the actual registry and staging source without the x86 compute pool.

Use --native on Linux to compile the complete production translation unit.
"""
from pathlib import Path
import argparse
import os
import subprocess
import tempfile

VENDOR = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--native", action="store_true")
p.add_argument("--sanitize", choices=["address,undefined", "undefined"])
p.add_argument("--source-ref", help="Read production source from this git revision.")
a = p.parse_args()
if a.native and a.source_ref:
    p.error("Use a separate checkout for a native baseline run.")
source = (subprocess.check_output(["git", "show", f"{a.source_ref}:third_party/exllamav3_moe/moe_mul1.cpp"], text=True)
          if a.source_ref else (VENDOR / "moe_mul1.cpp").read_text())
with tempfile.TemporaryDirectory(prefix="revfix-cpu-") as name:
    tmp = Path(name)
    if a.native:
        unit = VENDOR / "moe_mul1.cpp"
    else:
        begin = source.index("static MoeCpuMatrix make_matrix_raw")
        end = source.index("// Prime from the handoff worker", begin)
        raw = source[begin:end]
        begin = source.index("struct StageCtx")
        end = source.index("bool exl3_moe_cpu_has_avx2", begin)
        stage = source[begin:end]
        rate = source[source.index("constexpr int rate_k2"):source.index("constexpr int tile_words32")]
        unit = tmp / "layer.cpp"
        unit.write_text('''#include "moe_mul1.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
std::vector<MoeCpuLayer*> g_layers;
std::mutex g_layers_mutex;
''' + raw + "\nnamespace {\n" + rate + stage)
    cmd = [os.environ.get("CXX", "clang++"), "-std=c++17", "-O1", "-g", "-pthread",
           "-I", str(VENDOR), str(unit), str(VENDOR / "tests/revfix_test.cpp"), "-o", str(tmp / "test")]
    if a.sanitize:
        cmd += ["-fsanitize=" + a.sanitize, "-fno-omit-frame-pointer"]
    subprocess.run(cmd, check=True)
    print(f"COMPILED {'native CPU layer' if a.native else 'extracted registry and staging'}", flush=True)
    failed = 0
    for case in [["stage"], ["shape"], ["leak"]] + [["alloc", str(i)] for i in range(12)]:
        r = subprocess.run([str(tmp / "test"), *case])
        print(f"EXIT {' '.join(case)}: {r.returncode}", flush=True)
        failed += r.returncode != 0
    raise SystemExit(bool(failed))
