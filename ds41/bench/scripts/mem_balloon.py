"""Hold memory so a box with more RAM behaves like a smaller PC (page cache included: the container's memory limit
counts the file cache too). Usage: python mem_balloon.py --keep-gib 119.9 & ... kill it after the benchmark.
Reads the cgroup limit (v2 memory.max or v1 memory.limit_in_bytes) and touches (limit - keep) GiB of anonymous memory.
"""
import argparse
import time

import numpy as np


def limit_bytes():
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            if v.isdigit():
                return int(v)
        except OSError:
            pass
    raise SystemExit("no cgroup memory limit found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-gib", type=float, required=True)
    a = ap.parse_args()
    hold = limit_bytes() - int(a.keep_gib * 2**30)
    if hold <= 0:
        raise SystemExit("the limit is already below --keep-gib")
    blocks = []
    step = 1 << 30
    for _ in range(hold // step):
        blocks.append(np.ones(step // 8))   # writes every page: resident
    print(f"holding {hold / 2**30:.1f} GiB; the container keeps {a.keep_gib} GiB", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
