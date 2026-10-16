#!/usr/bin/env python3
"""SSD read bandwidth as the engine reads experts: O_DIRECT, random 4 KiB-aligned offsets in experts.bin, each read
EXPERT bytes in 2 MiB parts, T threads, for SECONDS. Prints one JSON line (GB/s, reads/s).
Usage: ssdread.py FILE THREADS [SECONDS] [EXPERT_BYTES]"""
import json, mmap, os, random, sys, threading, time

path, threads = sys.argv[1], int(sys.argv[2])
seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 15
expert = int(sys.argv[4]) if len(sys.argv) > 4 else 7 << 20
PART = 2 << 20
size = os.path.getsize(path)
done_bytes, done_reads, stop = [0] * threads, [0] * threads, time.time() + seconds


def worker(i):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    buf = mmap.mmap(-1, PART)                 # page-aligned, as O_DIRECT needs
    rng = random.Random(i)
    while time.time() < stop:
        at = rng.randrange(0, size - expert) // 4096 * 4096
        for off in range(0, expert, PART):
            done_bytes[i] += os.preadv(fd, [memoryview(buf)[:min(PART, expert - off)]], at + off)
        done_reads[i] += 1
    os.close(fd)


t0 = time.time()
ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
for t in ts: t.start()
for t in ts: t.join()
dt = time.time() - t0
print(json.dumps({"threads": threads, "expert_bytes": expert, "GBps": round(sum(done_bytes) / dt / 1e9, 2),
                  "experts_per_s": round(sum(done_reads) / dt, 1), "ms_per_expert_at_this_depth":
                      round(1000 * dt * threads / max(1, sum(done_reads)), 2)}))
