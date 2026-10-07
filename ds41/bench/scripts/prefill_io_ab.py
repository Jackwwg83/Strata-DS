"""ds41/bench/scripts/prefill_io_ab.py - the prompt reading of ds41_serve with the expert stream through the file
cache or with O_DIRECT (DS41_UNBUFFERED), on a box whose file cache cannot keep the pack beside the RAM tier.

    python ds41/bench/scripts/prefill_io_ab.py --exe build/ds41_serve --pack PACK --expert-profile F

Per mode: a new engine, then three different 2000-token prompts (the session starts over each time), the prompt ms of
each from the DONE line.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "ds41"))
from test_serve_protocol import Engine, prompt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--expert-profile", required=True)
    ap.add_argument("--threads", default="16")
    a = ap.parse_args()
    cmd = [a.exe, "--serve", "--pack", a.pack, "--max-context", "16384", "--threads", a.threads,
           "--expert-profile", a.expert_profile]
    for mode in ("0", "1"):
        os.environ["DS41_UNBUFFERED"] = mode
        t0 = time.time()
        e = Engine(cmd)
        ready = time.time() - t0
        ms = []
        for seed in (1, 2, 3):
            _, _, done, err = e.request(prompt(2000, 300 + seed), 1)
            assert done is not None, err
            ms.append(float(done[3]))
        e.send("QUIT")
        e.p.wait(timeout=300)
        print(f"DS41_UNBUFFERED={mode}: READY after {ready:.0f} s; 2000-token prompts "
              + ", ".join(f"{m / 1000:.1f} s" for m in ms), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
