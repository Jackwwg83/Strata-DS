"""ds41/bench/scripts/prompt_window_ab.py - how ds41_serve should read the new part of a prompt that continues the
session: verify windows of 4 tokens, or one batched prefill pass. Prints the prompt time of each path per length.

    python ds41/bench/scripts/prompt_window_ab.py --exe build/ds41_serve --pack PACK --expert-profile F

Each case: another conversation (the session starts over), a 300-token base prompt with 8 output tokens, then the
base plus its output plus N new tokens (the measured request: it reuses the session and reads N + 1 tokens).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "ds41"))
from test_serve_protocol import Engine, prompt  # noqa: E402

LENGTHS = [8, 24, 64, 128, 256, 512]


def run(cmd: list[str]) -> dict[int, float]:
    e = Engine(cmd)
    out = {}
    for n in LENGTHS:
        e.request(prompt(30, 1000 + n), 1)                 # another conversation: start over
        base = prompt(300, n)
        _, gen, _, _ = e.request(base, 8)
        _, _, done, err = e.request(base + gen + prompt(n + 1, 2000 + n)[1:], 1)
        assert done is not None, err
        out[n] = float(done[3])                            # prompt ms
        print(f"  new {n}: reused {done[8]}, read {done[14]}, prompt {done[3]} ms", flush=True)
    e.send("QUIT")
    e.p.wait(timeout=120)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--expert-profile", default="")
    ap.add_argument("--threads", default="8")
    a = ap.parse_args()
    cmd = [a.exe, "--serve", "--pack", a.pack, "--max-context", "16384", "--threads", a.threads]
    if a.expert_profile:
        cmd += ["--expert-profile", a.expert_profile]
    print("windows (--window-prompt-max 100000)", flush=True)
    win = run(cmd + ["--window-prompt-max", "100000"])
    print("one prefill pass (--window-prompt-max 0)", flush=True)
    pas = run(cmd + ["--window-prompt-max", "0"])
    print("new tokens | windows ms | pass ms")
    for n in LENGTHS:
        print(f"{n:10d} | {win[n]:10.0f} | {pas[n]:7.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
