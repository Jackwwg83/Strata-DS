#!/usr/bin/env python3
"""Test-only line engine. Emit committed HF golden ids. Do not load model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--serve', action='store_true')
    ap.add_argument('--pack', required=True)
    ap.add_argument('--max-context', type=int, required=True)
    ap.add_argument('--expert-profile')
    ap.add_argument('--ram-budget-gib', type=float)
    ap.add_argument('--threads', type=int)
    args = ap.parse_args()
    if not args.serve:
        ap.error('ds41_serve runs only with --serve (the server always passes it)')
    golden = json.loads(Path(__file__).with_name('deepseek_golden.json').read_text(encoding='utf-8'))
    ids = next(c['ids'] for c in golden['cases'] if c['name'] == 'completion-True')
    print('INFO engine=fake-ds41 batch_slots=0', flush=True)
    print(f'READY {args.max_context} stop', flush=True)
    print('fake-ds41: ready; deterministic golden output; no model weights', file=sys.stderr, flush=True)
    for line in sys.stdin:
        command = line.split()[0]
        if command == 'QUIT':
            break
        if command == 'STOP':
            continue
        if command != 'GEN':
            print(f'ERR unsupported {command}', flush=True)
            continue
        fields = line.split()
        maximum = int(fields[1])
        prompt = fields[-1].split(',')
        out = ids[:maximum]
        print(f'RESUME 0', flush=True)
        print(f'PP {len(prompt)} {len(prompt)}', flush=True)
        for token in out:
            print(f'T {token}', flush=True)
        print(f'DONE {len(out)} {len(prompt)} 1.0 1.0 stop', flush=True)
        print(json.dumps({'engine': 'fake-ds41', 'event': 'generation', 'prompt_tokens': len(prompt),
                          'prompt_sha256': hashlib.sha256(fields[-1].encode()).hexdigest(),
                          'output_tokens': len(out), 'eos': bool(out and out[-1] == golden['eos_id'])}),
              file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
