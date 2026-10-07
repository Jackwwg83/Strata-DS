"""Pack a synthetic checkpoint with pinned Engram files. No torch or downloads."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
ENGRAM = ROOT / 'ds41/data/engram'


class EngramCopy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.src, self.out = self.root / 'src', self.root / 'pack'
        self.src.mkdir()
        tensors = {'embed.weight': ('BF16', [2], b'\x00\x01\x02\x03')}
        for w in ('w1', 'w3', 'w2'):
            for p in ('trellis', 'suh', 'svh', 'mul1'):
                tensors[f'layers.0.ffn.experts.0.{w}.{p}'] = ('I16', [16], bytes(range(32)))
        hdr, blob = {}, b''
        for name, (dt, shape, data) in tensors.items():
            hdr[name] = dict(dtype=dt, shape=shape, data_offsets=[len(blob), len(blob) + len(data)])
            blob += data
        raw = json.dumps(hdr).encode()
        shard = 'model-00001-of-00001.safetensors'
        (self.src / shard).write_bytes(struct.pack('<Q', len(raw)) + raw + blob)
        (self.src / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': dict.fromkeys(tensors, shard)}))
        for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
            (self.src / name).write_text('{"engram_layer_ids": [1, 14]}')
        self.copy = self.root / 'engram'
        shutil.copytree(ENGRAM, self.copy)

    def run_pack(self, *extra):
        # macOS has pread but no preadv. The production installer runs on Linux.
        runner = '''import builtins, os, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'transformers', 'sympy'):
        raise AssertionError('forbidden import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
if not hasattr(os, 'preadv'):
    def preadv(fd, buffers, offset):
        data = os.pread(fd, len(buffers[0]), offset)
        buffers[0][:len(data)] = data
        return len(data)
    os.preadv = preadv
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
        return subprocess.run([sys.executable, '-c', runner, str(ROOT / 'tools/ds41/pack.py'),
                               '--src', str(self.src), '--out', str(self.out), '--layers', '1', '--experts', '1',
                               '--engram-from', str(self.copy), *extra], capture_output=True, text=True)

    def test_cli_copies_verified_files_without_heavy_imports(self):
        r = self.run_pack()
        self.assertEqual(r.returncode, 0, r.stderr)
        for name in ('engram_hash.txt', 'engram_tokenmap.bin'):
            self.assertEqual((self.out / name).read_bytes(), (ENGRAM / name).read_bytes())
        self.assertIn('finished 1', (self.out / 'pack_info.txt').read_text())
        self.assertEqual((self.out / 'dense.bin').read_bytes(), b'\x00\x01\x02\x03')
        self.assertEqual((self.out / 'experts.bin').stat().st_size, 4096)

    def test_corrupt_or_missing_copy_never_finishes(self):
        for name in ('engram_hash.txt', 'engram_tokenmap.bin'):
            with self.subTest(name=name):
                original = (self.copy / name).read_bytes()
                (self.copy / name).write_bytes(b'bad')
                r = self.run_pack()
                self.assertNotEqual(r.returncode, 0)
                self.assertIn('SHA-256', r.stderr)
                self.assertFalse((self.out / 'pack_info.txt').exists())
                (self.copy / name).unlink()
                self.assertNotEqual(self.run_pack().returncode, 0)
                (self.copy / name).write_bytes(original)

    def test_bf16_option_is_refused_before_torch_import(self):
        r = self.run_pack('--wo-a-bf16')
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('--wo-a-bf16', r.stderr)
        self.assertNotIn('forbidden import', r.stderr)


if __name__ == '__main__':
    unittest.main()
