"""The DGX Spark DeepSeek recipe (docs/DEEPSEEK_SPARK.md) on a mocked GB10. No GPU and no network."""
import contextlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup

EXL3 = 'LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw'
EXL3_REV = 'afc7aa0aab98ee735c0f54e0a122019af1ac08f1'
OFFICIAL = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'
OFFICIAL_REV = '1bae0e1088993b9b82de08f59de5be5a34e42170'
MODEL = '1017-EXL3-2.9BPW'
TAG = 'deepseek-1017-exl3-2.9bpw'
SPARK = dict(index=0, name='NVIDIA GB10', vram_gb=0.0, arch='121', driver='580.173.02', unified=True)
CARD = dict(index=0, name='RTX 4090', vram_gb=24, arch='89', driver='580.97')

O1, O2, O3 = (f'model-{i:05d}-of-00048.safetensors' for i in (1, 2, 3))
Q1, Q2, Q3 = (f'model-{i:05d}-of-00043.safetensors' for i in (1, 2, 3))
# name -> (shard, bytes): official experts and the quantized linears are large, the rest small
OFFICIAL_TENSORS = {
    'layers.0.ffn.experts.0.w1.weight': (O1, 4000), 'layers.0.attn.wq_a.weight': (O1, 2000),
    'layers.0.attn.wq_a.scale': (O1, 64), 'layers.0.attn_norm.weight': (O1, 16), 'embed.weight': (O1, 32),
    'layers.0.ffn.gate.weight': (O1, 48), 'layers.0.ffn.gate.bias': (O1, 8),
    'layers.0.attn.compressor.wkv.weight': (O1, 40), 'layers.0.attn.indexer.k_norm.weight': (O1, 24),
    'mtp.0.attn.wq_a.weight': (O2, 56), 'mtp.0.ffn.experts.0.w1.weight': (O2, 3000), 'head.weight': (O2, 900),
    'vision.blocks.0.norm1.weight': (O1, 12),     # in both repositories; the pack leaves vision out
    'layers.1.engram.embed.weight': (O3, 5000), 'layers.1.engram.embed.scale': (O3, 80),
}
EXL3_TENSORS = {
    'vision.blocks.0.norm1.weight': Q1, 'layers.0.ffn.experts.0.w1.trellis': Q2, 'layers.0.attn.wq_a.trellis': Q2,
    'layers.0.attn.wq_a.suh': Q2, 'layers.0.attn.wq_a.svh': Q2, 'layers.0.attn.wq_a.mul1': Q2,
    'layers.0.ffn.experts.0.w1.mul1': Q2, 'layers.0.attn_norm.weight': Q2,
    'embed.weight': Q2, 'layers.0.ffn.gate.weight': Q2, 'mtp.0.ffn.experts.0.w1.trellis': Q3, 'head.trellis': Q3,
}
MERGED = {
    'layers.0.attn_norm.weight': ('exl3', Q2), 'embed.weight': ('exl3', Q2),
    'layers.0.ffn.gate.weight': ('official', O1), 'layers.0.ffn.gate.bias': ('official', O1),
    'layers.0.attn.compressor.wkv.weight': ('official', O1), 'layers.0.attn.indexer.k_norm.weight': ('official', O1),
    'mtp.0.attn.wq_a.weight': ('official', O2), 'layers.1.engram.embed.weight': ('official', O3),
    'layers.1.engram.embed.scale': ('official', O3),
    'layers.0.ffn.experts.0.w1.trellis': ('exl3', Q2), 'layers.0.ffn.experts.0.w1.mul1': ('exl3', Q2),
    'layers.0.attn.wq_a.trellis': ('exl3', Q2),
    'layers.0.attn.wq_a.suh': ('exl3', Q2), 'layers.0.attn.wq_a.svh': ('exl3', Q2),
    'mtp.0.ffn.experts.0.w1.trellis': ('exl3', Q3), 'head.trellis': ('exl3', Q3),
}


def safetensors(tensors):
    """A safetensors file of {name: bytes}: U8 tensors, each filled with a byte of its own."""
    hdr, data, off = {'__metadata__': {'format': 'pt'}}, b'', 0
    for i, (name, n) in enumerate(tensors.items()):
        hdr[name] = {'dtype': 'U8', 'shape': [n], 'data_offsets': [off, off + n]}
        data += bytes([i % 251 + 1]) * n
        off += n
    h = json.dumps(hdr).encode()
    return struct.pack('<Q', len(h)) + h + data


def read_safetensors(path):
    raw = Path(path).read_bytes()
    n = struct.unpack('<Q', raw[:8])[0]
    hdr = json.loads(raw[8:8 + n])
    hdr.pop('__metadata__', None)
    return {k: raw[8 + n + v['data_offsets'][0]:8 + n + v['data_offsets'][1]] for k, v in hdr.items()}


SHARDS = {f: safetensors({k: n for k, (s, n) in OFFICIAL_TENSORS.items() if s == f}) for f in (O1, O2, O3)}


def official_url(name):
    return f'https://huggingface.co/{OFFICIAL}/resolve/{OFFICIAL_REV}/{name}'


def exl3_url(name):
    return f'https://huggingface.co/{EXL3}/resolve/{EXL3_REV}/{name}'


class Parts(unittest.TestCase):
    def test_gpus_reads_a_unified_memory_gpu(self):
        smi = '0, NVIDIA GB10, [N/A], 12.1, 580.173.02\n1, RTX 4090, 24564, 8.9, 580.97\n'
        with mock.patch.object(setup, 'out', lambda cmd, **k: smi):
            found = setup.gpus()
        self.assertEqual(found[0], SPARK)
        self.assertEqual(found[1], dict(index=1, name='RTX 4090', vram_gb=24564 / 1024, arch='89', driver='580.97'))

    def test_deepseek_accepts_gb10(self):
        with mock.patch.object(setup, 'WIN', False), mock.patch.object(sys, 'platform', 'linux'):
            self.assertIsNone(setup.deepseek_problem(SPARK))
            self.assertIn('16 GB', setup.deepseek_problem({**CARD, 'vram_gb': 12}))

    def test_merge_takes_each_tensor_from_its_repository(self):
        official = {k: s for k, (s, n) in OFFICIAL_TENSORS.items()}
        self.assertEqual(setup.spark_merge(official, EXL3_TENSORS), MERGED)

    def test_download_tensors_keeps_only_the_named_tensors(self):
        calls = []

        def http_range(url, start, end):
            calls.append((start, end))
            return SHARDS[O1][start:end + 1]
        names = ['layers.0.ffn.gate.weight', 'layers.0.attn.indexer.k_norm.weight']
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, 'http_range', http_range), \
                contextlib.redirect_stdout(io.StringIO()):
            dst = Path(d) / O1
            setup.download_tensors(official_url(O1), dst, names)
            got = read_safetensors(dst)
            self.assertEqual(set(got), set(names))
            full = read_safetensors(self._file(d, SHARDS[O1]))
            for k in names:
                self.assertEqual(got[k], full[k])
            self.assertTrue(setup.done(dst))
            n = len(calls)
            setup.download_tensors(official_url(O1), dst, names)          # finished: nothing is asked for
            self.assertEqual(len(calls), n)
            setup.download_tensors(official_url(O1), dst, names[:1])      # another list: fetched again
            self.assertEqual(set(read_safetensors(dst)), set(names[:1]))

    def _file(self, d, raw):
        p = Path(d) / 'full.safetensors'
        p.write_bytes(raw)
        return p


class Installer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.commands, self.downloads, self.starts, self.questions = [], [], [], []
        (self.root / 'ds41/data').mkdir(parents=True)
        (self.root / 'ds41/data/expert-profile.bin').write_bytes(b'mock profile')
        self.cfg_path = self.root / f'strata-{TAG}.json'
        self.src = self.root / 'data/models' / TAG
        self.pack = self.root / 'data/packs' / TAG

    def invoke(self, flags=(), cards=None, ram=121.7, free=700, family=True, mtp=True):
        cards = [SPARK] if cards is None else cards
        indexes = {official_url('model.safetensors.index.json'):
                   {'weight_map': {k: s for k, (s, n) in OFFICIAL_TENSORS.items()}},
                   exl3_url('model.safetensors.index.json'): {'weight_map': EXL3_TENSORS}}

        def download(url, dst, what=None):
            if dst.exists() and setup.done(dst):
                return
            self.downloads.append(url)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(indexes.get(url, {})))
            setup.mark(dst)

        def http_range(url, start, end):
            return SHARDS[url.rsplit('/', 1)[1]][start:end + 1]

        def run(cmd, **kwargs):
            cmd = list(map(str, cmd))
            self.commands.append(cmd)
            if '--target' in cmd:
                bdir = Path(cmd[cmd.index('--build') + 1])
                bdir.mkdir(parents=True, exist_ok=True)
                (bdir / cmd[cmd.index('--target') + 1]).write_text('mock binary')
            if any(x.endswith('/ds41/pack.py') for x in cmd):
                pack = Path(cmd[cmd.index('--out') + 1])
                pack.mkdir(parents=True, exist_ok=True)
                if '--mtp-experts' in cmd:
                    if mtp:
                        for name in ('mtp_experts.bin', 'mtp_experts.txt'):
                            (pack / name).write_text('{}')
                    return types.SimpleNamespace(returncode=0)
                for name in ('dense.bin', 'experts.bin', 'index.txt', 'experts.txt', 'engram.txt',
                             'engram_hash.txt', 'engram_tokenmap.bin', 'tokenizer.json', 'tokenizer_config.json'):
                    (pack / name).write_text('# mock pack\n' if name == 'engram.txt' else '{}')
                (pack / 'pack_info.txt').write_text(f"source {cmd[cmd.index('--src') + 1]}\nfinished 1\n")
            return types.SimpleNamespace(returncode=0)

        def start_server(cmd):
            self.starts.append(cmd)
            return 0

        def forbidden(*args, **kwargs):
            raise AssertionError('Qwen-only step called')

        argv = ['setup.py'] + (['--family', 'deepseek'] if family else []) + list(flags) + ['--yes']
        with contextlib.ExitStack() as st:
            patches = dict(ROOT=self.root, WIN=False, GPU_PICK=None, OLD_GPUS=None,
                           data_folder=lambda d: (self.root / 'data', []), load_settings=lambda: {},
                           previous_config=lambda *a: None, save_settings=lambda s: None,
                           gpus=lambda: cards, amd_gpus=lambda: [], ram_gb=lambda: ram,
                           cpu_info=lambda: ('Mock CPU', True, True), cpu_cores=lambda: (10, 10),
                           page_file_gb=lambda: None, free_gb=lambda p: (p.mkdir(parents=True, exist_ok=True) or free),
                           pip_install=lambda *a: None, get_llama_cpp=lambda: self.root / 'llama.cpp',
                           install_build_tools=lambda *a: ('/mock/cuda/bin/nvcc', None),
                           find_tool=lambda name: '/mock/bin/' + name, source_hash=lambda p: 'mock-source',
                           source_version=lambda: '0.1.39', download=download, run=run, http_range=http_range,
                           get_prebuilt=forbidden, check_shards=forbidden, refresh_draft_vocab=forbidden,
                           saved_calibration=forbidden, recommend_pool_workers=forbidden)
            for key, val in patches.items():
                st.enter_context(mock.patch.object(setup, key, val))
            st.enter_context(mock.patch.object(sys, 'platform', 'linux'))
            st.enter_context(mock.patch.object(sys, 'argv', argv))
            st.enter_context(mock.patch.object(setup.os, 'cpu_count', lambda: 20))
            st.enter_context(mock.patch.object(setup.subprocess, 'call', start_server))
            st.enter_context(mock.patch('builtins.input', lambda p: self.fail('unexpected input: ' + p)))
            st.enter_context(mock.patch.dict('os.environ', {'STRATA_EXECV': ''}))
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                try:
                    code = setup.main()
                except SystemExit as e:
                    code = e.code
        self.output = output.getvalue()
        return code

    def install(self, *flags, **kwargs):
        code = self.invoke(['--no-start', *flags], **kwargs)
        self.assertEqual(code, 0, self.output)
        return json.loads(self.cfg_path.read_text())

    def test_gb10_installs_the_spark_recipe(self):
        cfg = self.install()
        self.assertEqual(cfg['format'], 'deepseek_v41')
        self.assertEqual(cfg['model_name'], 'deepseek-v4.1-flash-1017')
        self.assertEqual(cfg['args'], ['--pack', str(self.pack), '--max-context', '262144', '--expert-profile',
                                       str(self.root / 'ds41/data/expert-profile.bin'), '--threads', '16',
                                       '--window-prompt-max', '32', '--spec', 'dspark', '--draft-always'])
        self.assertEqual(cfg['env'], {'DS41_SSD_TAIL_SKIP': '1'})
        # LibertAI: the two shards the merge uses (not the vision one); DeepSeek: the Engram shard whole, the
        # small tensors of the others by range; the indexes and the tokenizer files
        self.assertEqual(set(self.downloads), {
            exl3_url(Q2), exl3_url(Q3), official_url(O3), exl3_url('model.safetensors.index.json'),
            *(official_url(n) for n in ('model.safetensors.index.json', 'config.json', 'tokenizer.json',
                                        'tokenizer_config.json'))})
        self.assertEqual(set(read_safetensors(self.src / O1)), {k for k, v in MERGED.items() if v == ('official', O1)})
        self.assertEqual(set(read_safetensors(self.src / O2)), {'mtp.0.attn.wq_a.weight'})
        index = json.loads((self.src / 'model.safetensors.index.json').read_text())['weight_map']
        self.assertEqual(index, {k: f for k, (r, f) in MERGED.items()})
        packs = [c for c in self.commands if any(x.endswith('/ds41/pack.py') for x in c)]
        self.assertEqual([c[2:] for c in packs], [
            ['--src', str(self.src), '--out', str(self.pack), '--engram-from', str(self.root / 'ds41/data/engram')],
            ['--src', str(self.src), '--out', str(self.pack), '--mtp-experts']])
        self.assertTrue(any('-DCMAKE_CUDA_ARCHITECTURES=121' in c for c in self.commands))
        self.assertTrue(any('--target' in c and 'ds41_serve' in c for c in self.commands))
        script = self.root / f'run-{TAG}.sh'
        self.assertIn(str(self.cfg_path), script.read_text())
        self.assertEqual(setup.choices_from_config(self.cfg_path)['model'], MODEL)
        self.assertIn('DGX Spark', self.output)
        old = (len(self.commands), len(self.downloads))
        self.assertEqual(self.invoke(family=False), 0, self.output)       # the second run starts it
        self.assertEqual(old, (len(self.commands), len(self.downloads)))
        self.assertEqual(len(self.starts), 1)

    def test_an_interrupted_pack_adds_only_the_dspark_experts(self):
        self.assertNotEqual(self.invoke(['--no-start'], mtp=False), 0)
        self.commands.clear()
        self.install()
        packs = [c for c in self.commands if any(x.endswith('/ds41/pack.py') for x in c)]
        self.assertEqual(len(packs), 1)
        self.assertIn('--mtp-experts', packs[0])

    def test_disk_before_download(self):
        self.assertNotEqual(self.invoke(['--no-start'], free=600), 0)
        for text in ('625', '414.8', '210.2'):
            self.assertIn(text, self.output)
        self.assertFalse(self.downloads)

    def test_check_names_the_spark_recipe(self):
        self.assertEqual(self.invoke(['--check']), 0, self.output)
        self.assertIn('DGX Spark', self.output)
        self.assertIn('625', self.output)
        self.assertFalse(self.downloads)

    def test_a_discrete_gpu_keeps_sage_and_may_ask_for_the_spark_recipe(self):
        self.assertEqual(self.invoke(['--no-start'], cards=[CARD], ram=128, free=500), 0, self.output)
        self.assertTrue((self.root / 'strata-deepseek-sage-1.59bpw.json').is_file())
        cfg = self.install('--model', MODEL, cards=[CARD], ram=128)
        self.assertIn('tested on a DGX Spark only', self.output)
        self.assertEqual(cfg['env'], {'DS41_SSD_TAIL_SKIP': '1'})

    def test_context_choice_is_kept(self):
        cfg = self.install('--context', '65536')
        self.assertEqual(cfg['args'][cfg['args'].index('--max-context') + 1], '65536')


if __name__ == '__main__':
    unittest.main()
