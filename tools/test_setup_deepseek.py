"""DeepSeek installer traffic on a mocked Linux PC. No GPU and no network."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup

REPO = 'vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw'
REV = 'eca94a388a70841858feed8f057a9862e897aba4'
MODEL = 'SAGE-1.59BPW'
FILES = ['config.json', 'generation_config.json', 'model.safetensors.index.json', 'tokenizer.json',
         'tokenizer_config.json', 'tokenizer.model'] + [f'model-{i:05d}-of-00017.safetensors' for i in range(1, 18)]
CARD = dict(index=0, name='RTX 4090', vram_gb=24, arch='89', driver='580.97')


class Installer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.commands, self.downloads, self.starts, self.api, self.questions = [], [], [], [], []
        (self.root / 'ds41/data').mkdir(parents=True)
        (self.root / 'ds41/data/expert-profile.bin').write_bytes(b'mock profile')
        self.cfg_path = self.root / 'strata-deepseek-sage-1.59bpw.json'

    def invoke(self, flags=(), system='linux', cards=None, ram=128, free=600, answers=None, family=True, fingerprint="mock-source", fail_build=False, previous=None):
        cards = [CARD] if cards is None else cards
        def download(url, dst, what=None):
            if dst.exists() and setup.done(dst):
                return
            self.downloads.append(url)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text('{}')
            setup.mark(dst)
        def run(cmd, **kwargs):
            cmd = list(map(str, cmd)); self.commands.append(cmd)
            if '--target' in cmd:
                if fail_build:
                    raise SystemExit(1)
                target = cmd[cmd.index('--target') + 1]
                bdir = Path(cmd[cmd.index('--build') + 1]); bdir.mkdir(parents=True, exist_ok=True)
                (bdir / target).write_text('mock binary')
            if any(x.endswith('/ds41/pack.py') for x in cmd):
                pack = Path(cmd[cmd.index('--out') + 1]); pack.mkdir(parents=True, exist_ok=True)
                src = cmd[cmd.index('--src') + 1]
                for name in ('dense.bin', 'experts.bin', 'index.txt', 'experts.txt', 'engram.txt',
                             'engram_hash.txt', 'engram_tokenmap.bin', 'tokenizer.json', 'tokenizer_config.json'):
                    (pack / name).write_text('# mock pack\n' if name == 'engram.txt' else '{}')
                (pack / 'pack_info.txt').write_text(f'source {src}\nfinished 1\n')
            return types.SimpleNamespace(returncode=0)
        def urlopen(req, **kwargs):
            self.api.append(req.full_url if hasattr(req, 'full_url') else req)
            siblings = FILES + ['README.md', 'exllamav3/config.json', 'other.safetensors']
            return io.BytesIO(json.dumps({'siblings': [{'rfilename': n} for n in siblings]}).encode())
        def user_input(prompt):
            self.questions.append(prompt)
            if answers is None:
                raise AssertionError('unexpected input: ' + prompt)
            return next((v for k, v in answers.items() if k in prompt), '')
        def start_server(cmd):
            self.starts.append(cmd)
            path = Path(cmd[cmd.index('--config') + 1])
            cfg = json.loads(path.read_text())
            event = dict(event='mock_server_start', mock=True, config=str(path),
                         model_name=cfg['model_name'], format=cfg['format'], args=cfg['args'])
            with Path(cfg['log']).open('a') as log:
                log.write(json.dumps(event) + '\n')
            return 0
        def forbidden(*args, **kwargs):
            raise AssertionError('Qwen-only step called')
        argv = ['setup.py'] + (['--family', 'deepseek'] if family else []) + list(flags)
        if answers is None:
            argv.append('--yes')
        with contextlib.ExitStack() as st:
            patches = dict(ROOT=self.root, WIN=system=='win32', GPU_PICK=None, OLD_GPUS=None,
                           data_folder=lambda d: (self.root / 'data', []), load_settings=lambda: {},
                           previous_config=lambda *a: previous, save_settings=lambda s: None,
                           gpus=lambda: cards, amd_gpus=lambda: [], ram_gb=lambda: ram,
                           cpu_info=lambda: ('Mock CPU', True, True), cpu_cores=lambda: (8, 16),
                           page_file_gb=lambda: None, free_gb=lambda p: (p.mkdir(parents=True, exist_ok=True) or free),
                           pip_install=lambda *a: None, get_llama_cpp=lambda: self.root / 'llama.cpp',
                           install_build_tools=lambda *a: ('/mock/cuda/bin/nvcc', None),
                           find_tool=lambda name: '/mock/bin/' + name, source_hash=lambda p: fingerprint,
                           source_version=lambda: '0.1.39', download=download, run=run,
                           get_prebuilt=forbidden, check_shards=forbidden, refresh_draft_vocab=forbidden,
                           saved_calibration=forbidden, recommend_pool_workers=forbidden)
            for key, val in patches.items():
                st.enter_context(mock.patch.object(setup, key, val))
            st.enter_context(mock.patch.object(sys, 'platform', system))
            st.enter_context(mock.patch.object(sys, 'argv', argv))
            st.enter_context(mock.patch.object(setup.urllib.request, 'urlopen', urlopen))
            st.enter_context(mock.patch.object(setup.subprocess, 'call', start_server))
            st.enter_context(mock.patch('builtins.input', user_input))
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

    def test_config_build_pack_download_script_and_second_start(self):
        cfg = self.install('--resident-budget-gib', '96', '--port', '8091', '--no-browser')
        self.assertEqual(cfg['format'], 'deepseek_v41')
        self.assertEqual(cfg['model_name'], 'deepseek-v4.1-flash')
        self.assertEqual(Path(cfg['exe']).name, 'ds41_serve')
        pack = self.root / 'data/packs/deepseek-sage-1.59bpw'
        expected = ['--pack', str(pack), '--max-context', '32768', '--expert-profile',
                    str(self.root / 'ds41/data/expert-profile.bin'), '--threads', '8', '--ram-budget-gib', '96']
        self.assertEqual(cfg['args'], expected)
        self.assertEqual(cfg['tokenizer'], str(pack))
        for key in ('cwd', 'log', 'port', 'lib_dirs'):
            self.assertIn(key, cfg)
        self.assertEqual(set(self.downloads), {f'https://huggingface.co/{REPO}/resolve/{REV}/{f}' for f in FILES})
        self.assertEqual(self.api, [f'https://huggingface.co/api/models/{REPO}/revision/{REV}'])
        self.assertTrue(any('--target' in c and 'ds41_serve' in c for c in self.commands))
        command = next(c for c in self.commands if any(x.endswith('/ds41/pack.py') for x in c))
        self.assertEqual(command[2:], ['--src', str(self.root / 'data/models/deepseek-sage-1.59bpw'),
                                      '--out', str(pack), '--engram-from', str(self.root / 'ds41/data/engram')])
        script = self.root / 'run-deepseek-sage-1.59bpw.sh'
        self.assertIn(str(self.cfg_path), script.read_text())
        self.assertIn('8091', script.read_text()); self.assertNotIn('--open', script.read_text())
        self.assertTrue(script.stat().st_mode & 0o111)
        ch = setup.choices_from_config(self.cfg_path)
        self.assertEqual((ch['family'], ch['model']), ('deepseek', MODEL))
        self.assertIn('--threads 8', setup.settings_summary(cfg))
        old = (len(self.commands), len(self.downloads))
        self.assertEqual(self.invoke(family=False), 0, self.output)
        self.assertEqual(old, (len(self.commands), len(self.downloads)))
        self.assertEqual(len(self.starts), 1)
        self.assertIn(str(self.cfg_path), self.starts[0])
        self.assertEqual(json.loads(self.cfg_path.read_text())['args'], expected)

    def test_platform_and_gpu_refusals(self):
        for system, cards, flags in [('win32', [CARD], []), ('darwin', [CARD], []),
                                      ('linux', [], ['--backend', 'hip']),
                                      ('linux', [{**CARD, 'vram_gb': 12}], []),
                                      ('linux', [{**CARD, 'arch': '80'}], [])]:
            with self.subTest(system=system, cards=cards):
                self.assertNotEqual(self.invoke(flags, system=system, cards=cards), 0)
                self.assertIn('DeepSeek', self.output)
                self.assertFalse(self.downloads)

    def test_low_ram_requires_deliberate_choice(self):
        self.assertNotEqual(self.invoke(['--no-start'], ram=64, answers={}), 0)
        self.assertIn('not tested', self.output)
        self.install(ram=64)
        self.assertIn('may be slow', self.output)
        self.assertIn('128 GB', self.output)

    def test_disk_before_download(self):
        self.assertNotEqual(self.invoke(['--no-start'], free=461), 0)
        self.assertIn('462', self.output); self.assertIn('341.8', self.output); self.assertIn('120', self.output)
        self.assertFalse(self.downloads); self.assertFalse(self.commands)

    def test_check_does_not_install(self):
        self.assertEqual(self.invoke(['--check'], ram=64), 0, self.output)
        self.assertIn('not tested', self.output); self.assertIn('462', self.output)
        self.assertFalse(self.commands); self.assertFalse(self.downloads); self.assertFalse(self.api)

    def test_interactive_family_is_available_and_not_default(self):
        self.install(family=False, answers={'Which model?': str(list(setup.FAMILIES).index('deepseek') + 1)})
        self.assertEqual(list(setup.FAMILIES)[0], 'qwen')
        self.assertIn('DeepSeek', self.output)
        self.assertFalse(any('KV cache?' in q or 'images?' in q or 'speed projection?' in q for q in self.questions))

    def test_context_limits_and_unsupported_options(self):
        for flags in (['--context', '262145'], ['--rope-scaling', 'yarn'], ['--gpus', '0,1'],
                      ['--model', 'Q2_0'], ['--calibrate'], ['--backend', 'sycl']):
            with self.subTest(flags=flags):
                self.assertNotEqual(self.invoke(['--no-start', *flags]), 0, self.output)
                self.assertFalse(self.downloads)
        cfg = self.install('--context', '262144', '--vision', 'yes', '--kv', 'q4_0',
                           '--low-ram', 'on', '--experimental-speed-projection', 'on')
        self.assertEqual(set(x for x in cfg['args'] if x.startswith('--')),
                         {'--pack', '--max-context', '--expert-profile', '--threads'})
        self.assertIn('262144', cfg['args'])

    def test_update_keeps_deepseek_config(self):
        cfg = self.install()
        self.assertEqual(self.invoke(['--update'], family=False), 0, self.output)
        self.assertEqual(json.loads(self.cfg_path.read_text()), cfg)
        self.assertFalse(self.starts)

    def test_repeat_setup_reuses_pack_and_files(self):
        self.install()
        before = len(self.commands)
        self.install('--context', '65536')
        self.assertEqual(len(self.commands), before)

    def test_host_needs_key_before_download(self):
        self.assertNotEqual(self.invoke(['--no-start', '--host', '0.0.0.0']), 0)
        self.assertFalse(self.downloads)
        cfg = self.install('--host', '0.0.0.0', '--api-key', 'mock-test-key')
        self.assertEqual(cfg['api_key'], 'mock-test-key')

    def test_nominal_16gb_card_is_allowed_with_warning(self):
        self.install(cards=[{**CARD, 'vram_gb': 15.99}])
        self.assertIn('under 24 GB', self.output)

    def test_nominal_24gb_card_is_not_warned(self):
        # an RTX 4090 reports 23.99 GB (24564 MiB): the measured card must not get the "not tested" warning
        self.install(cards=[{**CARD, 'vram_gb': 23.99}])
        self.assertNotIn('under 24 GB', self.output)

    def test_update_refreshes_library_paths(self):
        self.install()
        stamp = self.root / 'engine/DS41_BUILD.json'
        meta = json.loads(stamp.read_text()); meta['lib_dirs'] = ['/mock/new-cuda/lib64']
        stamp.write_text(json.dumps(meta))
        self.assertEqual(self.invoke(['--update'], family=False), 0, self.output)
        self.assertEqual(json.loads(self.cfg_path.read_text())['lib_dirs'], ['/mock/new-cuda/lib64'])

    def test_start_refuses_missing_engram_source(self):
        cfg = self.install()
        (Path(cfg['tokenizer']) / 'engram.txt').write_text('1 7 16 100 200 /missing/model-00016-of-00017.safetensors\n')
        self.assertNotEqual(self.invoke(family=False), 0)
        self.assertIn('Engram', self.output)
        self.assertFalse(self.starts)

    def test_thread_count_uses_physical_cores_and_caps_at_16(self):
        with mock.patch.object(setup, 'cpu_cores', lambda: (24, 0)):
            self.assertEqual(setup.deepseek_threads(), 16)
        with mock.patch.object(setup, 'cpu_cores', lambda: (6, 8)):
            self.assertEqual(setup.deepseek_threads(), 6)

    def test_failed_update_keeps_installed_engine_for_start(self):
        cfg = self.install()
        self.assertEqual(self.invoke(family=False, fingerprint='changed', fail_build=True), 0, self.output)
        self.assertIn('starting the installed', self.output)
        self.assertEqual(len(self.starts), 1)
        self.assertEqual(json.loads(self.cfg_path.read_text())['exe'], cfg['exe'])

    def test_resume_counts_existing_source_bytes(self):
        src = self.root / 'data/models/deepseek-sage-1.59bpw'
        src.mkdir(parents=True)
        with (src / 'model-00001-of-00017.safetensors.part').open('wb') as f:
            f.truncate(100_000_000_000)
        self.install(free=362)

    def test_build_tracks_cmake_and_cpu_expert_sources(self):
        with mock.patch.object(setup, 'ROOT', self.root):
            before = setup.source_hash(setup.DEEPSEEK_SOURCES)
            for name in ('cmake/ds41_engine.cmake', 'third_party/exllamav3_moe/moe_mul1.cpp'):
                path = self.root / name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('changed source')
                after = setup.source_hash(setup.DEEPSEEK_SOURCES)
                self.assertNotEqual(before, after)
                before = after

    def test_failed_update_reports_failure_without_losing_config(self):
        cfg = self.install()
        self.assertNotEqual(self.invoke(['--update'], family=False, fingerprint='changed', fail_build=True), 0)
        self.assertEqual(json.loads(self.cfg_path.read_text()), cfg)
        self.assertNotIn('Strata is updated', self.output)

    def test_changed_gpu_cannot_fall_back_to_incompatible_binary(self):
        self.install()
        self.assertNotEqual(self.invoke(family=False, cards=[{**CARD, 'arch': '120'}], fail_build=True), 0)
        self.assertFalse(self.starts)

    def test_cuda_choice_survives_update(self):
        cfg = self.install('--cuda', '12')
        self.assertEqual(cfg['cuda'], 12)
        stamp = self.root / 'engine/DS41_BUILD.json'
        self.assertEqual(json.loads(stamp.read_text())['toolkit'], 12)
        self.assertEqual(self.invoke(['--update'], family=False, fingerprint='changed'), 0, self.output)
        self.assertEqual(json.loads(stamp.read_text())['toolkit'], 12)

    def test_mock_start_consumes_the_written_config_and_records_an_event(self):
        self.assertEqual(self.invoke(['--no-browser']), 0, self.output)
        cfg = json.loads(self.cfg_path.read_text())
        event = json.loads(Path(cfg['log']).read_text().splitlines()[-1])
        self.assertEqual(event['event'], 'mock_server_start')
        self.assertEqual(event['format'], 'deepseek_v41')
        self.assertEqual(event['args'], cfg['args'])
        self.assertTrue(event['mock'])

    def test_new_checkout_reuses_previous_source_and_pack(self):
        cfg = self.install('--resident-budget-gib', '96')
        old = self.cfg_path
        self.root = self.root / 'new-checkout'
        self.root.mkdir()
        self.cfg_path = self.root / old.name
        before = len(self.downloads)
        self.assertEqual(self.invoke(family=False, previous=old), 0, self.output)
        new = json.loads(self.cfg_path.read_text())
        self.assertEqual(new['tokenizer'], cfg['tokenizer'])
        self.assertEqual(new['args'][-2:], ['--ram-budget-gib', '96'])
        self.assertEqual(len(self.downloads), before)

    def test_pin_never_falls_back_to_main(self):
        url = f'https://huggingface.co/{REPO}/resolve/{REV}/config.json'
        self.assertEqual(setup.hf_unpinned(url), url)


def evidence(folder):
    """Keep a mocked installation and its observed calls for review."""
    case = Installer()
    case.setUp()
    try:
        case.root = Path(folder).resolve()
        case.root.mkdir(parents=True, exist_ok=False)
        (case.root / 'ds41/data').mkdir(parents=True)
        (case.root / 'ds41/data/expert-profile.bin').write_bytes(b'mock profile')
        case.cfg_path = case.root / 'strata-deepseek-sage-1.59bpw.json'
        flags = ['--resident-budget-gib', '96', '--context', '32768', '--port', '8091', '--no-browser']
        first = case.invoke(flags, ram=119.9)
        output = case.output
        if first != 0:
            raise AssertionError(output)
        before = (len(case.commands), len(case.downloads))
        second = case.invoke(family=False)
        if second != 0 or before != (len(case.commands), len(case.downloads)):
            raise AssertionError(case.output)
        (case.root / 'setup.stdout.log').write_text(output + '\nSECOND RUN\n' + case.output)
        data = dict(mock=True, command=['setup.py', '--family', 'deepseek', *flags, '--yes'],
                    first_exit=first, second_exit=second, commands=case.commands, downloads=case.downloads,
                    starts=case.starts, config=json.loads(case.cfg_path.read_text()),
                    run_script=(case.root / 'run-deepseek-sage-1.59bpw.sh').read_text())
        (case.root / 'evidence.json').write_text(json.dumps(data, indent=2) + '\n')
        print(f'Mocked installer evidence: {case.root}')
        print(f'First run: {first}; second run: {second}; server boundary calls: {len(case.starts)}')
    finally:
        case.doCleanups()


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--evidence':
        evidence(sys.argv[2])
    else:
        unittest.main()
