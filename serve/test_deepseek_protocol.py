"""Exercise the real HTTP server and line protocol with a deterministic fake engine."""
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from serve.test_deepseek import ROOT, TOKENIZER_DIR, TOOL


@unittest.skipUnless(TOKENIZER_DIR, 'set DS41_TOKENIZER_DIR to run the DeepSeek protocol tests')
class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='ds41-server-')
        cls.directory = Path(cls.temp.name)
        cls.config = cls.directory / 'config.json'
        cls.log = cls.directory / 'server.log'
        cls.engine_log = cls.directory / 'engine.log'
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            cls.port = sock.getsockname()[1]
        cfg = {'format': 'deepseek_v41', 'tokenizer': str(Path(TOKENIZER_DIR).resolve()),
               'exe': str(ROOT / 'serve/fake_ds41_engine.py'), 'cwd': str(ROOT),
               'args': ['--pack', str(cls.directory), '--max-context', 8192,
                        '--expert-profile', 'unused', '--ram-budget-gib', 1, '--threads', 1],
               'parallel': 4, 'effort_position': 'end', 'expert_profile_save': 'unused',
               'vram_elastic': True, 'log': str(cls.engine_log)}
        cls.config.write_text(json.dumps(cfg))
        cls.output = cls.log.open('w')
        cls.proc = subprocess.Popen([sys.executable, 'serve/server.py', '--engine', 'strata',
            '--config', str(cls.config), '--port', str(cls.port)], cwd=ROOT, stdout=cls.output,
            stderr=subprocess.STDOUT, env={**os.environ, 'DS41_TOKENIZER_DIR': cfg['tokenizer']})
        for _ in range(200):
            if 'ready: http://' in cls.log.read_text():
                return
            if cls.proc.poll() is not None:
                break
            time.sleep(.05)
        cls.proc.terminate()
        cls.proc.wait(timeout=10)
        cls.output.close()
        raise AssertionError(cls.log.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(timeout=10)
        cls.output.close()
        cls.temp.cleanup()

    def post(self, path, body):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        try:
            conn.request('POST', path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
            response = conn.getresponse()
            raw = response.read().decode()
            self.assertEqual(response.status, 200, raw)
            return raw
        finally:
            conn.close()

    def test_openai_stream_and_nonstream(self):
        request = {'messages': [{'role': 'user', 'content': 'Find 台北.'}], 'max_tokens': 512,
                   'tools': [{'type': 'function', 'function': TOOL}]}
        raw = self.post('/v1/chat/completions', request)
        msg = json.loads(raw)['choices'][0]['message']
        self.assertEqual(msg['reasoning_content'], 'Need a lookup.\n')
        self.assertEqual(msg['content'], 'Checking 台北.\n')
        self.assertEqual(msg['tool_calls'][0]['function']['name'], 'lookup')
        self.assertEqual(json.loads(msg['tool_calls'][0]['function']['arguments'])['count'], 2)
        raw = self.post('/v1/chat/completions', {**request, 'stream': True})
        chunks = [json.loads(line[6:]) for line in raw.splitlines()
                  if line.startswith('data: ') and line != 'data: [DONE]']
        deltas = [c['choices'][0]['delta'] for c in chunks if c.get('choices')]
        self.assertEqual(''.join(d.get('reasoning_content', '') for d in deltas), msg['reasoning_content'])
        self.assertEqual(''.join(d.get('content', '') for d in deltas), msg['content'])
        calls = [c for d in deltas for c in d.get('tool_calls', [])]
        self.assertEqual(calls[0]['function']['name'], 'lookup')
        self.assertEqual(json.loads(''.join(c['function'].get('arguments', '') for c in calls)),
                         json.loads(msg['tool_calls'][0]['function']['arguments']))
        self.assertIn('data: [DONE]', raw)
        self.assertEqual(chunks[-2]['choices'][0]['finish_reason'] if not chunks[-1].get('choices') else
                         chunks[-1]['choices'][0]['finish_reason'], 'tool_calls')

    def test_responses(self):
        request = {'input': 'Find 台北.', 'max_output_tokens': 512,
                   'tools': [{'type': 'function', **TOOL}]}
        for stream in (False, True):
            raw = self.post('/v1/responses', {**request, 'stream': stream})
            if stream:
                events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: ')]
                body = events[-1]['response']
            else:
                body = json.loads(raw)
            reasoning, message, call = body['output']
            self.assertEqual(reasoning['content'][0]['text'], 'Need a lookup.\n')
            self.assertEqual(message['content'][0]['text'], 'Checking 台北.\n')
            self.assertEqual(call['name'], 'lookup')
            self.assertEqual(json.loads(call['arguments'])['count'], 2)

    def test_anthropic(self):
        request = {'messages': [{'role': 'user', 'content': 'Find 台北.'}], 'max_tokens': 512,
                   'tools': [{'name': TOOL['name'], 'input_schema': TOOL['parameters']}]}
        body = json.loads(self.post('/v1/messages', request))
        thinking, text, call = body['content']
        self.assertEqual(thinking['thinking'], 'Need a lookup.\n')
        self.assertEqual(text['text'], 'Checking 台北.\n')
        self.assertEqual((call['name'], call['input']['count']), ('lookup', 2))
        raw = self.post('/v1/messages', {**request, 'stream': True})
        events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: ')]
        deltas = [e['delta'] for e in events if e['type'] == 'content_block_delta']
        self.assertEqual(''.join(d.get('thinking', '') for d in deltas), thinking['thinking'])
        self.assertEqual(''.join(d.get('text', '') for d in deltas), text['text'])
        self.assertEqual(json.loads(''.join(d.get('partial_json', '') for d in deltas)), call['input'])


class PipeProtocolTests(ProtocolTests):
    """Use in-memory HTTP streams when testing without a TCP listener. The engine is a real child process."""

    @classmethod
    def setUpClass(cls):
        from serve.deepseek import DeepSeekTemplate, DeepSeekTokenizer
        from serve.server import DeepSeekEngine, Service
        cls.temp = tempfile.TemporaryDirectory(prefix='ds41-pipe-')
        cls.directory = Path(cls.temp.name)
        cls.engine_log = cls.directory / 'engine.log'
        cls.engine = DeepSeekEngine(str(ROOT / 'serve/fake_ds41_engine.py'),
            ['--pack', str(cls.directory), '--max-context', '8192'], log=str(cls.engine_log))
        cls.svc = Service(cls.engine, DeepSeekTokenizer(TOKENIZER_DIR), DeepSeekTemplate(),
                          format='deepseek_v41', model_name='deepseek-v4.1-flash')
        cls.raw_responses = []

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()
        cls.temp.cleanup()

    def post(self, path, body):
        import io
        from types import SimpleNamespace
        from unittest.mock import patch
        from serve.server import make_handler
        data = json.dumps(body).encode()
        request = (f'POST {path} HTTP/1.0\r\nHost: localhost\r\nContent-Type: application/json\r\n'
                   f'Content-Length: {len(data)}\r\n\r\n').encode() + data
        incoming, outgoing = io.BytesIO(request), io.BytesIO()

        class Connection:
            def makefile(self, mode, *args):
                return incoming

            def sendall(self, data):
                outgoing.write(data)

        handler = make_handler(self.svc)
        with patch.object(handler, '_watch_client'):
            handler(Connection(), ('127.0.0.1', 10000), SimpleNamespace())
        raw = outgoing.getvalue().decode()
        self.raw_responses.append(raw)
        self.assertIn(' 200 ', raw.splitlines()[0], raw)
        return raw.split('\r\n\r\n', 1)[1]


@unittest.skipUnless(TOKENIZER_DIR, 'set DS41_TOKENIZER_DIR')
class StartupTests(unittest.TestCase):
    def test_main_selects_deepseek_without_qwen_flags(self):
        import threading
        from unittest.mock import Mock, patch
        from serve import server
        from serve.deepseek import DeepSeekTokenizer, DeepSeekTemplate
        with tempfile.TemporaryDirectory() as directory:
            cfg = {'format': 'deepseek_v41', 'tokenizer': str(Path(TOKENIZER_DIR).resolve()),
                   'exe': str(ROOT / 'serve/fake_ds41_engine.py'),
                   'args': ['--pack', directory, '--max-context', 8192, '--expert-profile', 'unused',
                            '--ram-budget-gib', 1, '--threads', 1],
                   'gpu': [0, 1], 'layer_split': 'bad', 'parallel': 4,
                   'effort_position': 'end', 'expert_profile_save': 'unused', 'vram_elastic': True}
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(cfg))
            real_sleep = time.sleep

            def stop_main(seconds):
                if threading.current_thread() is threading.main_thread():
                    raise KeyboardInterrupt
                real_sleep(seconds)

            with patch.object(sys, 'argv', ['server.py', '--engine', 'strata', '--config', str(path)]), \
                    patch.object(server, 'Server'), patch.object(server, 'serve', return_value=Mock()) as serve, \
                    patch.object(server.time, 'sleep', side_effect=stop_main):
                self.assertEqual(server.main(), 0)
            svc = serve.call_args.args[0]
            self.assertIsInstance(svc.engine, server.DeepSeekEngine)
            self.assertIsInstance(svc.tok, DeepSeekTokenizer)
            self.assertIsInstance(svc.template, DeepSeekTemplate)
            self.assertEqual(svc.model, 'deepseek-v4.1-flash')
            self.assertFalse(svc.effort_end)
            self.assertEqual(svc.engine.batch, 0)
            self.assertEqual(svc.engine.spawn[1], [str(x) for x in cfg['args']])
