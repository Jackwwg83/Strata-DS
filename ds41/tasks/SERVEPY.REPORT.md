# DeepSeek V4.1 Python server report

## Result and limits

The Python implementation is complete. The full acceptance is not complete.
16 focused tests pass. All three API handlers return reasoning, text, and structured tool calls through a real fake-engine child process. Both streaming and non-streaming paths pass.

I cannot run the requested live traffic because this sandbox denies socket binding with `PermissionError: [Errno 1] Operation not permitted`. Approval is unavailable. A direct `serve/server.py` launch fails during its port check. No HTTP service remains running.

The real engine is not available on this machine. There is no GPU. No result below measures model quality, GPU inference, memory use, or real-engine throughput.

Git also cannot create the worktree index lock. No commit was created. `SERVEPY.COMMITS.md` contains the required fallback plan.

## Component status

| Component | Code | Deployment path | Observed data | Classification |
| --- | --- | --- | --- | --- |
| Tokenizer | Implemented | `serve/deepseek.py` | 18 HF golden strings match; whole and incremental UTF-8 round trips pass | Real local tokenizer files; synthetic corpus |
| Prompt renderer | Implemented | `DeepSeekTemplate` imports `ds41/proto/ref/encoding.py` | Reference comparisons pass for chat, tools, results, efforts, and later system messages | Reference test data |
| Output parser | Implemented | `DeepSeekOutputParser` | Character and token split tests pass; API tool arguments parse as JSON | Synthetic completions |
| Config and engine bridge | Implemented | `serve/server.py --engine strata --config CONFIG` | `main()` selects the adapter and starts the fake child with the expected args | TCP listener mocked in startup unit test |
| OpenAI / Responses / Anthropic | Implemented | Existing API handlers | Reasoning, text, and tool calls pass in both modes | In-memory HTTP streams; real child pipes; fake model output |
| Live listener | Config prepared | `/tmp/ds41-servepy-evidence/config.json`, port 18095 | Direct startup failed before listening | Blocked, not deployed |
| Real engine / GPU | External to this task | Intended `engine/ds41_serve` | None | Unavailable; not inferred |
| Commits | Plan prepared | `ds41/tasks/SERVEPY.COMMITS.md` | Git rejected index-lock creation | Blocked |

## Files

- `serve/deepseek.py`: Loads HF-format byte BPE without HF at runtime. Reuses Strata's BPE merge algorithm. Applies the ordered regex splits from the file. Matches added tokens before BPE. Returns raw token bytes to the existing incremental detokenizer. Loads EOS from `tokenizer_config.json`. Adapts normalized messages to the official encoder. Parses DSML incrementally.
- `serve/deepseek_golden.json`: 22.6 KB of text and HF ids. Contains English, Chinese, Japanese, code, digits, whitespace, CRLF, tabs, emoji, combining marks, controls, special tokens, DSML, four rendered prompts, and two completions. Source: `vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw` at `eca94a388a70841858feed8f057a9862e897aba4`.
- `serve/frontend.py`: Adds an optional `deepseek=False` argument to both normalizers. DeepSeek preserves later system messages and tool ids. It rejects images before text conversion can discard them. The default Qwen path is unchanged.
- `serve/responses.py`: Adds the same optional switch to preserve later system messages. Keeps the existing tool-result ordering.
- `serve/server.py`: Connects the format switch, tokenizer, template, parser, stop id, wrap-up, default model name, and normalizers. Launches the dedicated binary directly. Skips unsupported engine options. Rejects VRAM requests without writing a command. Makes the existing mock engine use the tokenizer's EOS when available.
- `serve/test_deepseek.py`: Golden, reference-renderer, normalization, parser-boundary, config, image, and budget tests. Real-tokenizer tests skip unless `DS41_TOKENIZER_DIR` is set.
- `serve/fake_ds41_engine.py`: Executable test fixture. Reads the line protocol. Emits real DeepSeek ids from committed HF golden data. Logs a generation event. Rejects unsupported commands and CLI flags. Loads no weights.
- `serve/test_deepseek_protocol.py`: Tests all three APIs and both response modes. Includes an actual subprocess/TCP test for a host that permits sockets, a child-process test with in-memory HTTP streams, and a `main()` configuration test with a mocked listener.
- `ds41/tasks/SERVEPY.PROGRESS.md`: Final checklist and reasons for blocked items.
- `ds41/tasks/SERVEPY.COMMITS.md`: Three commit units. Each has at most five files and the required trailer.
- `ds41/tasks/SERVEPY.REPORT.md`: This report and captured evidence.

No existing test assertion changed. The official encoder, engine, `setup.py`, and `docs/` are unchanged. Local tokenizer files and the virtual environment are not part of the changes.

## Design choices

`"format": "deepseek_v41"` is the switch. An absent key retains Qwen. The service holds the selected parser and constants. Request normalization also receives the switch because Qwen rewrites later system messages before rendering.

The renderer imports the reference from the checkout. The server already adds the repository root to `sys.path`. A copy would create a second source to maintain. Tools become `{"type": "function", "function": ...}` entries on the first system message. If needed, the adapter inserts that message. Assistant tool calls, tool results, and `reasoning_content` retain the reference input shape. The reference performs its own tool-result merge and reasoning-history policy.

DeepSeek skips `_late_system_to_user`, literal-think marking, the Qwen effort turn, and `effort_position`. The reference owns the prompt markers.

The tokenizer preserves the file's regex strings exactly. The file represents no normalization as an empty `Sequence`. The adapter accepts that and `null`. Added tokens match as whole strings before pre-tokenization. Their bytes use literal UTF-8. Ordinary tokens use the GPT-2 byte mapping. The observed EOS id is **1**, read from the supplied files rather than hard-coded. No runtime import of HF `tokenizers` exists.

The parser emits text as soon as it is safe. It announces each tool name when the invoke header arrives. It emits each argument after its closing parameter tag, then emits the final tool call. `string="true"` stays a string, even for `123`. `string="false"` uses JSON. It preserves whitespace and holds partial protocol tags. EOS ends the message. A truncated tool call stays incomplete. Argument streaming is per parameter, not per character; one large parameter stays buffered until its closing tag.

The budget wrap-up is exactly `</think>`. It closes reasoning without introducing Qwen text or newline trimming. The resume test checks the second prompt and final EOS.

### Effort map

| Server normalized value | Reference value | Numeric effort |
| --- | --- | --- |
| absent | `None` | 75, the reference default |
| low | 50 | 50 |
| medium | 75 | 75 |
| high / xhigh / max / maximum | 100 | 100 |
| none / off / minimal / disabled / false | `thinking_mode="chat"` | No effort prefix |

The existing normalizers map high/max aliases to `xhigh`. Anthropic budgets retain the existing thresholds: under 2048 is low; under 8192 is medium; otherwise xhigh. Thinking on uses `thinking_mode="thinking"`.

### Engine option audit

| Option or path | DeepSeek behavior |
| --- | --- |
| `--pack`, `--max-context` | Passed unchanged as strings; existing path/context reads support them |
| `--expert-profile`, `--ram-budget-gib`, `--threads` | Passed unchanged as strings |
| `--serve` | Not added; `ds41_serve` is the dedicated protocol binary |
| `--native` | Rejected when explicitly supplied |
| `parallel`, `--batch`, `--slots` | No batch flag added; explicit unsupported flags rejected |
| GPU layer split options | No layer-split flags added or validated for DeepSeek |
| `effort_position`, `--tail-role-token` | No tail-role flag added; explicit flag rejected |
| `expert_profile_save` | No save flag or learned-profile replacement |
| VRAM elasticity flags / command | No flags added; API rejects resizing before writing `VRAM` |
| Images / `GENI` | Request rejected with a DeepSeek-specific error |

The DeepSeek CLI allowlist contains the six options named in the task. New engine options need an explicit update. Batch-only Qwen EOS constants are unreachable on the supported DeepSeek path.

## Tests and commands

Working directory: `/Users/jackwu/Projects/Strata-DS-servepy`.

The initial worktree was clean on `feature/ds41-serve-py`.

### Test-first observations

- Tokenizer test failed because `serve.deepseek` did not exist. After implementation, the empty normalizer sequence exposed a validation error. The fix made the golden test pass.
- Renderer tests failed on the missing adapter and normalizer keyword. They passed after the adapter and normalization changes.
- Parser tests failed on the missing parser. All boundary tests passed after implementation.
- Config tests failed on Qwen layer-split handling and the missing service format argument. They passed after integration.
- The protocol test first failed at socket binding. The pipe variant then failed on the fake engine's malformed `DONE` field. Fixing the fixture made all API checks pass.
- The length-limit test exposed lost trailing newlines. The budget test exposed a Qwen EOS in the mock engine. The image test exposed image content discarded by text conversion. Each test passed after its fix.

### Focused checks with the real tokenizer

```sh
DS41_TOKENIZER_DIR=.ds41-tokenizer .venv-ds41/bin/python -m unittest serve.test_deepseek serve.test_deepseek_protocol.PipeProtocolTests serve.test_deepseek_protocol.StartupTests -v
```

```text
Ran 16 tests in 1.165s
OK
```

16 passed. The startup unit test replaces `Server` and `serve`; its printed ready line does not establish a real listener.

### Skip behavior without local tokenizer files

```sh
.venv-ds41/bin/python -m unittest serve.test_deepseek serve.test_deepseek_protocol -v
```

```text
Ran 19 tests in 0.070s
OK (skipped=11)
```

8 passed, 11 skipped.

### Full serve suite

```sh
DS41_TOKENIZER_DIR=.ds41-tokenizer .venv-ds41/bin/python -m unittest discover -s serve -p 'test_*.py'
```

```text
Ran 232 tests in 17.642s
FAILED (errors=120, skipped=9)
```

113 individual tests passed. There were zero assertion failures. All 120 errors ended in `PermissionError: [Errno 1] Operation not permitted`. Ten errors were class-setup failures, so the error count is not a count of individual tests run. The new TCP protocol class accounts for one of these errors.

Existing tests alone ran with 97 passes, 119 permission errors, zero assertion failures, and 9 skips. Nine errors were class-setup failures. This is not a claim that every existing test passed.

### Unchanged baseline comparison

```sh
mkdir -p /tmp/ds41-qwen-baseline
git archive HEAD serve tools | tar -x -C /tmp/ds41-qwen-baseline
cd /tmp/ds41-qwen-baseline
/Users/jackwu/Projects/Strata-DS-servepy/.venv-ds41/bin/python -m unittest discover -s serve -p 'test_*.py'
```

```text
Ran 216 tests in 16.760s
FAILED (errors=119, skipped=9)
```

The baseline also has 97 passes and zero assertion failures. All 119 errors are permission errors. The existing error identities match the changed checkout exactly. Existing socket-dependent assertions still need a permitted host.

`git diff --check` passed. The reference encoder diff is empty.

### Golden data generation

HF `tokenizers` was used only to create the JSON ids. The committed strings include reference-rendered prompts. To regenerate ids for those same strings:

```sh
.venv-ds41/bin/python - <<'PY'
import json
from pathlib import Path
from tokenizers import Tokenizer
p = Path('serve/deepseek_golden.json')
golden = json.loads(p.read_text())
tok = Tokenizer.from_file('.ds41-tokenizer/tokenizer.json')
for case in golden['cases']:
    case['ids'] = tok.encode(case['text'], add_special_tokens=False).ids
golden['eos_id'] = tok.token_to_id('<｜end▁of▁sentence｜>')
p.write_text(json.dumps(golden, ensure_ascii=False, indent=1) + '\n')
PY
```

## Runtime evidence

### Direct launch: blocked

```sh
.venv-ds41/bin/python serve/server.py --engine strata --config /tmp/ds41-servepy-evidence/config.json --port 18095
```

The process exited before startup. The server's existing port-check error message treats any bind error as an occupied port. The direct bind and protocol test identify the underlying error as sandbox permission denial. No request reached a TCP listener.

```text
usage: server.py [-h] [--engine {mock,strata}] [--config CONFIG] [--host HOST]
                 [--script SCRIPT] [--port PORT] [--gpu GPU]
                 [--tokenizer TOKENIZER] [--open] [--fit-max-tokens]
                 [--api-key API_KEY] [--mcp-config MCP_CONFIG] [--lazy]
                 [--api-monitor] [--idle-unload SECONDS]
                 [--min-free-vram-mib MIN_FREE_VRAM_MIB]
                 [--before-load BEFORE_LOAD]
server.py: error: port 18095 is already in use - is Strata (or another server) already running? Close it, or start this one with a different --port
```

### Simulated traffic: captured, not live TCP

The following request used the real OpenAI handler, service, tokenizer, renderer, detokenizer, DSML parser, and fake-engine child pipes. The HTTP input and output were in-memory streams. The disconnect watcher was disabled for that stream. The request did not use a socket. These are actual captured bytes and log lines from that execution, not hand-written expected output.

Request:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Find 台北."
    }
  ],
  "max_tokens": 512,
  "stream": true,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "lookup",
        "description": "Find a city.",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string"
            },
            "count": {
              "type": "integer"
            }
          }
        }
      }
    }
  ]
}
```

Server output (`/tmp/ds41-servepy-evidence/pipe-server.log`):

```text
[strata] starting the engine: reading the model's weights ...
[strata] done: 124 tokens in 0 s (197829.5 tok/s) (stop, cancel=False)
```

The narrator says it reads weights, but this fixture loads no weights. The fake DONE timing is fixed at 1 ms. The printed rates have no inference-performance meaning.

Child log (`/tmp/ds41-servepy-evidence/pipe-engine.log`):

```jsonl
fake-ds41: ready; deterministic golden output; no model weights
{"engine": "fake-ds41", "event": "generation", "prompt_tokens": 292, "prompt_sha256": "7e563a75f58066b3f15cdc7cb1f088683bc29be7e1a53edb2dfc29134e966b38", "output_tokens": 124, "eos": true}
```

The generation event records 292 prompt tokens and 124 output tokens. The last output id is EOS. The prompt hash links this event to the encoded request. No spool or database exists in this path.

Raw SSE (`/tmp/ds41-servepy-evidence/pipe-response.sse`):

```text
data: {"id": "chatcmpl-bb7ab6a969f042e8b8886614", "object": "chat.completion.chunk", "created": 1791349681, "model": "deepseek-v4.1-flash", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]}

: keep-alive

data: {"id": "chatcmpl-bb7ab6a969f042e8b8886614", "object": "chat.completion.chunk", "created": 1791349681, "model": "deepseek-v4.1-flash", "choices": [{"index": 0, "delta": {"reasoning_content": "Need"}, "finish_reason": null}]}

data: {"id": "chatcmpl-bb7ab6a969f042e8b8886614", "object": "chat.completion.chunk", "created": 1791349681, "model": "deepseek-v4.1-flash", "choices": [{"index": 0, "delta": {"reasoning_content": " a"}, "finish_reason": null}]}

data: {"id": "chatcmpl-bb7ab6a969f042e8b8886614", "object": "chat.completion.chunk", "created": 1791349681, "model": "deepseek-v4.1-flash", "choices": [{"index": 0, "delta": {"reasoning_content": " lookup"}, "finish_reason": null}]}

data: {"id": "chatcmpl-bb7ab6a969f042e8b8886614", "object": "chat.completion.chunk", "created": 1791349681, "model": "deepseek-v4.1-flash", "choices": [{"index": 0, "delta": {"reasoning_content": ".\n"}, "finish_reason": null}]}

data: {"id": "chatcmpl-bb7ab6a969f042e8b8886614", "object": "chat.completion.chunk", "created": 1791349681, "model": "deepseek-v4.1-flash", "choices": [{"index": 0, "delta": {"content": "Checking"}, "finish_reason": null}]}

data: {"id": "chatcmpl-bb7ab6a969f042e8b8886614", "object": "chat.completion.chunk", "created": 1791349681, "model": "deepseek-v4.1-flash", "choices": [{"index": 0, "delta": {"content": " "}, "finish_reason": null}]}

data: {"id": "chatcmpl-bb7ab6a969f042e8b8886614", "object": "chat.completion.chunk", "created": 1791349681, "model": "deepseek-v4.1-flash", "choices": [{"index": 0, "delta": {"content": "台北"}, "finish_reason": null}]}

data: {"id": "chatcmpl-bb7ab6a969f042e8b8886614", "object": "chat.completion.chunk", "created": 1791349681, "model": "deepseek-v4.1-flash", "choices": [{"index": 0, "delta": {"content": ".\n"}, "finish_reason": null}]}

data: {"id": "chatcmpl-bb7ab6a969f042e8b8886614", "object": "chat.completion.chunk", "created": 1791349681, "model": "deepseek-v4.1-flash", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_2ea9f6f6cc8c447c83207f75", "type": "function", "function": {"name": "lookup", "arguments": ""}}]}, "finish_reason": null}]}

data: {"id": "chatcmpl-bb7ab6a969f042e8b8886614", "object": "chat.completion.chunk", "created": 1791349681, "model": "deepseek-v4.1-flash", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{"}}]}, "finish_reason": null}]}

data: {"id": "chatcmpl-bb7ab6a969f042e8b8886614", "object": "chat.completion.chunk", "created": 1791349681, "model": "deepseek-v4.1-flash", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"city\":\"台北 \\\"true\\\"\\n🦊\""}}]}, "finish_reason": null}]}

data: {"id": "chatcmpl-bb7ab6a969f042e8b8886614", "object": "chat.completion.chunk", "created": 1791349681, "model": "deepseek-v4.1-flash", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": ",\"literal\":\"123\""}}]}, "finish_reason": null}]}

data: {"id": "chatcmpl-bb7ab6a969f042e8b8886614", "object": "chat.completion.chunk", "created": 1791349681, "model": "deepseek-v4.1-flash", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": ",\"count\":2"}}]}, "finish_reason": null}]}

data: {"id": "chatcmpl-bb7ab6a969f042e8b8886614", "object": "chat.completion.chunk", "created": 1791349681, "model": "deepseek-v4.1-flash", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": ",\"active\":true"}}]}, "finish_reason": null}]}

data: {"id": "chatcmpl-bb7ab6a969f042e8b8886614", "object": "chat.completion.chunk", "created": 1791349681, "model": "deepseek-v4.1-flash", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": ",\"extra\":{\"a\": [null, 1]}"}}]}, "finish_reason": null}]}

data: {"id": "chatcmpl-bb7ab6a969f042e8b8886614", "object": "chat.completion.chunk", "created": 1791349681, "model": "deepseek-v4.1-flash", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]}, "finish_reason": null}]}

data: {"id": "chatcmpl-bb7ab6a969f042e8b8886614", "object": "chat.completion.chunk", "created": 1791349681, "model": "deepseek-v4.1-flash", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 292, "completion_tokens": 124, "total_tokens": 416, "prompt_tokens_details": {"cached_tokens": 0}}, "timings": {"cache_n": 0, "prompt_n": 292, "prompt_ms": 1.0, "prompt_per_token_ms": 0.003, "prompt_per_second": 292000.0, "predicted_n": 124, "predicted_ms": 1.0, "predicted_per_token_ms": 0.008, "predicted_per_second": 124000.0}}

data: [DONE]

```

## Remaining acceptance and open questions

1. On a host that permits loopback sockets, run the full serve suite and `serve.test_deepseek_protocol.ProtocolTests` with `DS41_TOKENIZER_DIR` set. The latter starts `serve/server.py` with a DeepSeek config and checks actual HTTP traffic.
2. Start the fake-engine config and capture a live OpenAI SSE request with a tool. The in-memory evidence above does not replace this requested acceptance step.
3. When the real engine is available, check its CLI, READY/INFO/DONE fields, sampling keys, STOP behavior, and budget resume on a GPU host. No real-engine compatibility measurement was possible here.
4. Apply the three commit units when git can write its index.

No product question blocks the implemented Python paths. The remaining blockers are protected socket/git operations and unavailable real-engine hardware.

## Review changes (Claude, 2026-10-07)

- The server passes `--serve` to `ds41_serve`, as it does to `strata`. `ds41_serve` refuses to start without it.
  The fake engine now requires it too, so the protocol tests check that the server sends it.
- `engine_args` no longer has an allowlist for DeepSeek. It returns the config's args and adds nothing.
  `ds41_serve` refuses every flag it does not have (exit code 2, the flag named in the log), so the check stays
  in one place and the engine's tuning flags (`--vram-slots`, `--prefill-chunk`, ...) work from the config.
- Outside the sandbox, on macOS with Python 3.13: the full serve suite runs 287 tests with 1 error and 9 skips.
  main runs 268 tests with the same 1 error (`test_responses.OverHttp.test_json_schema_text_format`, failing on main
  before this branch) and 9 skips. The 19 new DeepSeek tests pass with `DS41_TOKENIZER_DIR` set.
