# Python server progress

- [x] Inspect the clean worktree on `feature/ds41-serve-py` before edits.
- [x] Read the server, frontend, Responses API, tokenizer, reference encoder, and serve tests.
- [x] Add HF tokenizer golden tests. Observe the missing adapter failure. Implement byte BPE and streaming decode.
- [x] Add renderer and normalization tests. Observe failures. Implement the reference adapter.
- [x] Add parser boundary tests. Observe failures. Implement incremental DSML events.
- [x] Add config and constants tests. Observe failures. Connect the DeepSeek switch.
- [x] Add budget and image edge tests. Observe failures. Fix EOS, wrap-up, and early image rejection.
- [x] Add a fake line engine with real DeepSeek token ids from HF golden data.
- [x] Check OpenAI, Responses, and Anthropic in stream and non-stream modes through the real handlers and child process.
- [x] Check `main()` configuration and child startup with the TCP listener replaced by a test double.
- [x] Run all existing serve tests without editing their assertions. Compare the unchanged baseline.
- [ ] Obtain a fully passing existing serve suite. The sandbox denies socket operations. Both versions have 119 PermissionError errors. The existing code has 97 passing tests, zero assertion failures, and 9 skips here.
- [ ] Start an actual HTTP listener with `serve/server.py` and send the requested live SSE request. The sandbox denies `bind(127.0.0.1)`. Approval is unavailable. The direct launch exits during its port check.
- [x] Capture raw SSE, server output, and a real child-process log event from the in-memory HTTP check. Mark this as simulated traffic, not live TCP evidence.
- [x] Write the file-by-file report, exact commands, effort map, evidence, and remaining acceptance steps.
- [ ] Create git commits. The sandbox denies creation of the worktree index lock.
- [x] Write the required fallback commit plan. Each unit has at most five files and the required trailer.
- [x] Confirm no changes to the reference encoder, engine, setup, docs, or existing test assertions.
- [x] Review every checklist item. Only the protected socket and git operations remain blocked.

The real engine is unavailable on this machine. There is no GPU validation. All generated model output in this task is deterministic fake-engine output.
