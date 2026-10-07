# DeepSeek installer progress

- [x] Check worktree, branch and existing changes. Branch: `feature/ds41-setup`. Only supplied Engram files were untracked.
- [x] Read all 4,422 original lines of `setup.py`, the pack tests, and the installer test patterns.
- [x] Run all 18 existing setup scripts before changes. Keep full output and the golden SHA-256.
- [x] Write failing pack tests. Add verified Engram copies without heavy imports. Three new and nine existing tests pass.
- [x] Write failing installer tests. Add Linux/NVIDIA gates, deliberate RAM risk acceptance, disk checks and strict downloads.
- [x] Add source build, pack, config and run-script paths. Skip Qwen model preparation and runtime flags.
- [x] Add check, settings summary, config family recovery, update, second start and previous-config adoption.
- [x] Run 23 new mocked installer tests. All pass without GPU or network.
- [x] Run accepted mocked traffic. First install and second start return 0. Keep config, script, commands and mock log events.
- [x] Update `docs/AI_SETUP.md` and `docs/MODELS.md` with measured facts and limits.
- [x] Run all existing setup scripts after changes. Same four pre-existing failures; no new failure. Full output retained.
- [x] Compare all 25 Qwen/Unsloth config records before/after. All unchanged.
- [x] Keep `tools/test_setup_golden.json` byte-identical. Fixture-normalized golden passes 4/4 before and after.
- [x] Review the diff and write `SETUP.REPORT.md` with commands, outputs, component status and evidence.
- [x] Commit the first unit, including both supplied Engram files. Commit: `ac6e7eb`, four files.
- [x] Write the requested fallback commit plan. All units have at most five files and the required footer.
- [ ] Commit the remaining units. The sandbox denied creation of the worktree `index.lock` on the second unit. No bypass attempted.
- [ ] Run actual CUDA build and model inference. No NVIDIA GPU or 341.8 GB checkpoint is available here. The current checkout also lacks the other developers' `ds41_serve` target and server format selector. The user-defined mocked traffic check is complete.
- [ ] Make every raw existing setup test pass. Four scripts already failed before edits due to platform-dependent or incomplete mocks. They have the same outcomes after edits. Fixing unrelated fixtures is outside this task; the report identifies each failure.

No pending user answer is required. Code, docs, test output and the commit plan remain in the worktree.
