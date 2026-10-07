# DeepSeek installer report

## Result and limits

The installer and the numpy-only pack path are implemented. The accepted mocked traffic check passed.
The first installation and a second plain start both returned 0. The second start did not download or pack again.
The harness read the written config at the server boundary and wrote two explicit mock events.

I cannot run real model traffic here because the 341.8 GB checkpoint and an NVIDIA GPU are unavailable.
No real inference server is deployed. No throughput was measured in this task.
The checkpoint size and hardware measurements in the docs come from the task's supplied measurements.

The current checkout also lacks the `ds41_serve` target and the server's `deepseek_v41` selector.
`rg -n 'ds41_serve|deepseek_v41' CMakeLists.txt cmake serve/server.py` returned no matches.
These are the other developers' integration changes. This task does not edit the engine, CMake, or server.
The installer writes the agreed contract and requests the agreed build target. Actual integration needs those changes.

## Component status

| Component | Code implementation | Deployment path | Data status |
| --- | --- | --- | --- |
| Engram copy | SHA-256 checks and `--engram-from` complete | `tools/ds41/pack.py` | Real supplied hash files; actual synthetic checkpoint packed; no heavy imports |
| Installer | Family, hardware gates, disk accounting, pins, pack, config and lifecycle complete | `setup.py` | Actual config and shell script written by `main()` in the harness |
| Download | Existing resumable `download()` with `.done` marks; strict SAGE pin | `<models-dir>/deepseek-sage-1.59bpw` | Mock file listing and mock small files; no checkpoint download |
| Engine | Source-build command and binary copy implemented | `engine/ds41_serve`; separate `engine/DS41_BUILD.json` | Mock build and placeholder binary; CUDA build not run |
| Server boundary | Config uses `format: deepseek_v41`; no Qwen engine flags | `serve/server.py --engine strata --config ...` | Mock call consumed the actual config; no HTTP process or inference |
| Logs | Harness event logging complete | Evidence folder's `strata-deepseek-sage-1.59bpw.log` | Two actual log rows labelled `mock: true`; no production spool/db data |
| Library directories | Config field and update path complete | `lib_dirs` in config | Empty in this mock because `/mock/cuda` has no libraries; not proof of runtime linkage |

## Files

- `tools/ds41/pack.py`: add `--engram-from DIR`. Verify both pinned hashes, then copy both files.
  Keep the original computing path. Refuse `--wo-a-bf16` with the new flag because that conversion imports torch.
- `ds41/data/engram/engram_hash.txt`: include the supplied SAGE hash table.
  SHA-256: `a55179fea918e66a331ac4cc1bdf9a0b947117dd6478f74d470de2a7e42a5c49`.
- `ds41/data/engram/engram_tokenmap.bin`: include the supplied SAGE token map.
  SHA-256: `c60a86322ec17b4142bfef3c57a8d81fb428550cdf487f88a4320cb59fe46481`.
- `tools/ds41/test_pack_engram.py`: three tests use a tiny safetensors checkpoint written with the standard library.
  A subprocess blocks torch, transformers and sympy imports. Test copied bytes, completion, corruption and missing files.
- `setup.py`: register `deepseek`, `SAGE-1.59BPW`, and the pinned repository. Add isolated install and start paths.
  Add source build caching, update handling, config adoption, source reuse and Engram source checks.
- `tools/test_setup_deepseek.py`: 23 tests. Mock GPU, RAM, CPU, network, build, pack and server calls.
  Run the real installer flow, config writer, run-script writer and CMake command builder.
  `--evidence DIR` retains a first installation and a second start for review.
- `tools/test_setup_pins.py`: check image URLs only for families that provide them.
  Require `vision: false` when no image URL exists. Keep the Qwen pin assertions unchanged.
- `docs/AI_SETUP.md`: add the family, flags, measured hardware, disk costs, source retention and setup commands.
- `docs/MODELS.md`: add SAGE and its limits. Keep Qwen as the recommendation.
- `SETUP.BASELINE.log` and `SETUP.AFTER.log`: complete raw setup test output, commands and exit codes.
- `SETUP.QWEN.BEFORE.json`: retain the 25 config/exit results before installer changes.
- `SETUP.GOLDEN.log`: record before/after golden results with fixture-compatible executable normalization.
- `SETUP.PACK.log`: record the 12 passing pack tests.
- `SETUP.TDD.log`: retain the failing test stages before each implementation change.
- `SETUP.EVIDENCE.json`: retain command arrays, download URLs, config, script, stdout and mock event rows.
- `SETUP.PROGRESS.md`: checked task list and explicit unavailable validations.
- `SETUP.COMMITS.md`: four commit units, each with at most five files.
- `SETUP.REPORT.md`: this report.

## Design choices

1. Dispatch explicit DeepSeek installs before Qwen hardware sizing. Dispatch DeepSeek starts before Qwen config
   upgrades and multi-GPU changes. This prevents unknown Qwen flags from reaching the strict DeepSeek engine.
2. Keep Qwen first in the family menu. Hide DeepSeek on incompatible interactive hardware.
   Explicit DeepSeek requests fail on Windows, Mac, AMD, GPUs below the VRAM gate, or compute capability below 8.6.
   A nominal 16 GB card may report 15.99 GiB; the gate allows 15.9 GiB for this reporting difference.
3. Recommend a 128 GB PC. Warn below 120 GiB usable RAM. Interactive continuation defaults to no.
   `--family deepseek --yes` is explicit consent under the installer's existing risk rule.
   This also warns for the supplied measured 119.9 GiB machine, as the requested threshold requires.
4. Keep the SAGE revision strict. Qwen's legacy fallback to `main` remains unchanged.
   Read the file listing at the pinned API revision. Select root JSON, tokenizer files and model shards.
   Verify all 17 expected shard names and the required config/tokenizer names. Cache that listing for retries.
5. Reserve 341.8 GB for source plus 120.2 GB for pack: a rounded 462 GB disk budget.
   Subtract source and partial-download bytes already present. Check each filesystem when source and pack differ.
   A completed matching pack needs no new pack reservation. Keep every source shard on disk.
6. Build only `ds41_serve`. Do not download a release zip that cannot supply it.
   Cache the source fingerprint and GPU architecture in `DS41_BUILD.json`, separate from Qwen's `BUILD.json`.
   Track CMake and the CPU expert sources too. Preserve the CUDA selection through updates.
   A failed automatic update can start an existing compatible binary. An explicit failed update returns failure.
7. Offer existing contexts through 262144, with 32768 default. Do not apply RoPE or Qwen KV formulas.
   Use physical performance cores on a hybrid CPU, else physical cores, capped at 16. Use the existing psutil
   dependency only as a fallback when Linux topology files give no count.
8. Reuse the common atomic config writer and run-script writer. Tokenizer points at the pack directory.
   Preserve user config fields through the existing carry-over rules. Require a key for a non-loopback host.
9. A new checkout can adopt a previous DeepSeek config and reuse its source and pack.
   Engram paths are absolute. A moved source requires repacking; startup rejects missing Engram shards.

## TDD and commands

Read all 4,422 original lines of `setup.py` in chunks before changing it. Read the existing pack, Unsloth and
golden tests first. Run all 18 existing setup test scripts before installer changes.

The first pack test run failed because `--engram-from` did not exist. The first installer run failed because
`--family deepseek` did not exist. Later failing runs covered early host validation, missing Engram source,
update library paths, source fingerprints, failed updates, CUDA persistence, evidence events and config adoption.
See [SETUP.TDD.log](SETUP.TDD.log). All final new tests pass.

Run every setup script, before and after, with this command pattern:

```python
import pathlib, subprocess
for test in sorted(pathlib.Path("tools").glob("test_setup_*.py")):
    result = subprocess.run(["python3", str(test)], capture_output=True, text=True)
    print(test, result.returncode, result.stdout, result.stderr)
```

The actual commands and complete merged output are in [SETUP.BASELINE.log](SETUP.BASELINE.log) and
[SETUP.AFTER.log](SETUP.AFTER.log). The new DeepSeek file did not exist in the baseline run.

| Exact command | Tests run | Before | After |
| --- | ---: | --- | --- |
| `python3 tools/test_setup_amd.py` | 28 | FAILED (failures=1) | FAILED (failures=1) |
| `python3 tools/test_setup_choices.py` | 35 | FAILED (errors=2) | FAILED (errors=2) |
| `python3 tools/test_setup_config.py` | 13 | OK | OK |
| `python3 tools/test_setup_draft_vocab.py` | 7 | OK | OK |
| `python3 tools/test_setup_golden.py` | 4 | FAILED (failures=46) | FAILED (failures=46) |
| `python3 tools/test_setup_hybrid.py` | 4 | OK | OK |
| `python3 tools/test_setup_lowram.py` | 5 | OK | OK |
| `python3 tools/test_setup_oldcpu.py` | 7 | OK (skipped=1) | OK (skipped=1) |
| `python3 tools/test_setup_older_gpus.py` | 19 | OK | OK |
| `python3 tools/test_setup_parallel.py` | 3 | OK | OK |
| `python3 tools/test_setup_pins.py` | 16 | OK | OK |
| `python3 tools/test_setup_prompts.py` | 7 | OK | OK |
| `python3 tools/test_setup_remote_opt.py` | 6 | OK | OK |
| `python3 tools/test_setup_risk.py` | 34 | OK | OK |
| `python3 tools/test_setup_rope.py` | 19 | OK | OK |
| `python3 tools/test_setup_sycl.py` | 6 | OK | OK |
| `python3 tools/test_setup_unsloth.py` | 37 | FAILED (errors=1) | FAILED (errors=1) |
| `python3 tools/test_setup_update.py` | 8 | OK | OK |
| `python3 tools/test_setup_deepseek.py` | 23 | Absent | OK |

Raw before: 18 scripts, 258 unittest methods, 14 successful scripts.
Raw after: the same 18 scripts and 258 methods have the same outcomes; the 23 new methods also pass.
The full after run has 19 scripts, 281 methods, and 15 successful scripts.
The golden failure count is 46 subtest failures in two methods, not 46 additional test methods.

The four raw failures predate the change:

- `test_setup_amd.py`: the Windows ZIP fixture contains `strata.exe`; this macOS import uses `EXE = strata`.
- `test_setup_choices.py`: the CPU encoder fixture does not mock CPU detection. This non-x86 host changes the
  ISA floor, so it enters a real HIP rebuild path. Its attempted pip install stops at the managed Python guard.
- `test_setup_golden.py`: its global string replacement turns `strata-*.log` into `<EXE>-*.log` when EXE is `strata`.
- `test_setup_unsloth.py`: one AMD fixture does not mock the HIP build. It tries to write a ROCm stamp under the
  protected Python prefix and gets `PermissionError`.

No Qwen runtime change was made to work around these fixture problems.
For an independent golden comparison, set only `setup.EXE = "strata.exe"` before running the unchanged golden test.
This gives its normalizer the executable suffix expected by the checked-in fixture:

```python
import runpy, setup, sys
setup.EXE = "strata.exe"
sys.argv = ["tools/test_setup_golden.py"]
runpy.run_path(sys.argv[0], run_name="__main__")
```

Run this once against the original `4cb5af0` setup module and once against the edited module.
Both runs printed `Ran 4 tests` and `OK`. See [SETUP.GOLDEN.log](SETUP.GOLDEN.log).
Also compare `tools.test_setup_golden.record()` with [SETUP.QWEN.BEFORE.json](SETUP.QWEN.BEFORE.json):
all 25 exit/config records are equal. The golden baseline file stays byte-identical:

```
131f8e5f31061df2f209f49db86cc36f7b5c901424fa25795708b51c646107fd  tools/test_setup_golden.json
```

Pack command (use the already available Python environment with numpy, torch and safetensors for the legacy tests):

```sh
/private/tmp/claude-501/-Users-jackwu-Projects-Strata-DS/5590375a-bfc7-418f-86d0-9bbf31c371cd/scratchpad/venv/bin/python -W ignore::ResourceWarning -m unittest discover -s tools/ds41 -p 'test_pack*.py'
```

Output:

```
Ran 12 tests in 1.978s

OK
```

This is nine existing tests plus three new tests. See [SETUP.PACK.log](SETUP.PACK.log).
The new subprocess tests explicitly reject imports of torch, transformers and sympy.
No runtime dependency was added. A separate attempt to install numpy into a new temporary venv failed on DNS;
the tests then used the already available environment above.

## Accepted traffic evidence

Exact command:

```sh
python3 tools/test_setup_deepseek.py --evidence /private/tmp/strata-deepseek-setup-evidence-final
```

Output:

```
Mocked installer evidence: /private/tmp/strata-deepseek-setup-evidence-final
First run: 0; second run: 0; server boundary calls: 2
```

The harness runs `setup.main()` with these non-interactive arguments:

```sh
setup.py --family deepseek --resident-budget-gib 96 --context 32768 --port 8091 --no-browser --yes
```

It then runs a plain `setup.py --yes`. It does not mock config writing or shell-script writing.
It mocks the external build, download, pack and server boundaries. The separate pack tests execute the real packer.
The retained config, script and event below are the actual bytes produced by this harness. They are synthetic
installation evidence, not model inference evidence. Full details: [SETUP.EVIDENCE.json](SETUP.EVIDENCE.json).

Written config:

```json
{
 "exe": "/private/tmp/strata-deepseek-setup-evidence-final/engine/ds41_serve",
 "args": [
  "--pack",
  "/private/tmp/strata-deepseek-setup-evidence-final/data/packs/deepseek-sage-1.59bpw",
  "--max-context",
  "32768",
  "--expert-profile",
  "/private/tmp/strata-deepseek-setup-evidence-final/ds41/data/expert-profile.bin",
  "--threads",
  "8",
  "--ram-budget-gib",
  "96"
 ],
 "cwd": "/private/tmp/strata-deepseek-setup-evidence-final",
 "tokenizer": "/private/tmp/strata-deepseek-setup-evidence-final/data/packs/deepseek-sage-1.59bpw",
 "format": "deepseek_v41",
 "model_name": "deepseek-v4.1-flash",
 "log": "/private/tmp/strata-deepseek-setup-evidence-final/strata-deepseek-sage-1.59bpw.log",
 "lib_dirs": [],
 "port": 8091,
 "gpu": 0,
 "gpus_asked": true,
 "cuda": 13,
 "open_browser": false
}
```

Written executable run script:

```sh
#!/bin/sh
cd "/private/tmp/strata-deepseek-setup-evidence-final"
exec "/opt/homebrew/opt/python@3.14/bin/python3.14" "/private/tmp/strata-deepseek-setup-evidence-final/serve/server.py" "--engine" "strata" "--config" "/private/tmp/strata-deepseek-setup-evidence-final/strata-deepseek-sage-1.59bpw.json" "--port" "8091"
```

Observed CMake configure, build and pack command lines:

```sh
/mock/bin/cmake -G Ninja -DCMAKE_MAKE_PROGRAM=/mock/bin/ninja -S /private/tmp/strata-deepseek-setup-evidence-final -B /private/tmp/strata-deepseek-setup-evidence-final/build-ds41 -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/mock/cuda/bin/nvcc -DSTRATA_GGML_DIR=/private/tmp/strata-deepseek-setup-evidence-final/llama.cpp
/mock/bin/cmake --build /private/tmp/strata-deepseek-setup-evidence-final/build-ds41 --target ds41_serve -j 7
/opt/homebrew/opt/python@3.14/bin/python3.14 /private/tmp/strata-deepseek-setup-evidence-final/tools/ds41/pack.py --src /private/tmp/strata-deepseek-setup-evidence-final/data/models/deepseek-sage-1.59bpw --out /private/tmp/strata-deepseek-setup-evidence-final/data/packs/deepseek-sage-1.59bpw --engram-from /private/tmp/strata-deepseek-setup-evidence-final/ds41/data/engram
```

One row read from `strata-deepseek-sage-1.59bpw.log` after the second start:

```json
{"event": "mock_server_start", "mock": true, "config": "/private/tmp/strata-deepseek-setup-evidence-final/strata-deepseek-sage-1.59bpw.json", "model_name": "deepseek-v4.1-flash", "format": "deepseek_v41", "args": ["--pack", "/private/tmp/strata-deepseek-setup-evidence-final/data/packs/deepseek-sage-1.59bpw", "--max-context", "32768", "--expert-profile", "/private/tmp/strata-deepseek-setup-evidence-final/ds41/data/expert-profile.bin", "--threads", "8", "--ram-budget-gib", "96"]}
```

The mock server boundary reads the config file that `setup.py` wrote. The event records its actual model name,
format and arguments. There is no generated text, production request ID, spool entry, or database row.
The download listing in this fixture has 23 selected files, including all 17 shards. It is not the real Hub listing
or a claim that the supplied 36-file repository was downloaded.

## Remaining acceptance and open questions

- Merge the other developers' `ds41_serve` target and server `deepseek_v41` implementation.
  Then build and start this config on Linux + NVIDIA with the real pinned checkpoint.
- Verify runtime CUDA libraries, tokenizer loading, Engram source reads, one API request and actual engine logs.
  Those checks cannot run here. GPU/model availability is the concrete blocker.
- Windows and AMD remain intentionally unsupported. GPU inference under 24 GB VRAM and other low-RAM
  configurations remain unmeasured. Do not infer speed from the installer tests.
- The four pre-existing raw test fixture issues above remain. The unchanged Qwen records and normalized golden
  runs establish no config regression, but do not turn the raw suite into a fully passing suite.
- No owner decision is needed for the implemented scope.

## Commits

`ac6e7eb` committed the pack change, new pack tests and both supplied Engram files (four files).
The next `git add` failed because the sandbox refused the worktree's `index.lock`.
No alternate write path was used. All remaining code, docs and evidence remain in the worktree.
[SETUP.COMMITS.md](SETUP.COMMITS.md) lists the remaining units, each with at most five files and the requested footer.

## Review (Claude, 2026-10-07)

- Outside the sandbox, macOS, Python 3.13: `tools.test_setup_deepseek` 23 of 23 pass; the pack tests 12 of 12 pass
  (9 before). The full output of `test_setup_golden`, `test_setup_amd` and `test_setup_choices` is the same on main
  and on this branch, line for line except the line numbers in tracebacks. Those three fail on main on macOS and
  Linux: the golden test replaces the engine name `strata` with `<EXE>`, which also changes the log name
  `strata-<model>.log` on systems where the engine has no `.exe`.
- The docs said 24-29 tokens/s decode. That is code text. The chat requests through the server ran at 15-20 tokens/s
  (ds41/bench/results/2026-10-07-serve). Both are now in the docs, with the measured startup time (about 30 s).
