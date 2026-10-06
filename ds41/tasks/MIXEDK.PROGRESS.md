# Mixed-K progress

- [x] Read AGENTS.md and K10/K11/K12. Check branch and clean worktree.
- [x] Write K10 device job dispatch for independent K1..K6 projections.
- [x] Write K12 device reconstruction dispatch for independent K1..K6 projections.
- [x] Audit CPU registration, replacement, and AVX dispatch. Existing CPU rates are per matrix.
- [x] Replace both fixed tile-width asserts with integer K1..K6 validation.
- [x] Extend K10/K11/K12 tests. Keep fixed cases, timing output, and tolerances.
- [x] Add independent LinearEXL3 synthetic fixture generation for K10/K11.
- [x] Update the reversible vendor patch. Verify all 15 pristine hashes.
- [x] Run K10/K12 host layout, indexing, arithmetic, routing, and schedule checks.
- [x] Compile K11 test object and native ARM stub. Compile extracted x86 bands.
- [x] Run native extracted CPU registration/relocation checks and synthetic fixture reader.
- [x] Record real check output, per-file changes, dispatch, launch counts, and GPU commands.
- [ ] Compile CUDA and the production engine. Blocked: no nvcc or CUDA toolkit.
- [ ] Run GPU golden, graph, timing, and SAGE integration checks. Blocked: no CUDA GPU or real pack here.
- [ ] Confirm no 3-bit speed regression. Blocked: needs paired GPU measurements against the starting commit.
- [ ] Run full CPU golden and AVX2/AVX-512 numerical tests. Blocked: no native x86 target; translated x86 reports no AVX2/FMA.
- [ ] Commit in units of at most five files. Blocked: protected Git index outside writable roots. Exact six-unit commands are in MIXEDK.COMMITS.md.

Every item was reviewed. Written code and host checks do not establish GPU
acceptance. MIXEDK.REPORT.md gives separate states and the remaining commands.
