# Tier review commit plan

Branch: `fix/ds41-revfix-tiers`.
Base: `e2d84995f450a21fefcc214e25e0075128b908f6`.

These commits succeeded. Each ends with `Co-Authored-By: Codex <noreply@openai.com>`.

| Commit | Files | Change |
|---|---:|---|
| e2224a5 | 5 | Regression tests and host fault injection |
| b88d1d1 | 2 | Pack ranges, shapes, hash arrays, and paths |
| a902916 | 2 | Poison failed Engram readers |
| e3e1e94 | 5 | Residency synchronization and VRAM constructor cleanup |
| 4ba3f9f | 2 | ExpertStream constructor cleanup |

The sandbox refused the final git add. It denied creation of the worktree index.lock.
No lock was removed. No permissions were changed.
Only the following two documentation files remain to commit:

```sh
cd /Users/jackwu/Projects/Strata-DS-revfix-tiers
git add ds41/tasks/REVFIX.REPORT.md ds41/tasks/REVFIX.COMMITS.md
git commit -m 'docs(ds41): record tier fixes and validation evidence' -m 'List each fix and its regression. Record local output and exact RTX 4090 commands. Mark CUDA and sanitizer limits.' -m 'Co-Authored-By: Codex <noreply@openai.com>'
```

Keep `ds41/tasks/rev_tiers_out.md` and `ds41/tasks/rev_kernels_out.md` untracked.
They were present when this task started. Do not include them in this commit.
