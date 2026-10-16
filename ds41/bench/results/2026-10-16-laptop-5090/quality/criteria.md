# Decision criteria, written before the results

Written 2026-10-15 17:47 (laptop time), before any task evaluation or needle test had run (the first, HumanEval+
without skips, started 18:02). One exception: the KL divergence criterion was written after the default-vs-no-skip
KL result (0.0316) had been seen; the noise floor was not yet known.

"Clearly worse" means stop and report before publishing. The defaults are clearly worse if any of these holds:

1. HumanEval+ or GSM8K: default minus no-skip accuracy below -3 points AND exact McNemar p below 0.05.
2. Needle in a haystack: the default finds at least 4 fewer of the 60 than no skip, or at least 3 fewer of the 15 in
   any length.
3. KL divergence: the default's mean KL from no skip is more than 3 times the noise floor (no skip vs no skip) AND
   the same top token is below 90%.

Otherwise the numbers are reported with their intervals as the measured cost, without a verdict.

## Outcome

| Criterion | Measured | Met? |
| --- | --- | --- |
| 1, HumanEval+ | -1.2 points [-3.7, +1.2], p 0.63 | no |
| 1, GSM8K | +0.5 points [0.0, +1.5], p 1.0 | no |
| 2, needle | 60 of 60 both; every length 15 of 15 both | no |
| 3, KL, all text | 0.0316 = 2.65 x the floor 0.0119; same top token 91.8% | no |

Not part of the criteria, reported anyway: by text kind, English is at 2.9 x its floor with 89.7% same top token,
and Chinese at 2.7 x with 87.1%. The criterion was written for all text together.
