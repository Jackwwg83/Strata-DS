# 20 outputs read in full (defaults: route + SSD tail skip), 2026-10-15

Picked at random (HumanEval+: random.Random(7).sample(164, 10); GSM8K: random.Random(8).sample(200, 10)) from
he_ev_c.jsonl and gsm_ev_c.jsonl (greedy, thinking off, English). Counted: repetition loops, garbled text, a switch
of language, truncation.

| Output | Finish | Repetition | Garbled | Language switch | Note |
| --- | --- | ---: | ---: | ---: | --- |
| HumanEval/12 | stop | 0 | 0 | 0 |  |
| HumanEval/14 | stop | 0 | 0 | 0 |  |
| HumanEval/18 | stop | 0 | 0 | 0 |  |
| HumanEval/24 | stop | 0 | 0 | 0 |  |
| HumanEval/38 | stop | 0 | 0 | 0 |  |
| HumanEval/82 | stop | 0 | 0 | 0 |  |
| HumanEval/93 | stop | 0 | 0 | 0 |  |
| HumanEval/101 | stop | 0 | 0 | 0 |  |
| HumanEval/137 | stop | 0 | 0 | 0 |  |
| HumanEval/149 | stop | 0 | 0 | 0 |  |
| gsm8k/11 | stop | 0 | 0 | 0 |  |
| gsm8k/21 | stop | 0 | 0 | 0 |  |
| gsm8k/32 | stop | 0 | 0 | 0 |  |
| gsm8k/35 | stop | 0 | 0 | 0 |  |
| gsm8k/49 | stop | 0 | 0 | 0 |  |
| gsm8k/58 | stop | 0 | 0 | 0 | weighs two readings of "final price" before answering; answer correct (57) |
| gsm8k/63 | stop | 0 | 0 | 0 |  |
| gsm8k/94 | stop | 0 | 0 | 0 |  |
| gsm8k/96 | stop | 0 | 0 | 0 |  |
| gsm8k/180 | stop | 0 | 0 | 0 |  |

Total: 20 read, 0 repetition, 0 garbled, 0 language switch, 0 truncated. Not covered: thinking-on output and
Chinese output (the timing client of the speed runs does not keep text).
