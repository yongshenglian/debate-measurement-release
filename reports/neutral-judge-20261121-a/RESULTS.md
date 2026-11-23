# Study J results: does the "your earlier solution" framing cost the reader?

- **Protocol:** `PROTOCOL.md`. The five frozen files were re-checked against `frozen.sha256` after the run: all OK.
- **Outputs:** `out/`, 4,464 judge calls, every container exited 0, no invalid final answers in any condition.
- **Analyses:**
  - `analysis.out` is the pre-registered analysis (`j_analyze.py`); `analysis.json` holds its numbers.
  - `explore.out` is a **post-hoc, exploratory** split of the framing effect into label and position (`j_explore.py`).
- **GPU:** about 2.54 GPU-h, just over the 2.5 cap; the estimate was 1.3.
  - Gemma lane: 00:02:58 → 01:58:44, 1.93 h. Gemma on MATH alone took 59 min (long 2,048-token reviews).
  - Qwen lane: 00:02:58 → 00:39:41, 0.61 h.
  - The launcher had per-container timeouts but no lane-level cap. It finished slightly over budget instead of stopping at it.
- **Reproducibility:** the owner condition re-runs the original prompt with the same seeds. It reproduced the original final answers on 77–92% of problems (per task and judge). That is the size of vLLM batch nondeterminism on these long reviews. Every comparison below is within this run.

## Primary (Holm over 6 per family)

- **J1, neutral − owner accuracy on the disagreement set D:**

  | Task | Gemma judge | Qwen judge |
  |---|---|---|
  | GSM8K (D = 99) | +2.0 [−6.1, +10.1] | −7.1 [−17.2, +4.0] |
  | MATH (209) | +2.9 [−1.4, +7.7] | −2.9 [−9.1, +3.3] |
  | MMLU-Pro (250) | **+6.8 [+2.0, +11.6], Holm .047** | −1.6 [−6.8, +3.6] |

  - **Pre-registered reading: "owner framing costs the reader" holds for the strong judge on MMLU-Pro.**
  - The Gemma point estimate is positive on all three tasks. The Qwen one is negative on all three, none significant.
- **J2, system** (Gemma neutral judge on D, Gemma s0 elsewhere):

  | Task | − Gemma s0 (J2a) | − vote3 (J2b) |
  |---|---|---|
  | GSM8K | +0.5 [−0.3, +1.3] | −0.3 [−1.1, +0.5] |
  | MATH | +1.0 [−0.5, +2.7] | +0.5 [−1.2, +2.2] |
  | MMLU-Pro | +0.2 [−2.0, +2.3] | −1.0 [−3.2, +1.2] |

  - All Holm = 1.
  - **Pre-registered reading: a neutral judge does not capture the complementarity, and does not beat compute-matched voting.**
  - What neutral framing does is remove the loss that owner framing causes. On MMLU-Pro the owner-judge system is −2.7 [−5.2, −0.3] vs Gemma s0; the neutral-judge system is +0.2.

## Secondary

- **Accept/abandon, Gemma judge.** Neutral framing lowers both rates; abandonment falls more.

  | Task | Owner abandon | Owner accept | Neutral abandon | Neutral accept |
  |---|---|---|---|---|
  | GSM8K | 26.6% | 85.7% | 17.2% | 66.7% |
  | MATH | 11.9% | 58.8% | 6.7% | 52.9% |
  | MMLU-Pro | 30.8% | 41.3% | 18.8% | 39.1% |

  - This is the same dial as the independence sentence (study I), moved by the label alone.
- **J3, retention of the judge's own answer, label_swap − owner:**
  - Gemma: +18.2, +5.7, +25.6 (GSM8K, MATH, MMLU-Pro).
  - Qwen: −19.2, +1.9, +32.8.
  - As pre-registered, J3 mixes label with position (label_swap moves own to the second block). The exploratory split below separates them.
- **Position, under neutral labels:**

  | Task | Gemma picks block A | Qwen picks block A |
  |---|---|---|
  | GSM8K | 44.9% | 62.1% |
  | MATH | 45.2% | 53.3% |
  | MMLU-Pro | 40.4% | 35.2% |

  - Gemma prefers the second solution on every task.
  - Qwen prefers the first on GSM8K and the second on MMLU-Pro.
  - Position effects are as large as the label effect, and their sign depends on model and task.

## Exploratory: label vs position (post hoc)

- **Measure:** at fixed content order, the change in how often the reader picks the first-block solution when it is labelled "Your earlier solution" instead of "Solution A".
  - owner vs neutral: the judge's own solution is first;
  - label_swap vs neutral_swap: the other model's solution is first.

| Task | Gemma: on own | Gemma: on other's | Qwen: on own | Qwen: on other's |
|---|---|---|---|---|
| GSM8K | −11.1 | −3.0 | −6.1 | −4.0 |
| MATH | −1.9 | −3.8 | −8.6 | −9.6 |
| MMLU-Pro | −7.2 | −2.8 | −5.2 | −1.6 |

- **Pooled:** Gemma −4.6 [−6.8, −2.5]; Qwen −5.8 [−8.8, −3.0] (n = 558 each).
- **Reading:** all 12 estimates are negative.
  - Calling a solution "yours" makes both models about 5 pp *less* likely to end on it, whoever actually wrote it.
  - This is the opposite of the self-preference reported for LLM judges. "Review your earlier solution" reads as an invitation to revise it.
  - The cost falls on the strong reader: its labelled answer is usually the right one.
- **Status:** a post-hoc analysis, to be confirmed. A pre-registered replication on one more task or model pair would settle it.

## What this changes

- **Design rule:** present peer solutions under neutral labels, and average or randomise their order. Owner framing alone cost the strong reader up to 6.8 pp on disagreements.
- **The ceiling stays.** Even neutral, the two-model system does not beat the strong model alone or a 3-call vote. Reading captures no complementarity at 7B/27B on these tasks.
