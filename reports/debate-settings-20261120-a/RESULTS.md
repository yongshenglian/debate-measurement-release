# Study S results: which debate settings make debate help or hurt

- **Run:** 2026-11-20. Qwen2.5-7B, the 300 shared GSM8K problems, 5 agents, rounds 0–3.
  - All 7 arms completed: 300 problems × 4 rounds × 5 agents each.
  - Frozen files verified intact after the run.
- **Compute:** about 3.6 GPU-h, against a 6 GPU-h cap.
  - Generation took 11,144 s; model loading and the two Gemma reader passes make up the rest.
- **Outputs:**
  - `analysis.out` and `analysis.json`: pre-registered.
  - `audit/score.out`: pre-registered reader audit.
  - `none-sens/score.out`: post-hoc sensitivity.

## Reader audit (pre-registered)

- **Method:** Gemma-3-27B blind extraction on 100 random outputs per arm.
- **Result:** every arm passes the 95% threshold.
  - Agreement ranges from 96.8% to 100% on outputs where our reader returns a number.
- **Gap the threshold does not cover:** in the DMAD-protocol arms, some outputs our reader scored as "no answer" were outputs where Gemma did read a number:

  | Arm | No-answer outputs Gemma read as a number |
  |---|---|
  | S2 | 11 / 100 |
  | S5 | 13 / 100 |
  | S1, S3, S4, S6, S7 | 1–6 / 100 |

  - Typical case: after debate, the agent wraps its answer in a sentence with a second number. Example: `<answer>The cost to fill the pool is $12.00, based on a cost of $0.10 per cubic foot.</answer>`.
- **Post-hoc sensitivity** (`collaboration/when_debate/settings_none_sens.py`):
  - Gemma read all 1,952 no-answer outputs, and its reading filled the gaps.
  - Both readings are reported below.

## Per arm (pre-registered reader; Gemma-filled reading in brackets)

| Arm | Change from S2 | Vote r0 → final | P1 final − r0 | P2 gate − r0 | Individual r0 → final |
|---|---|---|---|---|---|
| S1_dov | (whole DoV protocol) | 93.4 → 92.5 | −0.9 [−3.4, +1.6] *(−1.1)* | −0.2 *(−0.5)* | 84.1 → 89.3 |
| **S2_dmad** | — | 94.2 → 87.8 | **−6.3 [−9.3, −3.5]** *(−3.8 [−6.4, −1.4])* | **−3.8 [−6.3, −1.6]** *(−1.6 [−3.6, +0.2])* | 86.2 → 86.7 |
| S3_dmad_dovword | DoV debate wording | 94.2 → 93.7 | −0.5 [−2.8, +1.7] *(−0.6)* | +0.5 *(+0.4)* | 86.2 → 92.3 |
| S4_dmad_raw | no chat template | 93.7 → 92.2 | −1.5 [−3.8, +0.8] *(−0.5)* | −0.5 *(+1.2)* | 87.3 → 91.0 |
| **S5_dmad_512** | 512 tokens | 93.4 → 85.3 | **−8.1 [−11.6, −4.7]** *(−5.6 [−8.5, −2.9])* | **−4.7** *(−2.3 [−4.3, −0.4])* | 81.9 → 83.5 |
| S6_dmad_ansonly | answers only | 94.2 → 94.7 | +0.5 [−1.2, +2.2] *(+0.9)* | +0.2 *(+0.5)* | 86.2 → 88.1 |
| S7_dov_chat | DoV + chat template | 93.4 → 94.3 | +0.9 [−1.2, +3.1] *(+0.9)* | +0.9 *(+0.6)* | 89.1 → 93.7 |

**Factor contrasts on P1** (paired, Holm-adjusted across C2–C6):

| Contrast | Estimate | Holm p | Gemma-filled (unadjusted p) |
|---|---|---|---|
| C1: S1 − S2 (whole protocol, replication) | +5.4 [+1.7, +9.4] | p = .005 (unadjusted; C1 is excluded from Holm) | +2.7 [−0.8, +6.3] (p = .14) |
| C2: debate wording | **+5.8 [+2.7, +9.2]** | **.001** | +3.2 [+0.7, +6.0] (p = .021) |
| C3: chat template (DMAD) | **+4.8 [+1.3, +8.5]** | **.019** | +3.3 [+0.5, +6.2] (p = .020) |
| C4: max tokens | −1.7 [−5.3, +1.8] | .46 | −1.8 [−4.7, +1.0] |
| C5: answers only | **+6.8 [+3.8, +10.2]** | **<.001** | +4.7 [+2.2, +7.3] (p < .001) |
| C6: chat template (DoV) | +1.8 [−1.1, +4.8] | .46 | +2.1 [−0.8, +5.1] |

- **Under the Gemma-filled reading,** only C5 survives Holm correction. Holm-adjusted p ≈ .08 for C2 and C3.

## Reading against the pre-registered rules

1. **The harm in the DMAD protocol replicates.** It is a real effect, not only a reading artifact.
   - The pre-registered effect is −6.3 pp.
   - About 40% of it comes from answers drifting into sentences. That share is still a real failure for any downstream reader, but it is a format failure, not a reasoning failure.
   - The remaining −3.8 pp [−6.4, −1.4] comes from agents converging on wrong answers.
2. **Three factors each "drive" the difference** (pre-registered rule: Holm-adjusted CI excludes 0, same sign as C1):
   - the debate wording;
   - the chat template;
   - sharing full reasoning rather than answers only.
   - Changing any **one** of them removes the harm.
   - **The most robust factor is sharing reasoning (C5):** it survives both readings.
3. **Token limit:** the 512-token limit does not rescue DMAD (C4 is not significant). If anything, it makes the harm worse, through more truncated answers.
4. **"Debate helps under setting X":** **not met in any arm.**
   - No P1 or P2 interval lies entirely above 0.
   - The best arms are S7 (+0.9), S6 (+0.5 / +0.9 Gemma-filled) and S3 with gating (+0.5).
   - All of these are statistically indistinguishable from voting.
   - On these 300 problems, the round-0 vote is already 93–95%.
5. **What debate does do in the safe settings:** it raises **individual** accuracy to roughly the round-0 vote level.
   - S3: 86.2 → 92.3.
   - S7: 89.1 → 93.7.
   - S1: 84.1 → 89.3.
   - Debate makes each agent about as good as the majority. It does not make the majority better.

## Mechanism (descriptive)

- **In S2, debate breaks problems that were unanimous and correct at round 0** (6 of 204; 5.5 expected-vote units).
- **Where unanimous problems break (S2 and S5 only), they fail the same way:**
  - In round 1, one agent mutates to a wrong answer.
  - In rounds 2–3, the other agents adopt it.
  - Example, problem 114: 18 ×5 → round 1 two agents say 48 → round 3: 48, 24, 18, 18, 48.
- **In S1, S3, S4, S6 and S7, no unanimous problem is broken.** This includes the arms that show answers only (S6) and that use the DoV wording (S3).

## Positive design implications (for the paper)

- **Debate is safe when** agents see peers' answers only, or the debate wording frames peers as "additional advice", or there is no chat template.
  - In those settings it lifts individual agents to the vote level.
- **Gating (debate only when fewer than 4 of 5 agree)** never hurts by more than the CI noise in the safe settings. In the harmful settings it halves the damage:
  - S2: −6.3 → −3.8.
  - S5: −8.1 → −4.7.
- **A gain over voting needs room above the vote,** for example harder tasks with more contested problems.
  - On contested problems (≤3/5 agree), the safe arms rescue more than they break: S3 +6.0/−4.5, S7 +5.8/−3.0, S6 +4.2/−3.7.
  - Contested problems are only 9–17% of these 300, so the net is small.
