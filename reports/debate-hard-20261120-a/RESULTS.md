# Study H results: debate and peers on hard problems (MATH Levels 4–5, integer answers)

- **Run:** 2026-11-20. Every job completed.
  - Settings part: 300 problems × 4 arms.
  - Peer part: 600 problems. Reviews could not be analysed for 1 task (Gemma) and 2 tasks (Qwen); those tasks are excluded.
- **Compute:** about 6.0 GPU-h of GPU occupancy, at the 6 GPU-h cap.
  - Generation took about 5.7 GPU-h, including about 0.5 GPU-h of one GPU idling between stages.
  - The Gemma fill and audit took about 0.3 GPU-h.
  - No arm was dropped.
- **Files:**
  - `analysis_filled.out`: the primary, Gemma-filled reading.
  - `analysis_ours.out`
  - `audit/score.out`
  - `peer/analysis.out`, `peer/mech.out`, `peer/hr3.out`
  - `pool_posthoc.out`: post hoc.

## Reader audit (pre-registered)

| Arm | Agreement | Verdict |
|---|---|---|
| H_full | 97.8% | PASS |
| H_dovword | 97.9% | PASS |
| H_ans | 93.3% (84/90) | **FAIL** by the rule |

- **On inspection, all 6 H_ans disagreements are Gemma misreads.** In each, the agent's `<answer>` tag holds the number our reader took, and Gemma picked an intermediate number from the working. One example: the answer tag says `128`; Gemma returned `1961256`, a numerator from the working.
- As pre-registered, the arm is reported under both readings. **The conclusions are the same under both.**

## Part 1: can debate beat voting with room above the vote? (Qwen2.5-7B, Gemma-filled reading)

| | H_full (full reasoning) | H_ans (answers only) | H_dovword (DoV wording) |
|---|---|---|---|
| Vote, round 0 → 3 | 68.8 → 66.8 | 68.8 → 69.6 | 68.3 → 66.4 |
| Individual, round 0 → 3 | 59.3 → 65.3 | 59.0 → 63.7 | 59.1 → 65.6 |
| **P1** final − round-0 vote | −1.9 [−4.8, +0.8] | **+0.8 [−1.9, +3.4]** (HS1) | −1.9 [−5.2, +1.4] |
| P2 G4 gate − round-0 vote | −0.9 [−3.6, +1.6] | +1.3 [−1.2, +3.8] | −1.9 [−5.1, +1.2] |
| **Final − 20-sample vote** | −6.5 [−10.3, −2.9] | **−3.7 [−7.1, −0.4]** (HS2) | −7.0 [−10.7, −3.3] |
| Contested (≤3/5) problems: rescued / broken | +8.3 / −11.1 | +11.7 / −7.8 | +11.1 / −16.7 |

- **Compute-matched vote (H_vote20):**
  - 20 independent samples vote 73.4%. The first 5 of them vote 66.8%.
  - 20 − 5: **+6.5 [+3.2, +9.9]**.
- **HS3 contrasts** (Holm-adjusted):

  | Contrast | Estimate | Holm p |
  |---|---|---|
  | H_ans − H_full | +2.7 [−0.9, +6.5] | .30 |
  | H_dovword − H_full | +0.1 | .99 |

- **Our reader only:**
  - P1: H_full −2.1, H_ans +0.1, H_dovword −2.4.
  - Final − vote20: −7.9, −4.6, −7.7. All three CIs lie below 0.

### Reading against the pre-registered rules

1. **"Debate beats voting on hard problems": not met.** Even in the safest arm, debate is indistinguishable from the round-0 5-agent vote, although round-0 votes are now 68% and 45–50% of problems are contested.
2. **Compute-matched voting beats every debate arm.**
   - Spending the same 20 calls on independent samples gives 3.7–7.0 pp more than debating.
   - Post hoc: a vote over all 20 answers that a debate produced is still 2.5–6.2 pp below 20 independent answers.
   - Debate makes samples correlated, which wastes their information.
3. **The GSM8K safety ranking does not replicate on hard problems.**
   - Answers-only is the best arm and the only one that rescues more contested problems than it breaks, but its advantage over full reasoning is not significant.
   - With full reasoning shown, debate is no longer clearly harmful (−1.9, n.s.).
4. **Debate still makes each agent better** (individual accuracy +4.7 to +6.5 pp) **without making the group better.** This replicates study S.

## Part 2: does a strong model get worse reading a weaker one on hard problems?

| Reviewer | First answer | Self | + Qwen peer | + Gemma peer |
|---|---|---|---|---|
| Gemma-27B | 84.1 | 84.3 | 82.8 | 86.6 |
| Qwen-7B | 64.7 | 65.2 | 68.1 | **76.8** |

- **HR1, Gemma reads Qwen − self:** −1.5 [−3.3, +0.3].
  - Not entirely below 0, so **not shown to harm**, though the direction is harmful.
  - Net vs first answer: −8 problems of 599.
- **HR2, Qwen reads Gemma − self:** **+11.5 [+8.5, +14.5].** That is seven times the GSM8K gain (+1.6).
- **Same-model controls:**
  - Qwen reads Qwen: +2.8 [+0.3, +5.4].
  - Gemma reads Gemma: +2.3 [+0.5, +4.3].
  - On hard problems, even an equal peer helps a single reviewer. Self-review alone does nothing: +0.2 and +0.5.
- **HR3, prediction test:**
  - Predicted −3.93 pp from the GSM8K rates (134 problems where Gemma is right and Qwen wrong; 17 the other way).
  - The value lies outside the HR1 CI [−3.3, +0.3], so **not supported**.
  - Reason: the rates themselves change with the task (below). The accounting identity still holds with task-specific rates. Predicted vs observed net: −12 vs −8, +13 vs +15, +20 vs +20, +70 vs +72.

### Pooled rates where the answers differ (post hoc, Wilson 95% CIs)

| Reviewer | Accepts a correct peer | Abandons own right answer for a wrong peer |
|---|---|---|
| Gemma, hard | 68.2% [53.4, 80.0] (30/44) | **18.5% [13.2, 25.3]** (29/157) |
| Qwen, hard | 58.3% [51.3, 64.9] (116/199) | **37.7% [27.2, 49.5]** (26/69) |
| Gemma, GSM8K (study R) | 88.2% | 28.8% |
| Qwen, GSM8K (study R) | 44.1% | 32.3% |

- **The GSM8K pattern does not generalise.**
  - On hard problems the stronger model resists wrong peers about twice as well (18.5% vs 37.7%, CIs disjoint).
  - Its advantage at accepting correct peers shrinks (68% vs 58%, overlapping).
  - On GSM8K it was the reverse.
- **What generalises is the accounting:**
  - the net effect = rescues on disagreements minus abandonments on disagreements;
  - the rates are task- and model-specific and have to be measured, not assumed.

## Caveats

- **Single models, single task family:** one debating model (Qwen-7B) and one task family (integer-answer MATH).
- **Peers are natural, not randomised.**
- **Compute matching applies to Part 1 only.** Part 2's single-review comparisons are not compute-matched against sampling: a review uses 2–3 calls.
- **Token limit:** 2,048 tokens still truncates a few Gemma solutions (about 9% in the smoke run).

## Deviation found 2026-11-23 (release hash check)

- `collaboration/when_debate/reverse_mech.py`, which computes HR4, no longer matches `frozen.sha256`.
- It was edited at analysis time (2026-11-20 23:26:13), one second before `peer/mech.out` was written. The protocol's "unchanged" is therefore wrong.
- The earlier version was not kept.
- HR4 in `peer/mech.out` is not covered by the freeze. HR1–HR3 (`reverse_analyze.py`) match their hashes.
- The paper's accept and abandon rates are recomputed independently in `reports/revision-reanalysis-20261122`.
