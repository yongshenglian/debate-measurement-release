# Study M results: debate and peer review on MMLU-Pro (non-math, 10 options)

- **Run:** 2026-11-21. Every job completed with exit 0.
  - Debate: 300 questions × 5 arms.
  - Peer review: 600 questions × 3 conditions × 2 reviewers × 2 prompt versions. 0 invalid review outputs.
- **Compute:** about 5.8 GPU-h, against an 8 GPU-h cap. No arm was dropped.
  - Generation, from the log timestamps:
    - GPU 3 (Gemma solve and reviews): 2.25 h.
    - GPU 1 (H_full, H_full_indep): 1.43 h.
    - GPU 0 (Qwen solve, reviews, vote20): 1.0 h.
    - GPU 2 (H_ans, H_ans_indep): 0.87 h.
  - Gemma fill and audit: 1,001 + 400 jobs, 0.12 GPU-h.
- **Files:**
  - Debate: `debate_analysis.out` (primary, Gemma-filled reading), `debate_ours.out` (our reader alone), `audit/score.out`.
  - Peer review: `peer_analysis.out`.

## Reader audit (pre-registered, ≥95%)

| Arm | Agreement | Verdict |
|---|---|---|
| H_full | 99.0% | PASS |
| H_full_indep | 96.8% | PASS |
| H_ans_indep | 98.9% | PASS |
| H_ans | 93.8% (91/97) | **FAIL** |

- **The H_ans disagreements are the same kind as in study H.** The agent's `<answer>` tag says one option; Gemma took a different option from the prose.
  - Example: the agent computes "closest is 8.26" (option 6), then writes `<answer>8</answer>`.
- Both readings are reported, as pre-registered. **The conclusions are the same under both.**

## Part 1: debate (Qwen2.5-7B)

### Gemma-filled reading

| | H_full | H_ans | H_full + sentence | H_ans + sentence |
|---|---|---|---|---|
| Vote, round 0 → 3 | 55.9 → 54.7 | 55.9 → 58.7 | 55.9 → 54.8 | 55.9 → 58.2 |
| Individual, round 0 → 3 | 51.4 → 55.0 | 51.4 → 54.7 | 51.4 → 54.8 | 51.4 → 54.7 |
| P1 final − round-0 vote | −1.2 [−6.0, +3.5] | **+2.9 [−0.2, +6.0]** | −1.0 [−5.8, +3.6] | +2.4 [−1.1, +5.9] |
| **M1** final − vote20 (59.6) | −4.9 [−10.0, +0.0] | **−0.9 [−4.8, +3.0]** | −4.8 [−9.8, +0.1] | −1.4 [−5.8, +2.8] |
| Pooled 20 − vote20 | −2.3 | −0.9 | −3.9 | −1.4 |
| Final unanimity | 71.7% | 35.3% | 75.7% | 35.0% |

- **The compute-matched vote gains little here.** 20 samples vote 59.6% against 57.6% for 5: +2.0 [−1.1, +5.1]. On MATH the gain was +6.5.
- **M1: no arm differs significantly from vote20.**
  - Full-reasoning arms are about 5 pp below it, with CIs touching 0.
  - **Answers-only debate is level with vote20** (−0.9 [−4.8, +3.0]).
- **M2 contrasts (Holm over three):**

  | Contrast | Estimate | Holm p |
  |---|---|---|
  | H_ans − H_full | +4.0 [−0.4, +8.5] | .23 |
  | H_full_indep − H_full | +0.2 | 1.0 |
  | H_ans_indep − H_ans | −0.5 | 1.0 |

### Our reader alone

- P1 for H_ans: **+3.2 [+0.0, +6.4]**. The CI touches 0.
- Final − vote20: −4.4, −0.5, −4.3, −0.9.
- The H_ans − H_full contrast: +4.0, Holm .24.

## Part 2: peer review

### Neutral prompts

| Reviewer | First answer | Self | + Qwen peer | + Gemma peer |
|---|---|---|---|---|
| Gemma-27B | 66.3 | 66.2 | 64.0 | 65.5 |
| Qwen-7B | 54.5 | 55.8 | 56.0 | **65.5** |

- **M3, Qwen reads Gemma − self:** **+9.7 [+6.3, +13.0]**. **Replicates** (GSM8K +1.6, MATH +11.5).
- **M4, Gemma reads Qwen − self:** −2.2 [−5.0, +0.5]. The same direction as MATH (−1.5), not significant. **Not shown to harm.**
- **Same-model peers:**
  - Qwen reads Qwen: +0.2 [−2.5, +2.8].
  - Gemma reads Gemma: −0.7 [−3.5, +2.2].
  - On MATH both had helped (+2.8 and +2.3).

### Pooled rates on disagreements (Wilson 95% CIs) across the three tasks

| Reviewer, task | Accepts a correct peer | Abandons own right answer for a wrong peer |
|---|---|---|
| Gemma, GSM8K | 88.2% | 28.8% |
| Gemma, MATH | 68.2% | **18.5%** |
| Gemma, MMLU-Pro | 56.0% [45.3, 66.1] | **42.3% [35.0, 50.0]** |
| Qwen, GSM8K | 44.1% | 32.3% |
| Qwen, MATH | 58.3% | 37.7% |
| Qwen, MMLU-Pro | 73.6% [66.4, 79.8] | **54.6% [44.7, 64.2]** |

- **Both models abandon right answers most often on MMLU-Pro.**
  - Gemma's abandon rate is more than twice its MATH rate.
  - Qwen gives up a right answer more often than not.
- **The accounting identity still predicts the net** (predicted vs observed, in problems): −17 vs −14, −5 vs −5, +4 vs +9, +63 vs +66.

### The independence sentence (study I prompts, M5, Holm over three)

| Reviewer + peer | Sentence − neutral | Holm p |
|---|---|---|
| Gemma + Qwen peer | **+2.0 [+0.0, +4.2]** | .21 |
| Qwen + Qwen peer | −0.7 [−2.5, +1.2] | .79 |
| Qwen + Gemma peer | −1.2 [−3.7, +1.2] | .79 |

- **The same trade-off as on MATH (study I):**

  | Reviewer | Abandons own right answer | Accepts a correct peer |
  |---|---|---|
  | Gemma | 42.3% → **32.5%** | 56.0% → 52.4% |
  | Qwen | 54.6% → 51.5% | 73.6% → **63.8%** |

- **What that does to the nets:**
  - The strong reader with a weak peer moves from −2.2 to +0.7.
  - The weak reader with a strong peer loses a little.
- **No primary test is significant.**

## Reading

1. **Debate:** answers-only debate is the only arm that comes close to compute-matched voting, and it reaches parity.
   - On MATH it lost by 3.7 pp.
   - **On MMLU-Pro it ties**, because extra independent samples buy little on a 10-option knowledge task (+2.0 vs +6.5 on MATH). P1 for answers-only is the most positive of any task (+2.9, CI touching 0).
   - Full-reasoning debate is never better than answers-only on any task (GSM8K, MATH, MMLU-Pro).
2. **Weak reads strong** is the most robust positive result in the project: +1.6 on GSM8K, +11.5 on MATH, +9.7 on MMLU-Pro.
3. **Strong reads weak** never helps:
   - −0.2 on GSM8K, −1.5 on MATH, −2.2 on MMLU-Pro;
   - always harmful in direction, never significant alone.
4. **The rates are task-specific; the accounting is not.**
   - Resistance to wrong peers is highest where a reader can check the work (Gemma on MATH, 18.5%).
   - It is lowest on knowledge questions, where the peer's argument cannot be checked (42–55%).
5. **The independence sentence is a dial.**
   - It cuts abandonment by about 10 pp for Gemma, on both MATH and MMLU-Pro.
   - It costs acceptance of correct peers by about 9–10 pp for Qwen, on both tasks.

## Caveats

- **Single models.**
- **MMLU-Pro categories are mixed;** no per-category test was planned.
- **The H_ans audit fails by the pre-registered rule.** It is reported under both readings.
- **The pooled rates across tasks come from different studies.** They used the same harness, but different token limits (1,024 on GSM8K, 2,048 here).
