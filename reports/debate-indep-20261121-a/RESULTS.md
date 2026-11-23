# Study I results: the user's independence sentence on MATH Levels 4–5

- **Run:** 2026-11-21. Every job completed.
  - Debate: 300 problems × 2 arms.
  - Peer review: 600 problems × 3 conditions × 2 reviewers. Tasks with an unreadable review are excluded pairwise (at most 2).
- **Compute:** about 3.4 GPU-h, against a 4 GPU-h cap. Above the 3 GPU-h estimate.
  - Generation, from the log timestamps: Gemma reviews 1.26 h, H_full_indep 0.99 h, H_ans_indep 0.57 h, Qwen reviews 0.39 h. Model loading adds about 0.1 h.
  - The Gemma fill took 824 jobs in 329 s.
- **Files:**
  - `debate.out`: the primary, Gemma-filled reading.
  - `debate_ours_prelim.out`: our reader alone.
  - `peer.out`, `mech.out`, `analysis_peer.json`.
- **Comparison:** study H's neutral runs (same problems and seeds).
  - Round 0 of the indep arms should be identical to study H's. Its vote is 68.3 against 68.8 in H_full and H_ans: one or two problems differ, from vLLM batch nondeterminism.
  - Study H's H_dovword run shows the same 68.3.

## Debate (Qwen2.5-7B, 5 agents × 4 rounds, Gemma-filled reading)

| | H_full | H_full + sentence | H_ans | H_ans + sentence |
|---|---|---|---|---|
| Vote, round 0 → 3 | 68.8 → 66.8 | 68.3 → 68.2 | 68.8 → 69.6 | 68.3 → 69.6 |
| P1 final − round-0 vote | −1.9 [−4.8, +0.8] | −0.0 [−3.3, +3.2] | +0.8 [−1.9, +3.4] | +1.3 [−1.8, +4.4] |
| Final − vote20 (73.4) | −6.5 [−10.3, −2.9] | −5.1 [−8.8, −1.5] | −3.7 [−7.1, −0.4] | **−3.8 [−7.1, −0.4]** (IS1) |
| Final-round unanimity | 83.0% | 77.0% | 50.7% | 52.3% |
| Contested problems: rescued / broken | +8.3 / −11.1 | +12.8 / −10.4 | +11.7 / −7.8 | +14.9 / −11.9 |

- **IS1 (primary): not met.** With the sentence, answers-only debate is still 3.8 pp below 20 independent samples. The CI lies entirely below 0.
- **IS2: not met.**
  - Full reasoning: +1.9 [−2.2, +6.2], Holm .76.
  - Answers only: +0.5 [−3.4, +4.4], Holm .78.
- **What the sentence does change:**
  - **Full-reasoning debate copies less.** Final unanimity falls from 83% to 77%.
  - **It rescues more contested problems** (+12.8 vs +8.3).
  - Both move in the expected direction, but not enough to beat even the round-0 vote.
- **Our reader alone** gives the same conclusions (`debate_ours_prelim.out`).

## Peer review (study H's solves, sentence vs neutral)

| Reviewer + peer | Neutral | Sentence | Sentence − neutral | Holm p |
|---|---|---|---|---|
| **Gemma + Qwen peer** (IR) | 82.8 | 84.3 | +1.5 [−0.3, +3.3] | .27 |
| **Qwen + Qwen peer** (IR) | 68.1 | 67.6 | −0.5 [−2.5, +1.5] | .69 |
| **Qwen + Gemma peer** (IR) | 76.6 | 74.1 | −2.5 [−5.0, −0.2] | .14 |
| Gemma self | 84.3 | 84.7 | +0.3 [−0.7, +1.3] | — |
| Gemma + Gemma peer | 86.7 | 86.0 | −0.7 [−2.0, +0.7] | — |
| Qwen self | 65.1 | 65.1 | +0.0 [−1.3, +1.3] | — |

- **No primary test is significant after Holm correction.**
- **The mechanism is a trade-off.** Pooled rates on disagreements, neutral (study H) → sentence:

  | Reviewer | Abandons own right answer for a wrong peer | Accepts a correct peer |
  |---|---|---|
  | Gemma | 18.5% → **12.7%** | 68.2% → 54.5% |
  | Qwen | 37.7% → 33.3% | 58.3% → **49.7%** |

- **Which side of the trade-off wins depends on the peer mix:**
  - A strong reader with a weak peer sees mostly wrong peers (134 own-right & peer-wrong vs 17 the other way), so the lower abandon rate dominates: +1.5.
  - A weak reader with a strong peer sees mostly correct peers (134 vs 17), so the lower acceptance dominates: −2.5.
  - The accounting identity predicts both observed nets within ±5 problems (`mech.out`).

## Reading

1. **The sentence does not rescue debate on hard problems.** Spending the same 20 calls on independent samples still wins by about 4 pp.
2. **In peer review the sentence is a dial, not a fix.**
   - It lowers both the abandon rate and the accept rate.
   - It helps only when the peer is more often wrong than the reader, and hurts when the peer is stronger.
   - So it should be used, if at all, by a reader who knows its peer is weaker. That is the opposite of the setting where peer review helps most.
3. **The GSM8K result** (+7.2 pp with a wrong peer, −1.1 pp with a correct one; `peer-warning-20261119-b`) **was measured on a half-wrong mix of matched twins.**
   - On natural peers, the direction is the same: lower abandon rates, lower accept rates.
   - The net depends on how often the peer is right, which twins hold fixed at 50%.

## Caveats

- **Single models, one task family.**
- **Natural peers:** the peer mix is not randomised. Study T measures the sentence on matched twins instead.
