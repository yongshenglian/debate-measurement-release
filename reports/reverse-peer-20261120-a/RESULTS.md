# Study R results: does a strong model get worse after reading a weaker model's work?

- **Run:** 2026-11-20. Full GSM8K test set (1,319 problems).
  - Every call completed.
  - **0 invalid outputs**, in any stage or condition, for either model.
- **Compute:** about 1.5 GPU-h, against a 3 GPU-h cap.
  - Generation took 4,169 s; model loading and one GPU's idle wait between stages make up the rest.
- **Outputs:**
  - `analysis.out` and `analysis.json`: pre-registered.
  - `mech_posthoc.out`: post-hoc pooled rates with Wilson CIs, from `collaboration/when_debate/reverse_mech.py`.

## Accuracy

| Reviewer | First answer | Self-review | + Qwen peer | + Gemma peer |
|---|---|---|---|---|
| Gemma-3-27B | 95.3 | 95.5 | 95.2 | 95.8 |
| Qwen2.5-7B | 92.0 | 92.2 | 92.3 | 93.8 |

### Primary R1 (Gemma reads Qwen, minus Gemma self-review)

- **Result:** −0.2 pp [−1.1, +0.6].
- **Pre-registered reading:** the CI includes 0, but its lower bound (−1.1) is just outside ±1 pp. Formally this is **inconclusive**, with any harm at most about 1 pp.
- **Plainly:** reading the weaker model's work leaves the strong model essentially unchanged.

### Mirror R2 (Qwen reads Gemma, minus Qwen self-review)

- **Result:** **+1.6 pp [+0.7, +2.5].**
- The weaker model benefits from the stronger one. This replicates the earlier peer studies with natural, unedited peers.

### Same-model controls

| Reviewer | Peer − self |
|---|---|
| Gemma reads Gemma | +0.3 [−0.4, +1.0] |
| Qwen reads Qwen | +0.1 [−0.7, +0.8] |

- Self-review alone does nothing: self − first answer is +0.2 for both models.

## Why: the strong model loses as many answers to a wrong peer as the weak one does

The table below covers only the problems where the two answers differ. Rates are pooled over both peer sources, with Wilson 95% CIs (post hoc).

| Reviewer | Accepts a correct peer (own wrong, peer right) | Abandons own right answer (own right, peer wrong) | Adopts the wrong peer's exact answer |
|---|---|---|---|
| Gemma-27B | **88.2% [73.4, 95.3]** (30/34) | **28.8% [19.7, 40.0]** (21/73) | 26.0% [17.3, 37.1] (19/73) |
| Qwen-7B | **44.1% [34.9, 53.8]** (45/102) | **32.3% [22.0, 44.6]** (20/62) | 30.6% [20.6, 43.0] (19/62) |

- **Being stronger doubles how often the model recognises a correct peer** (88% vs 44%).
- **Being stronger does not reduce how often it is talked out of a correct answer** (29% vs 32%; the CIs overlap almost completely).
  - About a quarter to a third of the time, either model drops a right answer for the peer's wrong one.
  - It almost always takes the peer's exact number.
  - When the peer is also right, corruption is about 0% (0.0–0.4%).
- **The net effect is therefore simple accounting.** It reproduces the observed net exactly for both cross-model pairs:

  net ≈ (# own wrong & peer right) × accept rate − (# own right & peer wrong) × abandon rate

  | Pair | Accepted | Abandoned | Net |
  |---|---|---|---|
  | Gemma reads Qwen | 21 × 86% = 18 | 64 × 30% = 19 | **−1** |
  | Qwen reads Gemma | 64 × 45% = 29 | 21 × 29% = 6 | **+23** (= +1.7 pp) |

- **Answer to the question:**
  - The strong model is **not** harmed on net, but only because Qwen is right 92% of the time.
  - Each time Qwen is wrong where Gemma is right, Gemma follows it about 30% of the time.
  - A weaker peer with lower accuracy, or a task where the weak model fails more often, would produce net harm. The rate at which the strong model abandons right answers does not shrink with strength.

## Caveats

- **Peers are natural, not randomised.** Peer correctness is confounded with difficulty.
  - The problems Gemma gets wrong are harder ones, yet it still accepts correct peers 88% of the time there. That makes the accept-rate gap conservative.
- **Small subgroups:** the "own right, peer wrong" cells are small (Gemma 73, Qwen 62), so the abandon-rate comparison can rule out only large differences. Roughly, it cannot exclude a gap of up to about 15–20 pp.
- **Prompt length:** prompts are not token-matched across conditions. GSM8K is near ceiling for both models.
- **Post-hoc measures:** the pooled rates and the accounting identity were computed post hoc. The pre-registered per-condition rows are in `analysis.out`.
