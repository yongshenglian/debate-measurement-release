# Study D: Du et al. GSM8K debate protocol with open models: results

- **Protocol:** `PROTOCOL.md`, frozen before any output.
- **Analysis:** `collaboration/du_rerun/analyze_du.py`; outputs in `out/du_*_analysis.json`.
- **Completion:** both runs completed, exit 0.
- **GPU time:** about 0.5 h.

## Vote accuracy, round 0 → round 2 (300 questions, 3 agents)

| Model | Scorer | Round 0 | Round 1 | Round 2 | Change, round 2 − round 0 [95% CI] |
|---|---|---|---|---|---|
| Qwen2.5-7B | Du et al. | 88.0 | 89.0 | 88.7 | +0.7 [−1.3, +2.7] |
| Qwen2.5-7B | validated (`\boxed`) | 87.9 | 89.1 | 89.0 | +1.1 [−0.8, +3.1] |
| Llama-3.1-8B | Du et al. | 83.0 | 84.0 | 82.7 | −0.3 [−4.0, +3.7] |
| Llama-3.1-8B | validated (`\boxed`) | 84.6 | 84.0 | 83.6 | −1.0 [−4.1, +2.0] |

- **On the authors' 100-question subset:**
  - Qwen: +0.0 (Du) and +0.3 (validated).
  - Llama: +1.0 and +2.3. All non-significant.
- **Verdict:** under both scorers, debate neither helps nor hurts the vote with these models. The scorer does not change the verdict.

## Parser diagnostics

| Model | Round | Authors' last-number fallback used | Misread (both parsers find an answer, they differ) | Vote with no readable answer, scored correct |
|---|---|---|---|---|
| Qwen | 0 / 1 / 2 | 0.0 / 0.0 / 0.0% | 0% | 0 |
| Llama | 0 / 1 / 2 | 15.2 / 6.1 / 3.1% | 0% | 0 |

- **Qwen with a chat template** writes clean `\boxed{N}`, so nothing triggers the parser's weak points.
- **Llama skips `\boxed` in round 0 (15%)** and converges on it after seeing peers. This is format contagion in the *repairing* direction.
- **Individual accuracy depends on the scorer:**
  - Llama individual accuracy rises +4.4 pp (77.8 → 82.2) under Du's lenient fallback.
  - It rises +13.0 pp (68.2 → 81.2) under the strict `\boxed` scorer.
  - The strict scorer counts round-0 answers given without `\boxed` as missing. So "debate improves individual agents" is partly format convergence.

## Reading

- **No artifact in Du et al.'s protocol** with these models. The chat-template prompting and `\boxed` format do not produce the escaped-brace or loop failures seen in Debate-or-Vote's raw prompting.
- **The "failure = correct" bug never fired,** because the last-number fallback almost always finds some number.
- **Format convergence across rounds is real here too.** Any scorer that is strict about format will make debate look like it improves individual agents.

## Deviation found after the run

- **What:** the Llama run used the weights mirror's older Llama-3 chat template, not the official 3.1 template.
- **How it differs:** the old template has no "Cutting Knowledge Date / Today Date" system block. The assistant header is present in both.
- **Expected effect on the Du protocol:** small. It is recorded as a deviation.
