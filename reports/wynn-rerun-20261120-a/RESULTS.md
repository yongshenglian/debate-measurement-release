# Study C: Wynn et al. open-model configurations: results

- **Protocol:** `PROTOCOL.md`, including its run log and the chat-template deviation.
- **Scope:** GSM8K, 5 seeds × 100 questions per configuration, 3 agents, rounds 0–2.
- **Code:** the authors' code, with only the two missing `vec_*` LLM configs added (pointing to local vLLM).
- **Analysis:**
  - pre-registered: `collaboration/wynn_rerun/analyze_wynn.py` → `analysis.out`, `analysis.json`;
  - post-hoc sensitivity: `sensitivity_tierule.py` → `sensitivity_tierule.out`.
- **CIs:** bootstrap over questions within seed, 20,000 draws, 95%.
- **GPU time:** about 3.5 GPU-h (two servers).

## Majority vote, round 0 → round 2 (pp)

| Config | Paper | Authors' scorer | Validated extraction, ties split evenly (pre-registered) | Validated extraction, authors' tie rule (post hoc) |
|---|---|---|---|---|
| 3×Llama | 87.6 → 84.2 (−3.4) | 86.6 → 85.2, **−1.4** [−3.4, +0.6] | 88.4 → 87.1, −1.3 [−3.1, +0.4] | 88.0 → 87.0, −1.0 [−3.0, +1.0] |
| 2L+1M | 82.6 → 75.8 (−6.8) | 86.2 → 77.8, **−8.4** [−11.8, −5.2] | 86.7 → 79.1, −7.7 [−10.6, −4.9] | 88.6 → 79.2, −9.4 [−12.6, −6.4] |
| 1L+2M | 61.0 → 64.8 (+3.8) | 81.0 → 68.0, **−13.0** [−17.2, −8.8] | 67.8 → 68.5, **+0.6** [−2.7, +4.0] | 82.4 → 69.4, −13.0 [−17.0, −9.0] |
| 3×Mistral | 43.6 → 46.4 (+2.8) | 43.0 → 43.8, +0.8 [−2.6, +4.2] | 52.5 → 52.0, −0.5 [−3.3, +2.3] | 51.2 → 52.0, +0.8 [−2.6, +4.2] |

## Findings

1. **Answer extraction is not the problem here.**
   - With the same vote rule, the authors' parser and ours give the same verdict in every configuration (compare the authors' column with the last column).
   - The parser drops `\boxed{240 sheets/month}` (it reads only digits, `.`, `,` and `$` inside braces). It also fails to read 5–29% of outputs per round, mostly Mistral's.
   - The string comparison wrongly fails 1–2% of numerically correct answers ("18.0" vs "18").
   - These misses are roughly balanced between right and wrong answers, so no verdict changes.
2. **Tie-breaking is a hidden choice that can flip a heterogeneous result.**
   - The rule: `most_frequent` returns the *first* answer to reach the maximum count. Agents are ordered with llm1 (Llama) first, so a three-way tie silently goes to Llama, the strongest model.
   - Three-way ties at round 0: 25% of questions in 1L+2M, 22% in 3×Mistral, 7% in 2L+1M.
   - Debate makes agents converge, so ties fall (to 7% by round 2 in 1L+2M), and with them Llama's hidden casting vote.
   - **Effect in 1L+2M:** −13.0 pp under the authors' rule, against +0.6 pp [−2.7, +4.0] with ties split evenly.
   - The "debate harms" verdict for 1L+2M therefore depends on an unreported tie-breaking choice. The paper reports +3.8 there; our authors'-scorer round 0 (81.0) is far above the paper's 61.0, which we cannot explain without their raw outputs.
3. **What replicates robustly:**
   - **2L+1M:** debate harms, −7.7 to −9.4 pp under every scoring (paper −6.8).
   - **The mechanism, in both scorers:** Llama agents are corrupted by the Mistral agent. Right→wrong flips are 16% of right agents in round 1, against 7% in 3×Llama.
4. **What does not replicate:**
   - 3×Llama's decline (paper −3.4): −1.0 to −1.4, not significant.
   - 3×Mistral's gain (paper +2.8): −0.5 to +0.8, not significant.
5. **Individual accuracy rises with debate in every configuration and under both scorers** (e.g. 3×Llama 82.9 → 84.8 validated). Vote declines come from correlation and herding, not from worse individual reasoning.

## Deviations

- **Serving:** local vLLM instead of the Together API.
- **Temperature:** the server default (Llama 0.6 from its generation_config; Mistral 1.0). The paper states only "default temperature".
- **Llama chat template:** the mirror's older Llama-3 template (no 3.1 date header).
