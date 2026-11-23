# Study T results: GSM8K twin findings on MATH Levels 4–5

- **Run:** 2026-11-21. Twin construction as in Amendment 1: an injected slip, continued by Qwen.
- **Twins:**
  - 521 problems had a base with an injection point.
  - 313 problems got valid C and E twins.
  - Failures: no valid E on 183 problems, no valid C on 25.
- **Recipients:**
  - Qwen: 312 problems × 13 conditions.
  - Gemma: 296 problems × 7 conditions. Problems whose base is Gemma's own sample are excluded.
- **Quality checks:** 0 invalid answers. Padding matched prompt lengths exactly (maximum deviation 0 tokens).
- **Compute:** about 2.2 GPU-h, against a 6 GPU-h cap.
  - Continuations: 3 × about 8 min.
  - Reviews: 2,018 s for Qwen, plus 1,606 s and 1,638 s for Gemma; plus model loading.
- **Files:** `analysis.out`, `analysis.json`, `build/build_report.json`.

## Qwen2.5-7B recipient (312 problems; own first answer right on 212)

| Test | MATH | GSM8K | Holm p | Verdict |
|---|---|---|---|---|
| T1 wrong working hurts: E_full − private | **−5.1 [−9.3, −1.0]** | −14.1 | .039 | **replicates**, about a third the size |
| T2a wrong answer alone: E_answer − private | −1.9 [−4.5, +0.3]; 90% CI [−3.8, 0.0] | about 0 | — | not shown equivalent within ±2 |
| T2b working vs answer: E_full − E_answer | −3.2 [−7.4, +1.0] | large | .31 | not replicated |
| T3 confident vs hedged: E_confident − E_hedged | **−6.1 [−10.6, −1.6]** | −12.0 | .037 | **replicates**, about half the size |
| T4 the user's sentence, net over C and E | +1.8 [−1.0, +4.5] | +3.0 | .31 | not replicated (same direction) |
| T5 correct working vs answer: C_full − C_answer | **+9.6 [+5.4, +13.8]** | about +9.8 (strong peers, full − answers) | <.001 | **replicates** |

### Accuracy by condition

| Condition | Accuracy |
|---|---|
| private | 68.6 |
| C_full | **86.2** (+17.6 vs private) |
| C_answer | 76.6 |
| E_full | 63.5 |
| E_answer | 66.7 |
| E_confident | 59.3 |
| E_hedged | 65.4 |
| E_full + indep | 67.3 |
| C_full + indep | 85.9 |

### Readers whose first answer was right

| Peer | Abandons the right answer | Adopts E's wrong answer |
|---|---|---|
| E_answer | 3.3% | 2.8% |
| E_full | 12.7% | 12.7% |
| E_confident | 17.0% | 15.6% |
| E_hedged | 10.4% | — |
| E_full + indep | **8.5%** | **7.1%** |

- **Your sentence nearly halves adoption of the wrong answer** (12.7% → 7.1%), and on correct peers it costs almost nothing (−0.3 pp).
  - On E alone it gives +3.8 [0.0, +7.7].
  - The net estimate is positive but its CI includes 0.
- **The check and warn sentences:** no net effect (−0.2 and 0.0).

## Gemma-3-27B recipient (296 problems; own first answer right on 270)

| Test | MATH | Holm p |
|---|---|---|
| T1 E_full − private | −0.3 [−2.7, +1.7] | .87 |
| T2b E_full − E_answer | +2.4 [−0.3, +5.1] | .38 |
| T5 C_full − C_answer | +2.4 [+0.3, +4.7] | .14 |
| T4 indep net | −1.2 [−2.9, +0.5] | .40 |

- **The strong reader is close to immune to a wrong peer:** it abandons a right answer 1.9% of the time with E_full.
- **It still gains from correct working:** C_full − private +5.4 [+3.0, +8.1].
- **Your sentence slightly lowers its use of correct peers** (C: −1.7).

## Reading

1. **What replicates on hard problems:**
   - wrong working hurts a weak reader (T1);
   - confident wording makes it worse than hedged (T3);
   - correct working helps far more than the bare correct answer (T5).
2. **What shrinks or fails:**
   - The harm of wrong working is about a third of the GSM8K size.
   - The "working, not the answer, carries the harm" contrast (T2b) is no longer significant.
3. **New on hard problems: the information value of a correct peer's working dominates its persuasion risk.**
   - For Qwen, correct working gives +17.6 pp, wrong working −5.1 pp.
   - So a half-right peer pool is net positive: (+17.6 − 5.1) / 2 ≈ +6 pp.
4. **The user's sentence keeps the same direction as on GSM8K but is smaller and not significant.**
   - Readers adopt the wrong answer about half as often (7.1% vs 12.7%), at almost no cost with correct peers for Qwen.
   - For Gemma, the strong reader, it does nothing useful.

## Caveats

- **Twin construction differs from GSM8K** (Amendment 1):
  - The twins share text only up to an injected number; the continuations are independent Qwen samples.
  - Only error slips that survived to the final answer are kept, which selects slips the continuer did not notice. That could bias T1 either way.
- **Base sampling:** problems were drawn by solvability (a correct base exists), so the set is easier than the full 600 (Qwen private 68.6%).
- **One error type:** a single numeric slip.
