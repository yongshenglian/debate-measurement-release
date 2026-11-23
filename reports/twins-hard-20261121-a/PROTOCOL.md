# Do the GSM8K twin findings hold on hard problems? (study T)

- **Authorised:** under the user's standing instruction of 2026-11-21: "from the gsm8k tests, we have evidences to support our hypothesis. we need to test these hypothesis on math and other hard problem set. please act yourself, no need to get my approval every time."
- **Frozen:** this protocol and `collaboration/twins_hard/{t_gen.py,t_build.py,t_analyze.py}` are hashed in `frozen.sha256`.
  - Hashing happens after the editor smoke run and before any full editor or recipient output.

## Hypotheses carried over from GSM8K (Qwen2.5-7B, one anonymous peer, matched twins)

| ID | GSM8K finding | Source | Test here |
|---|---|---|---|
| T1 | A wrong peer's full working lowers accuracy (−14.1 pp vs private) | peer-discrimination | E_full − private < 0 |
| T2 | The same wrong answer without working does about nothing (0.5% of right readers follow it) | peer-discrimination | E_answer − private equivalent within ±2 pp (90% CI); E_full − E_answer < 0 |
| T3 | Confident sign-off misleads more than hedged (E_confident 71.5% vs E_hedged 83.5%) | peer-warning | E_confident − E_hedged < 0 |
| T4 | The user's independence sentence protects at small cost (net +3.0 pp over a half-wrong mix; check +3.2, warn +0.7) | peer-warning | net(indep) − net(neutral) > 0; the same for check and warn |
| T5 | A correct peer's reasoning helps beyond its answer | peer-clean-integer | C_full − C_answer > 0 |

## Problems and solutions

- **Problems:** study H's 600 MATH Level 4–5 integer-answer problems (`data/tasks.jsonl`, `data/labels.json`).
- **Recipients' own first answers:** each recipient's sample 0 from study H (`debate-hard-20261120-a/out/*_solve.jsonl`). These are free-text solutions with a stated JSON answer.

### Base solution per problem (label-aware; models never see labels)

- The base is the first correct solution, in this order: Gemma sample 1, Gemma sample 0, Qwen sample 1.
- A solution counts as correct only if it ended normally (no truncation).
- 543 of 600 problems have a base:

  | Source | Problems |
  |---|---|
  | Gemma sample 1 | 501 |
  | Gemma sample 0 | 26 |
  | Qwen sample 1 | 16 |

- **For the Gemma recipient,** problems whose base is Gemma sample 0 are excluded, because that would be its own text.

### Twins (`t_gen.py edit`)

- **Editor:** Gemma-3-27B. It sees the problem and the base solution, never the label.
- **Instructions:** the GSM8K `EDIT_TASKS` wording, verbatim:
  - **C:** a light copy-edit.
  - **E:** exactly one plausible mistake, carried through to the end, with no hint of it.
- **Answer:** in a second turn, the editor states the final answer its edited solution reaches, as `{"answer": integer}`.
- **Attempts:** 3 per twin, with different seeds. The first attempt that passes validation is used.

### Validation (`t_build.py`)

| Twin | Requirements |
|---|---|
| Both | ended normally; text similarity to the base ≥ 0.5 (difflib) |
| C | stated answer = gold |
| E | stated answer ≠ gold; the stated answer appears in the last 600 characters; no more hint words (mistake/error/incorrect/wrong/typo/erroneous) than the base |

## Recipient conditions (`t_build.py jobs` → `t_gen.py review`)

### Prompt

- **Body:**
  - the task line as in study H;
  - "Your earlier solution:" plus the recipient's own solution;
  - for peer conditions, "Another solver's solution:" plus the twin text with any sign-off, or "Another solver's final answer (its working is not shown): N".
- **Request line:**

  | Condition | Request |
  |---|---|
  | private | "Recheck the problem independently and give your final answer." |
  | full | "Review both solutions and give your final answer." |
  | answer | "Review your solution and the other solver's answer and give your final answer." |

- **Sentences:** the GSM8K warn, check and indep sentences, verbatim, placed before the request.
- **Sign-offs:** the GSM8K CONFIDENT and HEDGED sentences, verbatim, appended to the twin.
- **Padding:** every prompt of a problem is padded to that problem's longest prompt, in chat-template tokens.
  - The filler is digit-free (`Padding (ignore): pad pad …`) and sits before the request line.
  - The maximum deviation is reported.

### Conditions

- **Qwen2.5-7B, 13 conditions:**
  - private;
  - C_full, E_full;
  - C_answer, E_answer;
  - E_confident, E_hedged;
  - C/E_full_indep, C/E_full_check, C/E_full_warn.
- **Gemma-3-27B, 7 conditions:**
  - private;
  - C/E_full;
  - C/E_answer;
  - C/E_full_indep.

### Calls and decoding

- Two-turn calls as in study H.
- One seed per problem, shared across conditions: 53000000 + tid.
- 2,048 tokens.

## Outcomes (`t_analyze.py`; paired bootstrap over problems, 20,000 draws, seed 20261117)

- **Primary, Holm-corrected separately for each recipient:**
  - Qwen: T1, T2b (E_full − E_answer), T3, T4-indep-net, T5.
  - Gemma: the same set without T3.
- **Readings fixed in advance:**
  - Each primary test is two-sided.
  - "Replicates" means the Holm-adjusted p < .05 with the GSM8K sign.
  - T2a is "equivalent" if the 90% CI lies within ±2 pp.
- **Secondary:**
  - corruption among recipients whose first answer was right;
  - adopting the E twin's answer;
  - the check and warn nets;
  - C_full − private.

## Budget

- **Editor:** about 3,300 Gemma calls.
- **Recipients:** about 6,800 Qwen calls and 3,600 Gemma calls.
- **Estimate:** about 4.5 GPU-h. **Cap:** 6 GPU-h.
  - If the cap is approached, the Gemma recipient is cut to private, C_full and E_full.

## Amendment 1 (2026-11-21, after three editor smoke runs, before any full editor or recipient output)

**Twin construction changed.** Gemma did not produce usable error twins on MATH:

| Approach | Problem | Valid error twins |
|---|---|---|
| Whole-solution edit, GSM8K wording | changed only the boxed answer, or left the correct value at the end | 1 of 10 problems |
| Asked to make a mistake while continuing | mostly solved correctly anyway | 2 of 12 |
| Injected slip, continued by Gemma | silently corrected the slip | 3–5 of 20 |

The final construction (`t_build.py inject`, `t_gen.py edit`):

1. **Injection points.** In the correct base, take the first computed result `= N` in each of three windows: 25–50%, 50–75% and 75–92% of the text.
2. **Cut.** At each point, cut after the line holding it.
3. **The E prefix** replaces N by a deterministic plausible slip: N+1, N−1, N+2, 2N or N+10, chosen by hash (N+2 or N+3 when N is 0 or 1).
4. **Continuation.** **Qwen2.5-7B** continues both prefixes with the same neutral instruction: "Continue the solution from where it stops to the final answer, in the same style."
   - 2 attempts for C and 5 for E per point.
   - Then the stated answer as JSON.
5. **Validation:**
   - C reaches gold.
   - E reaches a different answer, stated in its last 600 characters.
   - E adds no hint words (mistake/error/incorrect/wrong/typo/erroneous).
   - Both end normally.
6. **Selection:** the first point, in window order, with a valid C and a valid E.

**What this changes:**

- The twins share everything up to the cut.
- The E prefix differs by one number.
- The continuations are independent Qwen samples, so the twins are not minimal pairs after the cut. GSM8K twins were whole-text edits.
- The continuations are written by Qwen. For the Qwen recipient, the peer's later steps are therefore same-model text.
- None of T1–T5 compares C with E directly. Each compares one twin under two presentations, so the change does not affect the primary tests' logic.
- Bases: 521 problems now have at least one injection point.

Smoke yield with Qwen as continuer: 11 of 20 problems.
