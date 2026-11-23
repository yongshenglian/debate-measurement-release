# Does the "your earlier solution" framing cost the reader? (study J)

- **Approved:** by the user on 2026-11-21 ("yes. and then start to wrtie the paper."), after the proposal:
  > test whether framing causes that loss: show the two disagreeing solutions as anonymous "Solution A / Solution B", in both orders … about 1.5 GPU‑h.
- **Frozen:** this protocol, `collaboration/when_debate/{j_gen.py,j_analyze.py,j_launch.sh}` and `sets.json` are hashed in `frozen.sha256`.
  - Hashing happens after the smoke run (`smoke/`, 3 problems per task and judge, plumbing only) and before any full output.

## Why

- **In every task so far, reading a peer has captured none of the two models' complementarity** (`reports/router-precheck-20261121/NOTE.md`).
  - Gemma's first answer is wrong where Qwen's is right on 21 (GSM8K), 17 (MATH) and 46 (MMLU-Pro) problems.
  - Gemma reading Qwen loses about as many right answers as it gains.
  - Qwen reading Gemma never beats copying Gemma.
- **Owner framing as a cause.** If the "your earlier solution" framing makes a reader overweight or underweight its own answer, a neutral presentation of the same two solutions might pick the right one more often.
- **What is known:**
  - LLMs as pairwise judges are standard.
  - Self-preference in LLM judges is documented (for example, Panickssery et al., 2024, on self-recognition).
- **What is new:**
  - The same content under owner, neutral and swapped-label framing, on reasoning tasks with gold answers.
  - Tied to the accept/abandon accounting, and to the system question: does any framing let a two-model system beat its stronger model at matched calls?

## Problems and solutions (no new solving)

- **Tasks:** the three existing peer sets and their solves.

  | Task | Source | Problems |
  |---|---|---|
  | GSM8K | study R | 1,319 |
  | MATH Levels 4–5 | study H | 600 |
  | MMLU-Pro non-math | study M, neutral | 600 |

- **Disagreement set D:** problems where Gemma's and Qwen's sample-0 answers are both valid and differ.

  | Task | D | Gemma right | Qwen right | Neither |
  |---|---|---|---|---|
  | GSM8K | 99 | 64 | 21 | 14 |
  | MATH | 209 | 134 | 17 | 58 |
  | MMLU-Pro | 250 | 117 | 46 | 87 |

## Conditions (`j_gen.py`)

- **Judges:** Gemma-3-27B and Qwen2.5-7B.
- **Content:** on each problem in D, each judge sees its own sample 0 and the other model's sample 0, under four framings.
  - The rest of the prompt is identical: the task line, then "Review both solutions and give your final answer."

| Condition | First block | Second block |
|---|---|---|
| owner | "Your earlier solution:" + own | "Another solver's solution:" + other. Identical to the original PEER prompt. |
| neutral | "Solution A:" + own | "Solution B:" + other. Same order as owner; only the labels change. |
| neutral_swap | "Solution A:" + other | "Solution B:" + own |
| label_swap | "Your earlier solution:" + **other** | "Another solver's solution:" + **own** |

- **Decoding, two-turn JSON contract and seeds** are as in the original reviews (47000000 + tid, shared by conditions).
- **Token limits** as originally: 1,024 for GSM8K, 2,048 for MATH and MMLU-Pro (`hard_reverse.py`; `m_reverse.py` for the MCQ task line).
- **Reproducibility check:** the owner condition re-runs the original prompt. Its agreement with the original review answers is reported.

## Outcomes (`j_analyze.py`; paired bootstrap, 20,000 draws, seed 20261117)

### Primary

- **J1, framing effect on the reader:** accuracy on D, neutral − owner.
  - For each task × judge (6 tests), Holm over the 6.
- **J2, system effect, Gemma judge:** over all problems, the system answer is the neutral-judge answer on D and Gemma s0 elsewhere.
  - **J2a:** system − Gemma s0 (does reading capture complementarity?).
  - **J2b:** system − vote3, at matched calls: vote3 is a vote over Gemma s0, Gemma s1 and Qwen s0, with ties broken toward Gemma s0. The system also uses 3 calls: Gemma s0, Qwen s0 and the judge.
  - 3 tasks × 2 = 6 tests, Holm over the 6.

### Secondary

- **J3, label vs authorship:** the share of final answers equal to the judge's own first answer, label_swap − owner.
  - If retention follows the label "your earlier solution", it falls under label_swap.
  - If it follows actual authorship (self-recognition or one's own reasoning), it stays.
- **Position:** retention under neutral (own first) − neutral_swap (own second).
- **Averages and rates:** (neutral + neutral_swap)/2 − owner, and label_swap − owner accuracy.
- **Accept/abandon** by condition.
- **The same system numbers** for the owner judge and for the Qwen judge.

### Readings fixed in advance

- **"Owner framing costs the reader":** a J1 contrast is positive after Holm correction.
- **"A neutral judge captures complementarity":** J2a is positive after Holm correction.
- **"… and beats compute-matched voting":** J2b is also positive.

## Budget

- **Jobs:** 558 problems × 4 conditions × 2 judges = 4,464 two-turn calls.
- **Estimate** from study M review timing: about 1.3 GPU-h, plus 6 model loads. **Cap:** 2.5 GPU-h.
