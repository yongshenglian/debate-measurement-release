# Do the debate and peer findings hold outside math? MMLU-Pro (study M)

- **Authorised:** under the user's standing instruction of 2026-11-21 to test the GSM8K hypotheses "on math and other hard problem set" without asking each time.
- **Frozen:** this protocol, `collaboration/when_debate/{m_prepare.py,m_reverse.py,m_analyze.py,m_launch.sh}` and the data are hashed in `frozen.sha256` before any full-run output.
  - Smoke outputs are in `smoke/`: Qwen H_full on 20 questions; 40 Qwen solves.

## Problems

- **Source:** MMLU-Pro test (TIGER-Lab, cached), every category except math (math is covered by studies H, I and T).
  - 10,681 questions are eligible.
  - 600 are drawn with seed 20261124 (`data/`). Most have 10 options.
- **Format:** options are numbered 1..k, and the gold answer is the option number.
  - Each question ends with "The final answer is the number of the correct option."
- **Splits:**
  - Debate part: the first 300 questions (`data/mmlupro300.csv`).
  - Peer part: all 600.
- **Smoke run (Qwen):** individual accuracy 55%, 5-agent vote 61%.

## Part 1: debate (Qwen2.5-7B; `hard_gen.py` / `i_gen.py` unchanged)

- **Arms:** H_full, H_ans, H_full_indep, H_ans_indep, and the compute-matched vote H_vote20.
  - The same DMAD prompts, 2,048 tokens and seeds as studies H and I.
- **Answer reading:** the **MCQ reader** (`m_analyze.mcq`).
  - It takes the option number at the start of the last `<answer>` segment, or a single "option N" inside it.
  - **Primary reading:** the MCQ reader, plus blind Gemma (MCQ system prompt) for every output it scores as no-answer.
  - **Audit:** 100 outputs per arm, pass at ≥95%.
- **Outcomes** (paired bootstrap, 20,000 draws, seed 20261117):

  | Label | Outcome |
  |---|---|
  | M1 (primary) | final − vote20 for each arm (study H: vote20 beat every arm) |
  | M2 | P1 contrasts, Holm over three: H_ans − H_full; H_full_indep − H_full; H_ans_indep − H_ans |
  | Secondary | P1, G4 gate, pooled-20 − vote20, final unanimity |

## Part 2: peer review (Qwen2.5-7B and Gemma-3-27B; `m_reverse.py`)

- **Harness:** `reverse_gen.py` with a 2,048-token limit and the multiple-choice task line ("Answer the following multiple-choice question step by step.").
- **Solve:** 2 samples per model.
- **Review:** self, peer_qwen and peer_gemma, run twice:
  - neutral (`out-neutral/`);
  - with the study I sentences (`INDEP=1`, `out-indep/`).
- **Outcomes:**

  | Label | Outcome | Earlier results |
  |---|---|---|
  | M3 | Qwen peer_gemma − self (weak reads strong) | +1.6 on GSM8K, +11.5 on MATH |
  | M4 | Gemma peer_qwen − self (strong reads weak) | −0.2 on GSM8K, −1.5 on MATH |
  | M5 | sentence − neutral, Holm over three: Gemma + Qwen peer, Qwen + Qwen peer, Qwen + Gemma peer | as in study I |
  | Secondary | accept and abandon rates on disagreements (`reverse_mech.py`), compared across GSM8K, MATH and MMLU-Pro | |

## Readings fixed in advance

- Two-sided tests.
- A result **"replicates"** if its 95% CI excludes 0 with the earlier sign (after Holm correction where stated).

## Budget

- **Estimate:** about 6.5 GPU-h. **Cap:** 8 GPU-h.
- **If over:** H_ans_indep is dropped first, then the indep reviews.
