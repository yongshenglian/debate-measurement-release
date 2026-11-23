# Study P: private revision versus peer review on the same disagreement sets (studies K and L)

- **Approved:** by the user on 2026-11-23 ("go ahead with a and B").
- **Motivation:** the third external review found that study L's small gains over the judge's first answer come mainly from new correct answers on problems where both initial answers were wrong, not from adopting the peer's correct answer.
  - Without a matched private-revision baseline, peer review cannot be told apart from an extra review opportunity.
  - The review recommended this experiment as the most valuable addition.
- **Frozen:** this protocol, `collaboration/when_debate/{p_gen.py,p_launch.sh,p_analyze.py}`, the harness files they import, and the reused inputs are hashed in `frozen.sha256`.
  - Hashing happens after the smoke runs and before any full-run output.

## Design

- **Judges and disagreement sets** (all reused read-only):
  - **Gemma-3-27B:** its study K disagreement set, 640 problems.
  - **Qwen3.8-27B:** its study L disagreement set, 650 problems.
  - **Both:** the same 1,500 MMLU-Pro problems. D is the set of problems where the judge's sample 0 and Qwen2.5-7B's sample 0 are both valid and differ.
- **Private revision:** the judge sees the question and only its own frozen sample 0. The prompt is the harness's existing self-review prompt, the one used for private revision in studies R, H and M:
  ```
  Answer the following multiple-choice question step by step.

  Question:
  {q}

  Your earlier solution:
  {own}

  Review your earlier solution and give your final answer.
  ```
  - It then answers the same JSON turn as every peer-review call.
- **Matched to the peer-review calls:**
  - same seeds (67000000 + 10·tid + rep, reps 0–2);
  - same sampling settings, 2,048-token limit, serving image and model checkpoints;
  - thinking off for Qwen3.8.
  - Only the peer's block, and the wording that refers to it, is absent.
- **This private prompt does not match any single framing exactly.** The labels differ across the eight framings, so each framing is compared with the same private-revision call.

## Outcomes (`p_analyze.py`)

- **Primary, per judge** (Holm over 3 within each judge):
  - accuracy on D for framing f minus private revision, for f ∈ {neutral, nswap, misattr};
  - per-problem means over the 3 seeds;
  - two-sided sign-flip test (100,000 draws) and percentile bootstrap (20,000 draws).
- **Readings fixed in advance, per judge and framing:**
  - **"Communication adds beyond private revision":** f − self > 0 with Holm p < .05.
  - **"Private revision does at least as well":** f − self < 0 with Holm p < .05.
  - **Otherwise inconclusive,** unless the interval lies within ±2 pp on D. Then it is called "no difference detected beyond ±2 pp".
- **Secondary:**
  - all eight framings minus private revision;
  - the correctness-cell decomposition for private revision and every framing, as contributions to accuracy over all 1,500 problems:
    - gains the peer's right answer (only the peer was right);
    - loses its own right answer (only the judge was right);
    - new right answer (both were wrong);
  - framing minus private revision within each cell;
  - systems over 1,500 problems against judge s0 and vote(j0, j1, j2);
  - truncation by condition.
- **Interpretation guard:** revising only on D uses the peer's answer to choose D. The private-revision system here is therefore a matched comparison, not a stand-alone policy.

## Budget

- **Calls:** Gemma 640 × 3 = 1,920; Qwen3.8 650 × 3 = 1,950.
- **Estimate** from studies K and L: about 1 GPU-h for Gemma and 3–4 GPU-h for Qwen3.8.
- **Cap by construction:** 4 × 2,400 s + 4 × 5,400 s = 8.7 GPU-h.
- **Order:** Gemma first, then Qwen3.8, both on GPUs 0–3.
