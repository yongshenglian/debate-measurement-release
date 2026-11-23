# Study K: pre-registered replication of the framing effect, one label cue at a time

- **Approved:** by the user on 2026-11-22 ("approave all. we will do the human labelling."), after the proposal:
  > Framing replication on untouched MMLU-Pro problems (~600 disagreements, Gemma as judge), one label cue changed at a time, both orders, three decoding seeds; optionally a three-independent-Gemma vote baseline. ~12 GPU-h, cap 15 GPU-h.
- **Motivation:** an external review of the paper draft. Study J's one significant result (Gemma, MMLU-Pro, neutral − owner +6.8 pp) is test-sensitive:
  - the bootstrap Holm p is .047;
  - the exact McNemar Holm p is .069 (29 problems fixed vs 12 broken).
- **Study J's framings changed several cues at once.** The exploratory "yours" label effect was found after seeing the data.
- **This study:**
  - re-tests the J contrast on untouched problems;
  - separates the cues;
  - pre-specifies an exact-style test.
- **Frozen:** this protocol, `collaboration/when_debate/{k_prepare.py,k_gen.py,k_analyze.py,k_launch.sh}` and `data/` are hashed in `frozen.sha256`.
  - Hashing happens after the smoke run (`smoke/`, plumbing only, not analysed for outcomes) and before any full-run output.

## Problems (no overlap with any earlier study)

- **Source:** MMLU-Pro test, non-math categories (10,681 questions).
  - Same seed (20261124) and shuffle as study M.
  - Study M used positions 0–599; study K takes positions 600–2,099, giving 1,500 questions.
  - `k_prepare.py` asserts that positions 0–599 reproduce study M's question ids.
- **Format:** question text, option numbering and gold labels exactly as in study M.

## Generation (`k_gen.py`, study M harness)

- **Harness:**
  - multiple-choice task line;
  - free text up to 2,048 tokens, then "State only your final answer as JSON." with an integer schema;
  - T 0.7, top-p 0.95, top-k 20;
  - vLLM 0.29 at the pinned digest, no network.
- **Solves:**
  - Gemma-3-27B samples 0, 1 and 2; Qwen2.5-7B sample 0.
  - Seed 61000000 + 10·tid + sample.
  - Gemma samples 1 and 2 exist only for the vote baselines.
- **Disagreement set D:** Gemma s0 and Qwen s0 both valid and different. Expected about 600 problems (42% in study M).
- **Judge:** Gemma-3-27B on every problem in D.
  - Eight framings, identical except for the two block labels and their order.
  - "own" = Gemma s0, "other" = Qwen s0.
  - Each framing × 3 decoding replicates, with seed 67000000 + 10·tid + replicate, shared by the eight framings.

| Framing | First block | Second block | Role |
|---|---|---|---|
| owner | Your earlier solution: own | Another solver's solution: other | the study R/H/M/J prompt |
| neutral | Solution A: own | Solution B: other | K1 |
| yours | Your earlier solution: own | Solution B: other | K2 (own-block label only) |
| another | Solution A: own | Another solver's solution: other | K3 (other-block label only) |
| yoursol | Your solution: own | Solution B: other | K4 ("earlier" removed) |
| nswap | Solution A: other | Solution B: own | position; K5 baseline |
| misattr | Your earlier solution: other | Solution B: own | K5 (label on the other model's solution) |
| owner_swap | Another solver's solution: other | Your earlier solution: own | position under owner labels |

Every prompt ends "Review both solutions and give your final answer." No other wording changes.

## Outcomes (`k_analyze.py`)

- **Unit of analysis:** the problem. Per problem and framing, each outcome is averaged over the 3 replicates.
- **Primary test:** a two-sided sign-flip randomisation test on per-problem differences.
  - 100,000 draws, seed 20261123.
  - Holm over K1–K5.
- **Intervals:** percentile bootstrap over problems (20,000 draws).
- **Sensitivity:** exact McNemar with transition counts on each replicate separately.

### Primary (Holm over 5)

| Test | Contrast | Outcome | Expected from study J |
|---|---|---|---|
| K1 | neutral − owner | accuracy on D | positive (replication of J1) |
| K2 | yours − neutral | final = Gemma's own s0 | negative (exploratory J finding) |
| K3 | another − neutral | final = Qwen's s0 | none (two-sided) |
| K4 | yoursol − yours | final = Gemma's own s0 | none (two-sided) |
| K5 | misattr − nswap | final = Qwen's s0 (the labelled block) | negative (label, not authorship) |

### Readings fixed in advance

- **"Study J's framing effect replicates":** K1 > 0 with Holm p < .05.
- **"The label 'Your earlier solution' by itself lowers retention":** K2 < 0 with Holm p < .05.
- **"The effect follows the label, not authorship":** K5 < 0 with Holm p < .05.
- **K3 and K4** say whether the "another solver" label and the word "earlier" carry any effect.
- **A null result** is reported as inconclusive unless its interval excludes effects of ±3 pp.
  - That margin was fixed here before data and is not chosen to fit results.

### Secondary

- **Accuracy and retention** for every framing.
- **Position:**
  - neutral vs nswap retention;
  - owner vs owner_swap.
- **Rates within D**, by framing:
  - abandon (Gemma right, Qwen wrong);
  - accept (Qwen right, Gemma wrong);
  - recovery when both are wrong.
- **Systems over all 1,500 problems**, with a common denominator; an invalid answer counts as wrong:
  - the judge on D and Gemma s0 elsewhere, for the neutral and owner framings;
  - compared with Gemma s0, vote(g0, g1, q0) and vote(g0, g1, g2), ties going to g0;
  - all are 3-generation-call systems. Measured prompt and output tokens are reported for each, so equal calls and equal tokens can be told apart.
- **The oracle**, g0 or q0 right.

## Budget

- **Calls:**
  - solves: 1,500 × 4 = 6,000;
  - judge: about 600 × 8 × 3 ≈ 14,400.
- **Estimate** from study M timings: about 10–12 GPU-h.
- **Cap by construction** in `k_launch.sh`: 4 × (3,000 s + 10,200 s) = 14.7 GPU-h.
  - Output is written in chunks of whole problems.
  - If a lane times out, only problems with all 24 judge calls enter the analysis, and the shortfall is reported.
