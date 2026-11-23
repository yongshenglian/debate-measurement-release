# Study K results: pre-registered framing replication, one label cue at a time

- **Protocol:** `PROTOCOL.md` (frozen; `sha256sum -c frozen.sha256` run from the repo root: all 11 files OK after the run).
- **Analysis:** `python3 collaboration/when_debate/k_analyze.py reports/framing-rep-20261122-a out` → `analysis.out`, `out/analysis.json`. Numbers below are copied from `analysis.out`.
- **Run:** 2026-11-22, 14:38–17:48 UTC. Every container exited 0.
  - **Compute:** 2.32 GPU-h for solves plus 8.39 GPU-h for the judge = **10.7 GPU-h**, against a cap of 14.7.
- **Completeness:** 1,500 problems. Invalid first answers: Gemma 0, Qwen 0.
  - The disagreement set D has **640** problems, all with their 24 judge calls (15,360 calls).
  - 18 judge calls (0.1%) hit the first-turn token limit. All 15,360 have a valid JSON answer.
- **Composition of D:**
  - only Qwen right: 120;
  - only Gemma right: 303;
  - both wrong: 217.

## Primary contrasts (Holm over five)

Each outcome is a per-problem mean over 3 decoding seeds. Intervals are 95% problem bootstraps; p values come from the sign-flip test.

| Test | Contrast | Outcome | Estimate (pp) | Holm p | Pre-registered reading | Exact McNemar per replicate (p) |
|---|---|---|---|---|---|---|
| K1 | neutral − owner | accuracy on D | **+3.5** [+1.2, +5.8] | .009 | **J's framing effect replicates** | .022, .009, .18 |
| K2 | yours − neutral | keeps own answer | **−17.5** [−20.5, −14.5] | <.001 | **"Your earlier solution" by itself lowers retention** | all < 1e-15 |
| K3 | another − neutral | takes other's answer | −2.1 [−4.3, −0.1] | .058 | not significant; the CI does not exclude ±3 pp → **inconclusive** | .021, .76, .35 |
| K4 | yoursol − yours | keeps own answer | **+3.4** [+0.7, +6.1] | .034 | removing "earlier" raises retention a little | .12, .63, .005 |
| K5 | misattr − nswap | takes other's answer | **−3.0** [−4.5, −1.6] | <.001 | **the effect follows the label, not authorship** | .002, .053, .010 |

**Reading:**
- **K1** replicates study J's direction at about half its size: J gave +6.8 pp on 250 MMLU-Pro disagreements.
- **K2 is the largest effect.** Labelling Gemma's own solution "Your earlier solution", with nothing else changed, makes Gemma *keep* it less often: 54.2% against 71.7% under "Solution A".
  - The ownership label pushes the reader away from the labelled solution. It does not anchor the reader to it.
- **K4:** dropping "earlier" recovers only 3.4 of those 17.5 pp. Most of the effect comes from the second-person ownership label itself.
- **K5:** put the label on *Qwen's* solution and Gemma adopts that solution less often (5.9% against 9.0%).
  - The effect follows the label, not who wrote the text.
  - This contrast sits near a floor: the second block is strongly preferred (see position below).

## Every framing (on D, n = 640; means over 3 seeds)

| Framing | Blocks (first / second) | Accuracy | Final = own (g0) | Final = other (q0) | Abandons own right (n=303) | Accepts other's right (n=120) | Recovers, both wrong (n=217) |
|---|---|---|---|---|---|---|---|
| owner | YE own / AN oth | 45.2 | 61.8 | 33.1 | 28.3 | 51.9 | 4.5 |
| neutral | A own / B oth | 48.7 | 71.7 | 22.7 | 17.2 | 42.5 | 4.5 |
| yours | YE own / B oth | 43.3 | 54.2 | 39.8 | 34.3 | 58.1 | 4.0 |
| another | A own / AN oth | 49.2 | 74.6 | 20.6 | 14.6 | 38.6 | 4.5 |
| yoursol | "Your solution" own / B oth | 43.9 | 57.6 | 36.4 | 32.1 | 53.1 | 5.2 |
| nswap | A oth / B own | 49.8 | 86.5 | 9.0 | 4.7 | 18.9 | 3.4 |
| misattr | YE oth / B own | 49.5 | 89.8 | 5.9 | 3.5 | 15.0 | 3.1 |
| owner_swap | AN oth / YE own | 46.7 | 69.5 | 24.3 | 20.2 | 38.9 | 4.9 |

YE = "Your earlier solution:", AN = "Another solver's solution:", A/B = "Solution A/B:".

- **Secondary: position.** The judge favours the **second** block.
  - Under neutral labels it keeps its own answer 14.7 pp [12.0, 17.4] less often when that answer comes first.
  - Under owner labels the gap is 7.8 pp [4.7, 10.8].
  - Both p < .001.
- **Accuracy moves much less than behaviour.** Across framings, retention spans 54–90% but accuracy spans only 43–50%.
  - Shifting toward one block trades abandoning right answers against accepting right ones.
  - Gemma is right more often than Qwen on D (303 against 120), so framings that favour Gemma's own block score a little higher.
- **The oracle on D** (either solution right) is 66.1%. No framing exceeds 50%.

## Systems over all 1,500 problems (common denominator; 3 generation calls each)

| System | Accuracy (%) | − Gemma s0 | − vote(g0, g1, q0) | − vote(g0, g1, g2) |
|---|---|---|---|---|
| Gemma s0 | 65.1 | | | |
| vote(g0, g1, q0) | 66.3 | | | |
| vote(g0, g1, g2) | 67.1 | | | |
| judge, neutral on D | 65.6 | +0.6 [−0.7, +1.8] | −0.6 [−1.9, +0.6] | **−1.4 [−2.8, −0.0]** |
| judge, owner on D | 64.2 | −0.9 [−2.4, +0.6] | **−2.1 [−3.6, −0.7]** | **−2.9 [−4.5, −1.3]** |
| oracle g0 or q0 | 73.1 | | | |

- **The judge systems** use judge accuracy averaged over the three seeds on D, and Gemma s0 elsewhere.
- **Neither framing reliably beats Gemma alone.** Both fall below a vote of three independent Gemma samples.
- **None of the 8.0 pp oracle headroom is recovered.**

**Tokens** (prompt + output, both turns, measured):
- Gemma solves 8.00M over 3 samples (about 2.67M each). Qwen solve 2.52M.
- Judge about 2.64M per framing per seed (7.69–8.02M over 3 seeds).
- So a judge system costs about 7.8M tokens and vote(g0, g1, g2) about 8.0M: equal calls and roughly equal tokens.

## What this changes in the paper

- **§6.2:** J1 is now a pre-registered, replicated result, smaller than J's estimate.
- **The new primary finding** is a large, label-driven drop in retention: K2, K4 and K5.
  - The cue is the second-person label, and it acts on whichever block carries it.
  - There is also a large second-position preference.
- **§6.3:** the conclusion "no framing captures the complementarity" holds on untouched problems, now against the stronger baseline vote(g0, g1, g2).
- **Caveats:**
  - one reader (Gemma-3-27B), one task (MMLU-Pro non-math) and one peer (Qwen2.5-7B);
  - the position preference was a secondary outcome;
  - K5 sits near a floor.
