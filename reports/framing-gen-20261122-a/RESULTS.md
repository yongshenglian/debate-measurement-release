# Study L results: study K's framing tests with Qwen3.8-27B as judge

- **Protocol:** `PROTOCOL.md`, with two amendments, both made before any outcome was analysed:
  - the cap was lifted (user: "finish all the runs");
  - all framings are tested as systems, with `frozen_amend1.sha256` at 02:01 UTC.
- **Hashes:**
  - The original `frozen.sha256` still matches every file except `PROTOCOL.md` and `l_analyze.py`, which changed through the documented amendments.
  - `frozen_amend1.sha256` matches.
- **Analysis:** `python3 collaboration/when_debate/l_analyze.py reports/framing-gen-20261122-a out-qwen38` → `analysis-qwen38.out`.
- **Inkling 2 Mini was not run.** Its checkpoint needs GPU compute capability 9.0, and our A6000s are 8.6.
- **Run:** 2026-11-22 20:21 to 2026-11-23 03:52 UTC. All 1,500 solves and all 650 × 24 = 15,600 judge calls completed.
  - Every call gave a valid answer.
  - The judge containers report exit code 124 because their paused timeout fired after the work had finished. Each log has its completion line.
  - **Compute:** 3.4 GPU-h for solves plus 23.8 GPU-h for the judge = **27.1 GPU-h**.
- **Disagreement set D:** 650 problems.
  - judge only right: 467;
  - peer only right: 44;
  - both wrong: 139.
  - Qwen3.8 is much stronger than the Qwen2.5-7B peer: s0 81.1% against Gemma's 65.1% in study K.
- **First turns ending at the 2,048-token limit:** 15.3–18.1% across framings. There is no framing pattern, so this does not explain any framing effect.

## Primary contrasts (Holm over five), against the readings fixed in advance

| Test | Contrast | Qwen3.8 (pp) | Holm p | Study K, Gemma | Reading |
|---|---|---|---|---|---|
| K1 | neutral − owner, accuracy | +0.8 [−0.2, +1.8] | .80 | +3.5 [+1.2, +5.8] | **does not generalise** (the interval excludes effects above +3 pp) |
| K2 | yours − neutral, keeps own | +0.3 [−0.8, +1.3] | 1 | −17.5 [−20.5, −14.5] | **does not generalise** |
| K3 | another − neutral, takes other | −0.1 [−1.0, +0.8] | 1 | −2.1 | null; the interval excludes ±3 pp |
| K4 | yoursol − yours, keeps own | −0.3 [−1.3, +0.7] | 1 | +3.4 | null; the interval excludes ±3 pp |
| K5 | misattr − nswap, takes other | +0.2 [−0.7, +1.1] | 1 | −3.0 [−4.5, −1.6] | **does not generalise** |

- **Qwen3.8 is almost insensitive to the labels and the order.**
  - It keeps its own answer on 88.6–90.4% of D under every framing; Gemma ranged from 54% to 90%.
  - It abandons a right answer 3.1–4.4% of the time (Gemma: 3.5–34%).
  - It accepts a right peer answer 26.5–32.6% of the time.
  - Its second-position preference is gone: −0.4 [−1.5, +0.7] pp under neutral labels, against −14.7 for Gemma.
- **Study K's label effects are therefore judge-specific.**
  - They appear in Gemma-3-27B reviewing a weaker peer, not in Qwen3.8-27B reviewing a much weaker one.

## Systems over all 1,500 problems

- **Confirmatory (amendment; Holm over 2):**
  - nswap − judge s0: **+0.8 [+0.0, +1.5]**, Holm p = .045;
  - misattr − judge s0: **+0.9 [+0.2, +1.7]**, Holm p = .039.
  - **Both replicate** by the pre-registered reading: same sign and Holm p < .05.
- **But the gain is not specific to those framings with this judge.** Every framing gains +0.4 to +0.9 pp over judge s0. Neutral gives +0.8 [+0.0, +1.5], with uncorrected p = .042. Under Holm over 8, none of the eight is significant.
  - So for Qwen3.8, reviewing a peer's solution on disagreements gives a small gain under any framing.
- **No framing beats a vote of three independent Qwen3.8 samples.**
  - vote(j0, j1, j2) scores 82.1%.
  - The framings' differences from it are −0.2 to −0.6 pp, with every interval including 0 and every Holm p = 1.
  - The oracle (j0 or p0 right) is 84.0%.
- **Cost:**
  - **Logical calls:** judge systems 2 × 1,500 + 650 = 3,650 (2.43 per problem, cap 3), against 4,500 for the vote. Each call is 2 inference requests.
  - **Tokens:** a judge system uses about 11.1M (judge s0 about 4.06M, peer 2.52M, one seed of judge calls about 4.47M), against 12.2M for vote(j0, j1, j2).

## What this changes in the paper

- **Study K's large label and position effects did not generalise to a second, much stronger judge.** They are properties of the Gemma judge in this setting, not general facts about LLM review.
- **The small gain of order-sensitive review over the judge's first answer replicated.** With Qwen3.8 it holds for every framing, and in both judges it stays below independent sampling.
- **Caveats:**
  - Qwen3.8 is a different family from Gemma and far stronger than its peer: only 44 problems have only the peer right.
  - Thinking was off.
  - 15–18% of first turns were cut off.
  - One task.
