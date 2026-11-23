# Study P results: peer review against private revision, same judge, same problems

- **Protocol:** `PROTOCOL.md`. Frozen after the smoke runs and before full output; `sha256sum -c frozen.sha256` gives all OK after the run.
- **Analysis:** `p_analyze.py` → `analysis-gemma.out`, `analysis-qwen38.out`.
- **Run:** 2026-11-23, 04:19–05:10 UTC. Every container exited 0.
  - Gemma: 1,920 calls. Qwen3.8: 1,950 calls.
  - Every call gave a valid answer.
  - **Compute:** 0.9 GPU-h (Gemma) plus 2.1 GPU-h (Qwen3.8) = **3.0 GPU-h**.
- **First turns ending at the token limit under private revision:** Gemma 0.9%, Qwen3.8 13.4%. Under peer review: 0.1–0.3% and 15–18%.

## Primary: accuracy on D, framing − private revision (Holm over 3 per judge)

| Judge | neutral − self | nswap − self | misattr − self | Pre-registered reading |
|---|---|---|---|---|
| Gemma-3-27B (D = 640) | +0.3 [−2.6, +3.1] | +1.4 [−0.9, +3.6] | +1.1 [−1.1, +3.3] | all **inconclusive** (Holm p ≥ .81; intervals exceed ±2 pp) |
| Qwen3.8-27B (D = 650) | +0.3 [−0.9, +1.5] | +0.3 [−0.9, +1.5] | +0.6 [−0.7, +1.9] | all **no difference detected beyond ±2 pp** (Holm p = 1) |

- **None of the three prespecified peer-review policies showed a statistically detectable accuracy advantage over the tested private-revision baseline, for either judge.** Intervals are per contrast, not simultaneous. Qwen3.8's exclude differences above 2 pp; Gemma's permit gains up to about 3.6 pp.
- The private-revision prompt differs from the peer prompts in labels, position and instruction, so this is a baseline comparison, not an isolated effect of peer content.
- **For Gemma, three framings did significantly worse than private revision** (secondary, unadjusted):
  - yours: −5.1 [−8.5, −1.7];
  - yoursol: −4.6 [−7.8, −1.3];
  - owner: −3.2 [−6.5, +0.1].

## Routes: contributions to accuracy over all 1,500 problems (pp)

| Judge | Condition | Gains peer's right answer | Loses own right answer | New right answer, both wrong | Net |
|---|---|---|---|---|---|
| Gemma | private revision | +1.51 | −2.33 | +1.29 | +0.47 |
| Gemma | neutral | +3.40 | −3.47 | +0.64 | +0.58 |
| Gemma | nswap | +1.51 | −0.96 | +0.49 | +1.04 |
| Gemma | misattr | +1.20 | −0.71 | +0.44 | +0.93 |
| Gemma | owner | +4.16 | −5.71 | +0.64 | −0.91 |
| Qwen3.8 | private revision | +0.36 | −0.62 | +0.91 | +0.64 |
| Qwen3.8 | neutral | +0.89 | −1.02 | +0.91 | +0.78 |
| Qwen3.8 | nswap | +0.84 | −0.96 | +0.89 | +0.78 |
| Qwen3.8 | misattr | +0.89 | −0.98 | +1.00 | +0.91 |

- **"Gains peer's right answer" under private revision** means the judge reached the answer the peer had without seeing it.

**Reading:**
- **Qwen3.8:**
  - Private revision alone gains +0.6 [+0.0, +1.3] pp over the judge's first answer (p = .051), almost entirely from new right answers where both were wrong (+0.91 pp). Peer review has the same aggregate both-wrong contribution, but not on the same problem–seed outcomes: under neutral labels 28 are correct under both, 13 only under peer review, 13 only under private revision (`paired-outcomes.out`, post hoc, descriptive).
  - The peer adds about +0.5 pp of the peer's right answers but costs about −0.4 pp of the judge's own right answers. The net is about zero.
  - **Private revision recovers a similar aggregate gain, so study L's improvement over the first answer does not establish an additional benefit from peer review.**
- **Gemma:**
  - Private revision also gains a little (+0.5 [−0.5, +1.4] pp).
  - Under neutral or owner labels, the peer raises adoption of its right answers well above private revision (+1.9 to +3.1 pp in that cell) but raises losses of Gemma's own right answers about as much or more. The new-answer route falls by about 0.6 pp.
  - **nswap and misattr do not gain more peer answers than private revision does** (+0.0 and −0.3 pp in that cell). Their advantage is losing fewer of Gemma's own right answers (+1.4 and +1.6 pp), offset by fewer new answers (−0.8 pp).
  - Under neutral labels, relative to private revision: +1.89 (peer right), −1.14 (own right), −0.65 (both wrong). The near-cancellation involves all three routes.
  - So even for Gemma, the policies that beat its first answer do so mostly by protecting its own right answers when its solution comes second, not by adopting the peer's.
- **Peer review changed the distribution of recoveries and corruptions; no tested framing, for either judge, showed a detectable net advantage over private revision.**
- **Systems:**
  - private revision − vote(j0, j1, j2): Gemma −1.5 [−2.8, −0.2], Qwen3.8 −0.4 [−1.5, +0.6].
  - Revising only on D uses the peer's answer to choose D, so this is a matched comparison, not a stand-alone policy.
