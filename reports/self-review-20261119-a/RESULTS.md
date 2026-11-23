# Self-review: the check and independence sentences without a peer

The protocol is `self-review-v1` (`PROTOCOL.md`, frozen at prepare). The run is
`self-review-20261119-a`, launched and audited on 2026-11-19. It answers the
user's question: "what happens if we ask the llm to look at its own answer with
check or my sentence?"

## Integrity

- **Replay audit:** passed. All 5,176 recipient calls were replayed
  independently and the full grid of 647 tasks × 4 conditions × 2 replicates is
  complete. There was one invalid generation (`self_doubt`), scored as wrong.
- **GPU time:** 14,140 s, or 3.9 GPU-hours, against the cap of 8.
- **Private rerun:** it used Study 3's exact prompt and seed, and 98.0% of
  answers came out identical (1,268 of 1,294). vLLM is not bit-deterministic,
  so the rerun reached 85.5% where Study 3 had 85.9%.
- **Matching:** every prompt was padded to Study 3's per-task token target.

## Accuracy

The reader's first answer was right on 79.6% of the 647 tasks.

| Condition | Accuracy | vs first answer | Corruption (right → wrong) | Rescue (wrong → right) | Answer changed |
|---|---|---|---|---|---|
| private | 85.5% | +6.0 [3.7, 8.2] | 2.8% | 40.2% | 14.3% |
| self_check | 86.2% | +6.6 [4.4, 9.0] | 2.7% | 43.2% | 15.1% |
| self_indep | 86.5% | +6.9 [4.8, 9.1] | 2.3% | 42.8% | 15.1% |
| self_doubt | 86.4% | +6.8 [4.7, 9.0] | 2.2% | 42.0% | 14.9% |

## Primary results (Holm over four, ±2 pp margin)

| ID | Contrast | Estimate (pp) | 95% CI | 90% CI | Holm p | Label |
|---|---|---|---|---|---|---|
| R1 | self_check − private | +0.7 | [−0.6, +2.0] | [−0.4, +1.8] | 0.36 | equivalent within 2 pp |
| R2 | self_indep − private | +0.9 | [−0.2, +2.2] | [−0.1, +1.9] | 0.32 | equivalent within 2 pp |
| R3 | net(C/E_full_check) − self_check | +2.6 | [+0.7, +4.4] | [+1.0, +4.1] | 0.033 | **benefit** |
| R4 | net(C/E_full_indep) − self_indep | +2.1 | [+0.3, +3.9] | [+0.6, +3.6] | 0.068 (uncorrected 0.023) | inconclusive |

"net" is the mean of Study 3's C and E conditions, i.e. half the peers wrong.

## Secondary results (uncorrected)

| ID | Contrast | Estimate (pp) | 95% CI | p |
|---|---|---|---|---|
| S1 | self_doubt − private | +0.9 | [−0.4, +2.2] | 0.24 |
| S2 | self_doubt − self_indep | −0.1 | [−1.0, +0.9] | 1.0 |
| S10 | self_check − self_indep | −0.2 | [−1.5, +1.0] | 0.81 |
| S3 | E_full_check − self_check | −2.4 | [−4.7, 0.0] | 0.054 |
| S4 | C_full_check − self_check | +7.5 | [+5.6, +9.4] | 5e−16 |
| S5 | E_full_indep − self_indep | −3.9 | [−6.2, −1.6] | 0.001 |
| S6 | C_full_indep − self_indep | +8.1 | [+6.3, +10.0] | 7e−19 |
| S7 | net(full) − private | 0.0 | [−2.1, +2.2] | 1.0 |
| S8 | net(full_check) − private | +3.2 | [+1.4, +5.3] | 0.001 |
| S9 | net(full_indep) − private | +3.1 | [+1.2, +4.9] | 0.002 |

## Reading

1. **Without a peer, the sentences do almost nothing.** A plain second look
   already gains 6 pp over the first answer. Adding the check sentence, the
   user's sentence, or the user's sentence aimed at the reader's own answer adds
   less than 1 pp. Each is equivalent to plain private within ±2 pp. Telling the
   model to distrust its own earlier answer (self_doubt) did not hurt either.
2. **So the sentences act on how the reader handles a peer, not on how it
   reviews itself.** Study 3's net gain of about 3 pp over private, which was
   exploratory there, is confirmed against this run's private (S8 and S9). It
   survives against the matched self-review control: +2.6 pp for check, which
   is Holm-significant, and +2.1 pp for indep, which is significant only before
   correction.
3. **A wrong peer still hurts, even with the sentence.** It costs 2.4 pp under
   check and 3.9 pp under indep, compared with reviewing alone under the same
   sentence. A correct peer adds 7.5–8.1 pp. With half the peers wrong, the
   peers are worth about +2–3 pp, but only when a sentence tells the reader what
   to do. With no sentence they are worth 0.0 pp (S7).

## Limitations

- Q3 and Q4 pair this run with Study 3's stored outputs. Seeds, prompts and
  lengths match, but the 98% replay shows that the two runs are not
  bit-identical.
- The self sentences differ from the peer sentences in their wording, as they
  must: "your proposal" instead of "the peer's proposal", and no peer clause.
