# Truth-sensitivity theory and debate prediction: feasibility check

Date: 2026-11-19. No GPU was used. Files in this folder:
- `load_debate.py` and `transitions.py` (analysis);
- this note.

## Question

The idea under test was a theory that predicts when debate beats voting, from
single-peer adoption rates measured with matched pairs (d′ and bias). The check
also asked whether Study 3's rates predict round-1 transitions in our existing
Qwen2.5-7B debate run (`reports/debate-vote-gsm8k-20261117-a`).

## 1. Novelty check: the theory is largely taken

**When Debate Helps** (Zhao et al., NeurIPS 2026, arXiv 2610.04686; code in
`source/related-work/when-debate-helps`) already does most of it.
- **Extends the martingale null.** It adds "verification evidence"
  pseudo-counts W, with a positive-drift condition (Prop. 2) and a
  correct-minority recovery condition (Prop. 3).
- **Recovery-damage decomposition (Prop. 1):** gain = recovered + newly correct
  − damage.
- **Probes truth sensitivity under a fixed proposal set:**
  - the trace effect, in "equivalent peer-support" units;
  - an answer-stripped, length-matched correct-vs-wrong rationale gap (T − F);
  - free generation.
- **Compares a verification-aware debate instruction** with the standard
  debate prompt.

**Debate Helps Weak Judges** (arXiv 2605.27483) adds two more results.
- Debate helps only when the critic out-classifies the judge and the judge
  verifies rather than treating the critic's claims as testimony.
- A pre-deployment audit predicts when debate helps.

**What remains distinct to us:**
- minimal-pair twins, where one inserted error is the only difference;
- behavioural adoption measured separately for right and wrong readers;
- the cost of each defence with correct peers;
- the warning vs procedure contrast;
- the hedge and confidence results;
- the self-review control.

These are refinements, not a new theory.

## 2. Free validation: the single-peer assay does not transfer to the debate protocol

Round 0 → round 1 in the dense debate (5 agents, each sees 4 peers) and the
sparse debate (each sees 2):

| Situation | Study 3 assay (one peer, our prompt) | Debate (natural peers, debate prompt) |
|---|---|---|
| Right reader, wrong peer(s) only | corruption 15.7% (1 wrong peer) | sparse, 2 wrong peers: 40% (n = 20) |
| Right reader, 1 wrong + 1 right peer | — | sparse: 11.3% (n = 80) |
| Wrong reader, correct peer | rescue 80.7% (1 correct peer) | sparse, 1 correct + 1 other: 46% (n = 28); dense, 4 correct: 90% (n = 21) |
| Wrong reader, wrong peer(s) only, finds the correct answer alone | 45.9% (1 different wrong peer); private second look 40.5% | dense: 0 of 51; sparse: about 3% (n = 63) |

**Why they differ:**
1. **Difficulty confound.** In natural debates, peer correctness tracks problem
   difficulty: when every peer is wrong, the problem is hard. Debate transition
   rates therefore mix the reader's truth sensitivity with difficulty. Twins
   assign peer correctness independently of difficulty, which is why our
   self-recovery rate (about 40–46%) is far above the debate's (about 0–3%).
2. **Prompt.** The debate prompt ("use these opinions carefully as additional
   advice to revise") invites adoption more than our "verify rather than
   following" instruction.
3. **Format degeneration dominates the debate's harm.** Empty extractions rose
   from 153 to 223 (dense) after one round. Right readers whose peers were
   empty often became empty themselves (e.g. 80% at 1 correct peer + 3 empty),
   and the outputs show copied "\) \) \)" loops. In round 1, correct agents
   lost their answer to an empty extraction 131 times (dense) and 97 times
   (sparse), against 21 and 31 switches to a wrong answer. Far more correct
   answers were lost to format contagion than to persuasion.

**Conclusion:** a pre-measured single-peer assay cannot predict debate outcomes
without modelling the protocol's prompt and its degeneration. Even then, the
remaining contribution would be close to the LVD paper's.

## Implication

The proposed "truth-sensitivity theory + predictive assay" paper should not go
ahead as the headline. Studies 1–3 and the self-review control stand as a
well-controlled empirical paper. They now need to be positioned against LVD,
2605.27483 and 2604.06756.

## 3. Format contagion and extraction artifact in Debate-or-Vote (follow-up, same day, no GPU)

**Files:**
- `empties.py`: classifies outputs and holds the robust extractor;
- `rescore.py` and `rescore.out`: the re-scores.

**Robust extractor.** It takes the last brace-closed "final answer: N" in the response, across lines and with
escaped braces (`\{final answer: N\}`). If there is none, it takes the last
`\boxed{N}`. Otherwise it falls back to the published extractor. Validation:
- where the published extractor yields a number, the two agree on 98.1–99.5% of
  responses;
- most disagreements are published-extractor errors (for example, it reads 2
  from `\frac{1}{2}`);
- 12 recovered answers were spot-checked by hand, and all are the model's stated
  answer.

**Finding 1: the decline over rounds is an extraction artifact.** These are four
faithful replications of the pinned author code (commit 82c929e).

| Run | Vote change by round 5, published extractor | Vote change by round 5, robust extractor |
|---|---|---|
| GSM8K dense | −5.4 pp [−9.2, −1.6] | +0.4 pp [−2.2, +3.0] |
| GSM8K sparse | −5.1 pp [−8.3, −2.1] | +1.5 pp [−0.8, +4.0] |
| Arithmetic dense | −16.3 pp [−24.3, −9.0] | −2.0 pp [−5.0, 0.0] |
| Arithmetic sparse | −9.0 pp [−15.0, −3.5] | −2.0 pp [−5.0, 0.0] |

Agents keep stating correct answers; the published regex stops reading them,
because the agents increasingly write `\{final answer: N\}`, often in
repetition loops cut off at the 512-token limit.

**Finding 2: individual accuracy rises with debate.**

| Run | Round 0 | Round 1 |
|---|---|---|
| GSM8K dense | 86.3% | 91.9% |
| GSM8K sparse | 86.0% | 89.4% |
| Arithmetic dense | 95.2% | 98.0% |

The paper offers flat mean agent accuracy across rounds as empirical support for
its martingale theory. With correct extraction, individual accuracy converges
upward toward the vote, while the vote itself stays flat. "Debate ≈ voting"
survives; "debate degrades" and "flat individual accuracy" do not.

**Finding 3: format contagion.** Within the same task and round (sparse ring), a
clean agent with a broken-format neighbour is 16.5 pp [12.9, 20.3] more likely to
break format in the next round. In the dense arm, the probability of breaking
rises with the number of broken peers:

| Broken peers | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| P(break next round) | 3.9% | 19% | 29% | 52% |

**Caveats:**
- The data cover one model (Qwen2.5-7B) and two tasks, from our replication, not
  the authors' raw outputs (which are not released).
- Our published-extractor round-0 individual accuracy (79.7%) is below the
  paper's 87.1%, so their outputs may contain fewer format failures. Their Table 1
  declines (GSM8K vote 0.94 → T5 0.83; arithmetic 0.99 → 0.67) look similar,
  though.
- The robust extractor is a heuristic. A blind audit of recovered cases, or a
  second independent extractor, is still owed.
