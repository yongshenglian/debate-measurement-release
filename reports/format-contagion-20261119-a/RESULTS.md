# Format contagion in multi-agent debate: results

- **Protocol:** `format-contagion-v1` (`PROTOCOL.md`, frozen with hashes in
  `frozen.sha256` before generation). One deviation is logged: a `__main__`
  guard added to the audit runner after a start-up crash, before any output.
- **Run:** 2026-11-19.
- **GPU time:** Qwen 5,200 calls in about 38 min on 1 A6000; Gemma 400 calls in
  about 35 min, including model load. About 1.3 GPU-hours in total.
- **Completeness:** all outputs complete, both containers exited 0, and the
  analysis is the frozen `fc_analyze.py` (`analysis.json`).

## Context

In our faithful replications of Debate-or-Vote (NeurIPS 2025; pinned author
code), re-scoring with a robust extractor removed the reported vote decline over
debate rounds:
- GSM8K: from −5.4 pp to +0.4 pp;
- arithmetic: from −16.3 pp to −2.0 pp.

See `reports/peer-theory-20261119/NOTES.md` §3.

This study tests the two links in that explanation:
- **B:** is the robust extractor right?
- **A:** do broken answer formats *spread* from peer to reader?

## B. Blind extraction audit: the robust extractor is validated

Gemma-3-27B, a different model family, saw each response with no question,
label or category, and reported the final numeric answer the model committed
to.

| Stratum (n) | Gemma = robust | Gemma = published | Robust correct | Gemma correct |
|---|---|---|---|---|
| Published fails, robust recovers (240) | **99.6%** | 0% | 97.1% | 96.7% |
| The two disagree (60) | 91.7% | 6.7% | 88.3% | 93.3% |
| Both agree (60) | 100% | 100% | 91.7% | 91.7% |
| Both fail (40) | — | — | 0% | 57.5% |

- **Pass criterion met:** at least 95% agreement on recovered cases.
- **The robust extractor is conservative.** Gemma finds a correct committed
  answer in 23 of 40 responses that both extractors miss, so re-scored accuracy
  is, if anything, an underestimate.

## A. Controlled twin test: a peer's broken format causes the reader's format to break

- **Problems:** 130 GSM8K problems.
- **Reader and peers:** Qwen2.5-7B reader with its own round-0 answer; 4 peers
  given verbatim.
- **Setup:** the exact Debate-or-Vote dense round-1 prompt and decoding.
- **Manipulation:** only the peers' final-answer markup was changed; every
  peer's content and stated answer is identical across conditions.
- **Samples:** 8 seeds per problem and condition, shared across conditions.

| Peer condition | Reader's format unreadable by published extractor | Accuracy as the published code scores it | Actual accuracy (robust) | Reader hit the 512-token limit |
|---|---|---|---|---|
| clean | 2.7% | 90.6% | 93.0% | 24.6% |
| 1 peer `\{final answer: X\}` | 8.0% | 85.5% | 93.0% | 31.0% |
| 4 peers `\{final answer: X\}` | 39.9% | 56.0% | 92.5% | 42.4% |
| 1 peer + repetition loop | 86.9% | 12.4% | 92.5% | 95.7% |
| 4 peers + repetition loop | 94.3% | 5.1% | 93.0% | 96.9% |

### Primary results (Holm over three)

| ID | Contrast | Estimate | 95% CI | Holm p | Prediction |
|---|---|---|---|---|---|
| F1 | 1 escaped peer − clean, broken format | **+5.3 pp** | [+3.2, +7.5] | 6e−6 | confirmed |
| F2 | 4 looping peers − clean, broken format | **+91.6 pp** | [+90.0, +93.3] | 4e−39 | confirmed |
| F3 | 4 looping peers − clean, actual accuracy | **0.0 pp** | [−1.1, +1.1] | 1.0 | within ±2 pp, as predicted |

### Secondary results

- **Dose-response:** 4 escaped peers vs 1 gives +31.9 pp [+27.7, +36.2]; 4
  looping peers vs 1 gives +7.4 pp [+3.9, +11.0].
- **Actual accuracy** changes by at most 0.5 pp in any condition (all CIs within
  ±2 pp).
- **Under the published scorer,** the same unchanged reasoning loses 5.1 pp (one
  escaped peer) to 85.5 pp (four looping peers).

## Reading

1. **Format is contagious, and content is untouched.** Readers copy their
   peers' answer markup and repetition loops: a single looping peer makes 87% of
   readers loop and run into the token limit. What they *answer* does not
   change.
2. **The published scorer turns contagion into apparent reasoning decline.**
   Scored with the Debate-or-Vote extractor, these readers look up to 85 pp
   worse when their peers' markup is broken, although their answers are as
   accurate as with clean peers.
3. **This explains the decline in our replications.** Across debate rounds,
   broken markup spreads: within the same task and round, a clean agent next to
   a broken neighbour is 16.5 pp [12.9, 20.3] more likely to break format. The
   published extractor then counts these agents as wrong, which produces the
   reported "debate degrades with more rounds". With extraction validated (B),
   the vote does not decline: GSM8K +0.4 pp, arithmetic −2.0 pp ns.

## Limitations

- **One reader model** (Qwen2.5-7B, as in the paper's main table), on GSM8K
  stimuli. The replication re-scores cover GSM8K and arithmetic.
- **The loop manipulation** used a fixed pattern taken from the observed logs.
  Natural loops vary.
- **The original paper's outputs are not released,** so we cannot re-score them.
  With the published extractor, our replication's round-0 individual accuracy
  (79.7%) is below the paper's 87.1%. Their reported declines look similar
  (GSM8K 0.94 → 0.83; arithmetic 0.99 → 0.67), but whether they are equally
  affected is unverified.
- **vLLM vs Hugging Face generate:** the twin test used vLLM, while the
  replication used the authors' HF generate. Both use the same raw prompt and
  decoding parameters.

## Amendment v2: generality across readers and prompt formatting

- **Design:** same stimuli, seeds 0–3 (130 problems × 5 conditions × 4 seeds per
  arm). Pre-registered before any v2 output.
- **Deviations:** three Mistral serving fixes, all logged in `PROTOCOL.md` and
  all made before any output.
- **Analysis:** frozen `fc_analyze_v2.py`, output `analysis_v2.json`.
- **GPU time:** about 2.2 GPU-h.

**Increase in broken reader format over clean peers** (pp, 95% CI). Actual
(robust) accuracy changed by at most 1.5 pp in every cell.

| Reader | 1 escaped peer | 4 escaped peers | 1 looping peer | 4 looping peers |
|---|---|---|---|---|
| Qwen2.5-7B, raw prompt (Debate-or-Vote) | +5.8 [+2.9, +8.7] | +37.1 [+32.1, +42.1] | +83.5 [+79.6, +87.1] | +92.5 [+90.0, +94.8] |
| Qwen2.5-7B, chat template | +3.1 [+1.2, +5.2] | +20.8 [+15.4, +26.5] | +3.5 [+1.5, +5.8] | +15.0 [+10.8, +19.4] |
| Mistral-Small-3.1-24B, chat template | +0.6 [−2.3, +3.3] | +3.1 [−0.2, +6.3] | +0.6 [−2.9, +4.0] | +10.0 [+5.8, +14.2] |
| Gemma-3-27B, chat template | 0.0 [−0.6, +0.6] | +1.0 [0.0, +2.3] | −0.2 [−0.6, 0.0] | +1.0 [+0.2, +2.1] |

**Clean-peer baselines:**

| Reader | Clean-peer broken rate | Clean-peer token-limit rate |
|---|---|---|
| Qwen raw | 2.7% | 23% |
| Qwen chat | 1.7% | 0.2% |
| Mistral | 8.5% | 0.6% |
| Gemma | 0.2% | 0.2% |

**Primary v2 results** (loop4 − clean, Holm over 6):

| Arm | Broken format | Holm p | Actual accuracy |
|---|---|---|---|
| Qwen chat | +15.0 pp | 7e−11 | +1.0 pp, within ±2 |
| Mistral | +10.0 pp | 5e−5 | −0.6 pp, within ±2 |
| Gemma | +1.0 pp | 0.5 | +1.0 pp, within ±2 |

### Reading

- **Contagion is real but depends on model and protocol.** It is extreme for a
  7B model prompted raw, without a chat template, which is the Debate-or-Vote
  setup. There, a single looping peer pulls 84% of readers into the loop.
- **A chat template removes loop contagion almost entirely** (+3.5 pp from one
  looping peer). Copying of escaped markup survives it (+21 pp with four peers).
- **The larger readers are far more resistant.** Mistral-24B is affected only
  by four looping peers (+10 pp); Gemma-27B is essentially immune.
- **In no arm does contagion change what the reader answers.** It changes only
  whether a brittle extractor can read it.

### Consequence for the claim

- **Specific, not general.** The "debate degrades over rounds" result in our
  Debate-or-Vote replications is an artifact of that protocol: raw prompting,
  a last-`{...}` regex and a 7B reader. It is not a general property of LLM
  debate.
- **Two lessons for multi-agent evaluation:** validate extraction per round,
  and remember that the measurement can fail in a way that spreads between
  agents.
- **Other models in the original paper are untested.** Whether its Llama-3.1-8B
  results are affected is not known.
