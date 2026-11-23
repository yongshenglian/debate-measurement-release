# Does the user's independence sentence help debate and peer review on hard problems? (study I)

- **Approved:** by the user on 2026-11-21 ("yes").
  - The proposal: add the user's independence sentence to (1) strong/weak peer review and (2) the answers-only and full-reasoning debates, on study H's hard problems; about 3 GPU-h.
- **Frozen:** this protocol, `collaboration/when_debate/i_gen.py`, `i_reverse.py`, `i_analyze.py` and `i_launch.sh` are hashed in `frozen.sha256` before any full-run output.

## Why

- **On GSM8K** (`peer-warning-20261119-b`), the user's sentence raised Qwen's accuracy with a wrong peer by 7.2 pp, at a 1.1 pp cost with a correct peer.
- **Study H** used neutral review prompts and the papers' own debate wording. It found:
  - Qwen abandons a right answer for a wrong peer 37.7% of the time.
  - Debate correlates agents and loses to a compute-matched 20-sample vote by 3.7–7.0 pp.

## Design

- **Everything is as in study H:** problems, models, decoding, 2,048 tokens, and seeds. The only change is one added sentence.
- **Neutral comparison:** the study H runs, paired by problem (same seeds).
- **Debate (Qwen, 300 problems):**
  - **Arms:** H_full_indep and H_ans_indep. The sentence is added as the last bullet of DMAD's debate instructions:
    > - Take a fresh look at the work and use your own judgement; do not be misled by the other agents, however convincing they may sound.
  - **Wording:** the user's sentence, with "the peer … it" changed to "the other agents … they".
  - **Round 0:** the same prompt and seed as in study H.
- **Peer review (Qwen and Gemma, 600 problems):**
  - Study H's solve files are reused unchanged (symlinked).
  - The sentence goes immediately before the review request:

    | Condition | Sentence |
    |---|---|
    | peer conditions | the user's sentence, verbatim: "Take a fresh look at the work and use your own judgement; do not be misled by the peer, however convincing it may sound." |
    | self | the self_doubt sentence from `self-review-20261119-a`, with "proposal" changed to "solution": "… do not be misled by your earlier solution, however convincing it may sound." |

- **Answer reading:** the same as study H.
  - Gemma-filled reading for the debate arms (blind Gemma on every no-answer output).
  - The JSON integer contract for reviews.

## Outcomes (`i_analyze.py`; paired bootstrap over problems, 20,000 draws, seed 20261117)

### Debate

| Label | Outcome |
|---|---|
| IS1 (primary) | H_ans_indep final vote − H_vote20 (study H's 20 independent samples) |
| IS2 | P1(arm_indep) − P1(arm) for full and answers-only, Holm over two |
| Secondary | P1, G4 gate, final − vote20, pooled-20 − vote20, final-round unanimity (a copying measure), contested rescued/broken |

### Peer review

- **IR (primary, Holm over three):** accuracy with the sentence − neutral accuracy, for:
  - Gemma + Qwen peer;
  - Qwen + Qwen peer;
  - Qwen + Gemma peer.
- **Secondary:**
  - the self conditions;
  - Gemma + Gemma peer;
  - accept and abandon rates (`reverse_mech.py` on the I outputs, compared with study H).

### Readings fixed in advance

- **"Debate with independence beats compute-matched voting":** the IS1 CI lies entirely above 0.
- **"The sentence helps debate":** an IS2 contrast is positive after Holm correction.
- **"The sentence helps peer review":** an IR contrast is positive after Holm correction. A negative one means it hurts, for example by blocking acceptance of a correct strong peer.

## Budget

- **Estimate:** about 3 GPU-h.
  - Debate: H_full took 0.85 h and H_ans 0.52 h in study H.
  - Reviews: Gemma 1.0 h, Qwen 0.3 h.
  - Gemma fill: 0.2 h.
- **Hard cap:** 4 GPU-h.
