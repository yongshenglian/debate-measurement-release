# Can debate beat voting when there is room above the vote? (study H)

- **Approved:** by the user on 2026-11-20 ("approved"), after the proposal: "a harder task where agents disagree more often … test the safe settings there, plus deliberately weaker peers to check the prediction above … about 4–6 GPU‑h".
- **Frozen:** this protocol, `collaboration/when_debate/hard_prepare.py`, `hard_gen.py`, `hard_reverse.py`, `hard_analyze.py` and the data files are hashed in `frozen.sha256` before any full-run output.
  - Smoke outputs in `smoke/` and `smoke2/` were used only to check plumbing, truncation and timing.

## Why

- **Study S (GSM8K, Qwen)** found settings that make debate safe:
  - answers-only sharing;
  - Debate-or-Vote wording;
  - no chat template.
- **But no setting beat voting,** because the round-0 vote was already 93–95%.
- **Study R found an accounting rule** for reading a peer's work:
  - net change ≈ (own wrong & peer right) × accept rate − (own right & peer wrong) × abandon rate;
  - the abandon rate was about 30% for both the strong and the weak model.
- **That rule predicts** that the strong model is harmed when the weak peer is wrong more often where the strong model is right.

## Problems

- **Source:** the MATH test set (hendrycks), Levels 4–5, keeping only problems whose gold answer (last `\boxed{}` of the reference solution) is an integer.
  - 1,498 problems are eligible.
  - They are shuffled with seed 20261123 (`hard_prepare.py`).
- **Settings part:** the first 300 (`data/math_hard300.csv`): 172 Level 4, 128 Level 5.
- **Peer part:** the first 600 (`data/tasks.jsonl`), which include those 300. Labels are in `data/labels.json` and never enter a prompt.
- **Why integers only:** this keeps the validated numeric readers and the JSON integer contract usable, with no symbolic matching.

## Part 1: debate settings (Qwen2.5-7B, BF16, pinned vLLM 0.29)

- **Harness:** `settings_gen.py`, unchanged (study S), through `hard_gen.py`.
- **Common settings:**
  - DMAD prompt family, chat template.
  - 5 agents, round 0 plus 3 debate rounds.
  - T 1.0, top-p 0.9, the same seeds as study S.
  - **2,048 max tokens.** Deviation from study S: in the smoke run, 512 and 1,024 truncated 21–35% of solutions.

| Arm | Debate wording | Peers' content shown | Role |
|---|---|---|---|
| H_full | DMAD | full reasoning + answer | harmful on GSM8K (S2) |
| H_ans | DMAD | answers only | safest on GSM8K (S6) |
| H_dovword | DoV ("additional advice") | full | safe on GSM8K (S3) |
| H_vote20 | — | — | **compute-matched vote:** 20 independent round-0 samples (agents 0–4 identical to the debate arms' round 0) |

### Reading answers

- **Primary reading, "Gemma-filled":**
  - our DMAD reader (validated + unclosed-tag rule);
  - plus a blind Gemma-3-27B extraction (format-contagion system prompt, greedy) for **every** output our reader scores as no-answer.
  - This was a post-hoc check in study S; it is pre-registered here.
- **Also reported:** our reader alone.
- **Audit:** the same as study S. 100 random outputs per arm, 50 from round 0 and 50 from the final round (`settings_audit.py`). Pass at ≥95% agreement on outputs where our reader returns a number.

### Outcomes (`hard_analyze.py`; paired bootstrap over problems, 20,000 draws, seed 20261117)

| Label | Outcome |
|---|---|
| HS1 (primary) | P1 of H_ans: 5-agent vote at round 3 − round-0 vote |
| HS2 | H_ans final vote − H_vote20 vote (compute-matched) |
| HS3 | P1 contrasts H_ans − H_full and H_dovword − H_full, Holm over the two |
| Secondary | P1, P2 (G4 gate: keep the round-0 vote when ≥4/5 agree) and final − vote20 for every arm; rescued/broken problems by round-0 agreement; individual accuracy and no-answer rate by round |

### Readings fixed in advance

- **"Debate beats voting on hard problems":** the HS1 CI lies entirely above 0.
- **"… and beats compute-matched voting":** also requires the HS2 CI to lie entirely above 0.
- **If HS1 is positive but HS2 is not:** debate beats a 5-sample vote but not an equally expensive vote.
- **The GSM8K safety finding replicates:** an HS3 contrast is positive after Holm correction.

## Part 2: peers on hard problems (Gemma-3-27B W4A16 and Qwen2.5-7B)

- **Harness:** `reverse_gen.py`, unchanged (study R), through `hard_reverse.py`.
  - The only change is the free-text limit, 1,024 → 2,048 tokens. In the smoke run, 1,024 truncated 19% (Qwen) and 35% (Gemma).
- **Design:** the same as study R:
  - two solutions per model;
  - each model reviews its own sample 0 under three conditions: self, peer_qwen, peer_gemma;
  - an anonymous peer, the two-turn JSON integer contract.

### Outcomes (`reverse_analyze.py` and `reverse_mech.py`, unchanged)

| Label | Outcome |
|---|---|
| HR1 (primary) | Gemma peer_qwen − Gemma self |
| HR2 | Qwen peer_gemma − Qwen self |
| HR3 (prediction test) | predicted net for Gemma reading Qwen |

- **HR3 formula:** predicted net = [N(Gemma first answer right, Qwen peer wrong) × (−0.288) + N(Gemma wrong, Qwen right) × (+0.882)] / N.
  - The rates are the pooled GSM8K rates from study R.
  - N is counted from the stage-1 solves.
  - **Supported** if the predicted value lies inside the HR1 95% CI.
- **HR4:** pooled accept and abandon rates for each reviewer (Wilson CIs). Does the GSM8K pattern replicate?
  - Pattern: strength raises acceptance of a correct peer, but not resistance to a wrong one.
  - Reported descriptively, with the Gemma − Qwen difference.

### Readings fixed in advance

- **"A strong model gets worse after reading a weaker one on hard problems":** the HR1 CI lies entirely below 0.

## Budget and cap

- **Smoke timing, small batches:**
  - about 300 s per debate arm on 20 problems;
  - solve 46 s (Qwen) / 170 s (Gemma) per 80 calls;
  - review 79 s / 323 s per 120 calls.
- **Estimate:** about 4–5 GPU-h in total, including the Gemma fill and audit. **Hard cap 6 GPU-h.**
  - Each container has a time limit.
  - If the cap is reached, H_dovword is dropped first and logged as a deviation.
