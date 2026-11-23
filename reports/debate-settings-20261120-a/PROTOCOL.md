# Which debate settings make debate help or hurt? (study S)

- **Approved:** by the user on 2026-11-20 ("do all the tests you proposed").
- **Frozen:** this protocol, `collaboration/when_debate/settings_gen.py` and `settings_analyze.py` are hashed in `frozen.sha256` before any full-run output. Smoke outputs are only used to check the plumbing.

## Motivation

- **Exploratory re-analysis** (`reports/when-debate-helps-20261120/`): the same model (Qwen2.5-7B) on the **same 300 GSM8K problems** behaves in opposite ways in the two released protocols.
  - **Debate-or-Vote:** on contested problems, debate rescues more wrong votes than it breaks.
  - **DMAD (fixed):** debate breaks more than it rescues, even on problems where all agents agreed.
- **Shared by both protocols:** T 1.0, top-p 0.9, dense topology, 5 agents.
- **Different between them:**
  - the debate-round wording;
  - the chat template (DoV prompts raw; DMAD uses the template);
  - max new tokens (512 vs 1,024);
  - the answer format (`{final answer: X}` vs `<answer>` with a stop string).
- **Goal:** find which setting turns debate from helpful to harmful, and test a positive design: debate only when agents disagree.

## Design

- **Model:** Qwen2.5-7B-Instruct BF16 (`models/debate-or-vote/qwen2.5-7b-instruct-a09a354`), pinned vLLM 0.29 image.
- **Problems:** the 300 GSM8K test problems used by both earlier runs (`gsm8k_test300.csv`).
- **Debate:** 5 agents. Round 0 plus 3 debate rounds; "final" means round 3.
- **Decoding:**
  - T 1.0, top-p 0.9 in every arm.
  - Per-agent seeds: `20261120 + 100000·problem + 1000·round + agent`, the same in every arm. Arms with identical round-0 prompts therefore have identical round 0.
- **Prompts:** taken verbatim from the released code; see `settings_gen.py`.

| Arm | Prompt family (round 0 + answer format) | Debate wording | Chat template | Max tokens | Peers' content shown |
|---|---|---|---|---|---|
| S1_dov | DoV | DoV | raw | 512 | full |
| S2_dmad | DMAD | DMAD | chat + generation prompt | 1,024 | full (parsed reasoning + answer) |
| S3_dmad_dovword | DMAD | **DoV** (DMAD answer instruction) | chat | 1,024 | full |
| S4_dmad_raw | DMAD | DMAD | **raw** | 1,024 | full |
| S5_dmad_512 | DMAD | DMAD | chat | **512** | full |
| S6_dmad_ansonly | DMAD | DMAD | chat | 1,024 | **answers only** |
| S7_dov_chat | DoV | DoV | **chat** | 512 | full |

## Answer reading

- **DoV family:** the robust extractor (`peer-theory-20261119/empties.robust`; blind-validated on Qwen DoV outputs).
- **DMAD family:** DMAD-validated plus the unclosed-tag rule (`collaboration/dmad_rerun/sensitivity_unclosed.unclosed`).
- **Validity audit (pre-registered, about 0.2 GPU-h):**
  - Gemma-3-27B blind extraction (system prompt as in the format-contagion audit) on 100 random outputs per arm, 50 from round 0 and 50 from the final round. Seed 20261122.
  - **Pass:** at least 95% agreement on items where our reader returns a number.
  - **Failure:** an arm that fails is reported with both readings, and its conclusions are flagged.
- **Scoring:** numeric equality after rounding to 1 decimal. No-answer agents are excluded from the vote; ties are split; a problem with no answers scores 0.

## Outcomes (`settings_analyze.py`)

### Per arm

- **P1:** final vote − round-0 vote, in pp.
- **P2:** gated policy G4 − round-0 vote, in pp. G4 keeps the round-0 vote when at least 4 of 5 agents agree at round 0, and uses the final-round vote otherwise.
- **Rescued and broken problems** (wrong → right and right → wrong vote, in expected-vote units), split by round-0 agreement: unanimous (5/5), 4/5, and contested (≤3/5).
- **Individual accuracy** by round; the vote by round.
- **Descriptive:** parse failures by round; G5 (unanimous threshold); vote over round-0 plus final answers.

### Factor contrasts

Differences in P1 between arms, paired by problem:

| Contrast | Arms | Isolates |
|---|---|---|
| C1 | S1 − S2 | the whole protocol (replicates the exploratory contrast) |
| C2 | S3 − S2 | debate wording |
| C3 | S4 − S2 | chat template (DMAD family) |
| C4 | S5 − S2 | max tokens |
| C5 | S6 − S2 | sharing full reasoning vs answers only |
| C6 | S7 − S1 | chat template (DoV family) |

- **Interval:** paired bootstrap over problems, 20,000 draws, seed 20261117, 95% percentile CI.
- **Holm correction** across C2–C6 for the claim "this factor changes P1"; C1 is a replication.
- **The same contrasts for P2** are secondary.

### Readings fixed in advance

- **A factor "drives" the difference** if its Holm-adjusted contrast excludes 0 and its estimate has the same sign as C1.
- **"Debate helps under setting X"** requires the P2 or P1 CI to lie entirely above 0 for that arm.
- **If no factor is significant:** we report that the difference is not attributable to any single factor (it may be an interaction), and we don't claim a mechanism.

## Budget

- About 0.55 GPU-h per arm × 7 arms ≈ 4 GPU-h, plus the audit ≈ 0.2 GPU-h.
- **Hard cap:** 6 GPU-h. If the cap is reached, the remaining arms are dropped in reverse order (S7, then S6) and logged as deviations.
