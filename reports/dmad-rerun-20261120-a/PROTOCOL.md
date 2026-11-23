# Study B: DMAD vanilla-MAD baseline re-run (pre-registration)

- **Approved:** by the user on 2026-11-20 ("approve both"); covers plan B in `reports/parser-audit-20261119/PLAN.md`.
- **Written:** before any full-run output. A 2-question smoke test per model checks only the mechanics, and its outputs are discarded.

## Target

DMAD (Zhu et al., arXiv 2601.19921 v3), Table 1, GSM8K:

| Model | Single model | Majority vote | Debate 5×5 (vanilla MAD) |
|---|---|---|---|
| Qwen2.5-7B | 0.908 | 0.908 | 0.847 |
| Llama-3.1-8B | 0.797 | 0.870 | 0.850 |

The paper's reading is that vanilla MAD underperforms the majority vote. It attributes this to low initial diversity and to the inability to use confidence.

## Code

- **Repository:** SpaceHunterInf/DMAD at commit b64418a. The debate loop is `inference/vllm_debate.run_parallel_debates`, imported and unmodified, with DMAD's prompts, parser, cleaning and majority vote.
- **Our wrapper:** `collaboration/dmad_rerun/run_arm.py`. It only builds the `SamplingParams` and the LLM exactly as `vllm_debate.main` does, and adds the arm manipulations listed below.
- **Data:** GSM8K test, via DMAD's `load_gsm8k(split="test", data_size=300, seed=0)` (its README example), saved as `gsm8k_test300.csv`.
- **Configs:** DMAD's own (`configs/debate_{qwen,llama}.yaml`).
  - 5 agents, 5 rounds, T 1.0, top-p 0.9, top-k −1, 1,024 max tokens, gpu_mem 0.85.
  - dtype: bfloat16 for Qwen, float16 for Llama.
- **Seed:** `--seed 42`, as in the README, passed to `SamplingParams`.
- **Prompts:** chat template applied with `tokenize=False`, without `add_generation_prompt`, as in DMAD's code.

## Known deviations

| Item | DMAD | Ours |
|---|---|---|
| vLLM version | v0.11.0 | 0.29.0 (pinned image) |
| Weights | HF hub | local copies, same weights |
| Tokenizer for Llama | slow | fast, if the slow one is unavailable (logged per arm) |
| Packages | — | pandas 2.2.3 added via `envs/dmad-extra` |

## Arms

Both models run every arm, on 300 questions.

| Arm | Manipulation | Purpose |
|---|---|---|
| faithful | `use_confidence=False`; `SamplingParams` exactly as DMAD (shared seed 42, stop `</answer>`) | the paper's vanilla MAD |
| nostop | as faithful, without the stop string | stop-token ablation: does stopping at the first `</answer>` record quoted peer answers? |
| agentseed | as faithful, but each (round, agent) gets its own seed 42 + 1000·round + agent | the diversity check: with one shared seed, five agents given an identical round-0 prompt may sample identical text |
| conf | `use_confidence=True` (untrained confidence prompt), otherwise as DMAD | the confidence-prompt contrast |

## Scoring

Every agent output is scored twice:
1. **DMAD's parser and matcher:** `parse_response`, then `clean_answer` (the last number inside the last `<answer>`), then `gsm8k_match`. The majority vote uses `get_majority_answer`.
2. **Validated extractor:**
   - Take the agent's own committed final answer: the last `<answer>…</answer>` in the full completion that is not inside a quoted peer block.
   - Then the same numeric cleaning, with a fallback to the robust GSM8K extractor (`\boxed{}`, "final answer").
   - A blind Gemma-3-27B audit (study E) of about 400 stratified outputs adjudicates.

## Primary contrasts

Each is computed for each model, under both scorers. Paired bootstrap over questions: 20,000 draws, seed 20261117, 95% CI.

| ID | Contrast |
|---|---|
| P1 | vanilla MAD (vote at the last round) − majority vote at round 0, in the faithful arm (the paper's headline contrast) |
| P2 | round-0 answer diversity (mean unique answers per question; % of questions with five identical round-0 texts), faithful vs agentseed |
| P3 | P1 in agentseed − P1 in faithful (does fixing the seed change the MAD-vs-vote verdict?) |
| P4 | faithful vs nostop, the last-round vote; and the share of outputs where stopping at the first `</answer>` changes the recorded answer |

## Secondary

- confidence − faithful, at the last round;
- per-round parse failure (PARSE_ERROR) rate;
- PARSE_ERROR strings shown to peers;
- breaking-pattern frequencies by round.

## Conclusion rule

DMAD's "MAD < vote" conclusion **changes** if, under valid scoring and/or with independent seeds, P1's sign flips or its significance at 95% is gained or lost.

## Budget

- 8 arms × 7,500 generations.
- Estimate: about 3 GPU-h, on GPUs 0 and 2. Each container has a 3 h timeout.

## Amendment 1 (2026-11-20, written before any `fixed`-arm output)

**What was seen first** (in the faithful Qwen arm and the agentseed Llama arm):
- Round-0 parse errors are 18% (Qwen) and 33% (Llama), and fall to 4–8% by the last round.
- The cause is DMAD's `apply_chat_template(messages, tokenize=False)` without `add_generation_prompt=True`. The model must open its own assistant turn, and it starts with junk (".imgur.com/q123456.png\n.user", "reasoning\n") and often drops the `<answer>` tags.
- Debate rounds show peers' answers wrapped in `<answer>…</answer>`, which repairs the format.
- The faithful Qwen arm also has 52% of questions with five identical round-0 texts (shared seed).

**New arm `fixed`:**
- agentseed, plus the chat template with `add_generation_prompt=True`, applied through a tokenizer proxy. DMAD's code is otherwise unchanged.
- Both models, 300 questions.

**Purpose:** DMAD's vanilla-MAD-vs-majority-vote verdict in a pipeline without the two defects.

**Contrast and analysis:** P1 (last-round vote − round-0 vote) in `fixed`, under both scorers, with the same analysis script.

**Status:** reported as confirmatory for the `fixed` arm, and labelled as an amendment.

**Budget:** about 0.8 GPU-h, within the approved 7.5.

## Deviation 1 (2026-11-20): `conf` arms dropped for budget

- **Why:** Llama arms take about 50 min and Qwen arms about 35 min, against the planned ~20. Ten arms would exceed the approved GPU budget.
- **Decision:** the `conf` arms (a secondary contrast only) are dropped before any `conf` output was analysed.
  - The Llama `conf` run was stopped 2 minutes in; its log is `out/llama-b_conf.ABORTED.log`.
  - The Qwen `conf` arm is stopped before it starts.
- **Kept:** faithful, nostop, agentseed and fixed, for both models.

## Deviation 2 (2026-11-20, found after the Llama `fixed` arm): Llama chat template

- **The mirror's template:** the Llama weights mirror (NousResearch@d10aef7) ships an older Llama-3 `chat_template`. It always appends the assistant header, ignores `add_generation_prompt`, and has no 3.1 system/date block.
- **The official template** (meta-llama/Llama-3.1-8B-Instruct, blob db88166e) appends the header only when `add_generation_prompt=True`.
- **Consequence:**
  - In our Llama arms, DMAD's missing-generation-prompt defect never occurs. `fixed` therefore equals `agentseed` for Llama, as observed.
  - **The Llama `faithful` and `nostop` arms are not faithful on this point.**
  - Llama conclusions about DMAD's pipeline are withheld pending a re-run with the official template.
- **Prepared:** `models/debate-or-vote/llama3.1-8b-instruct-official-template`. It holds hard links to the verified weights plus the official `tokenizer_config.json` (from RedHatAI/Llama-3.1-8B-Instruct, git blob sha1 verified identical to Meta's).
- **Re-run:** faithful + fixed Llama arms, about 1.7 GPU-h. Not launched: it needs user approval (over budget).
- **The Qwen arms are unaffected.**

## Amendment 2 (2026-11-20, approved by the user, before launch): Llama re-run with the official template

- **Arms:** `faithful` and `fixed` on Llama-3.1-8B, using `models/debate-or-vote/llama3.1-8b-instruct-official-template`. Same weights, official Meta `chat_template`.
- **Everything else as the earlier Llama arms:** float16, the same 300 problems, the same `run_arm.py`, image, and seeds.
- **Outputs:** `out/llama-t_faithful.json` and `out/llama-t_fixed.json`.
- **Primary (pre-specified):**
  1. vote at round 5 − majority vote at round 0, under DMAD's own `gsm8k_match` and under validated numeric scoring, for each arm;
  2. the faithful − fixed difference, which measures the effect of the two defects on Llama.
- **Prediction (from the Qwen arms):** faithful will be inflated relative to fixed, because with no assistant header the round-0 agents fail more often. If the two arms agree, that also counts as a finding: the defect would then be model-dependent.
- **The earlier Llama `faithful`/`nostop` arms** are kept, labelled "mirror template", and are not used for Llama conclusions about DMAD's pipeline.
