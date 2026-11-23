# Study D: Du et al. GSM8K debate protocol with open models (pre-registration)

- **Approved:** by the user on 2026-11-20 ("approve both"); covers plan D.
- **Written:** before any output.

## Target

Du et al. (ICML 2024): GSM8K, 3 agents, 2 rounds of debate. Reported: multi-agent debate 85.0 vs single agent 77.0 (GPT-3.5).

We cannot test their numbers, because gpt-3.5-turbo-0301 is retired. We test whether **their evaluation code** gives the same verdict as valid extraction when the same protocol is run on open models.

## Code

- **Repository:** composable-models/llm_multiagent_debate at commit 9846749.
- **Prompts:** from `gsm/gen_gsm.py`. `construct_message` is imported unchanged; the initial prompt is copied verbatim.
- **Scoring:** `gsm/eval_gsm.py` (`parse_answer`, `solve_math_problems`, `most_frequent`, `compute_accuracy`), imported.
- **Runner:** `collaboration/du_rerun/gen_du.py`.

## Settings

| Item | Setting |
|---|---|
| Models | Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct (local, checksummed) |
| Agents | 3 |
| Rounds | 3 generations (round 0 + 2 debate rounds), matching the paper's "2 rounds" and the eval file name `gsm_debate_3_3.json` |
| Questions | `random.seed(0); random.shuffle(test.jsonl)`, then the first 300. The first 100 are the authors' exact subset and are reported separately. |

## Deviations

- **Sampling:** offline vLLM 0.29 chat with each model's chat template, instead of the OpenAI API.
  - T 1.0 and top-p 1.0, the API defaults.
  - max_tokens 2048, where the API had none.
  - A distinct seed per call.
- **Batching:** calls are batched per (round, agent). Contexts are identical to the sequential loop.
- **Question count:** 300 instead of 100, for precision.

## Scorers

| Scorer | Method |
|---|---|
| Authors' (agent level) | last `{[0-9.,$]*}`, else the last number anywhere in the text |
| Authors' (vote level) | `most_frequent`, then `compute_accuracy` (returns correct if the voted answer is None) |
| Validated | the last balanced `\boxed{…}`, cleaned (`\text`, `$`, commas) to a single number; anything else is no answer. A blind Gemma-3-27B audit of about 300 stratified outputs adjudicates (study E). |

## Primary contrast (per model, both scorers)

- **Contrast:** vote accuracy at round 2 − round 0.
- **Interval:** paired bootstrap over questions, 20,000 draws, seed 20261117, 95% CI.
- **Verdict changes if:** the sign flips, or significance is gained or lost.

## Secondary

- Per round, the share of agent outputs where the authors' parser:
  - uses the last-number fallback;
  - misreads (authors' answer ≠ validated answer, both present);
  - returns None scored as correct.
- Single-agent accuracy, from round-0 agent 0.

## Budget

About 2 × 2,700 calls, under 1 GPU-h.
