# Study C: Wynn et al. "Talk Isn't Always Cheap" open-model configurations (pre-registration)

- **Approved:** by the user on 2026-11-20 ("approve both"); covers plan C.
- **Written:** before any output. A 1-question smoke test checks only the mechanics, and its outputs are discarded.

## Target

Wynn, Satija and Hadfield (arXiv 2509.05396 v2), Table 1, GSM8K: majority vote w/o debate → after debate, mean ± SE over 5 seeds × 100 questions.

| Configuration | w/o debate | After debate | Change |
|---|---|---|---|
| 3×Llama | 87.6 | 84.2 | −3.4 |
| 2L+1M | 82.6 | 75.8 | −6.8 |
| 1L+2M | 61.0 | 64.8 | +3.8 |
| 3×Mistral | 43.6 | 46.4 | +2.8 |

Single agent: Llama 76.4, Mistral 34.2.

## Code

- **Repository:** TheNormativityLab/talk-aint-cheap at commit 47ffdb5.
- **Environment:** `envs/wynn-py311`, built from the authors' pinned `requirements.txt`.
- **Only change:** the two LLM configs that `src/conf/basic_debate.yaml` references but the repository omits, `vec_llama_3_1_8B.yaml` and `vec_mistral_7B.yaml`. They are copies of `llama_3_1_8B.yaml` and `mistral_7B.yaml` (top-p 0.9, max_tokens 2048, no temperature), with the Together endpoint replaced by a local vLLM OpenAI server (`hosted_vllm/…`).
- **Everything else is the authors' code and config:** prompts, `summarize: true`, 2 rounds, the per-seed question shuffle, parsing and scoring.

## Settings

| Item | Setting |
|---|---|
| Models | Llama-3.1-8B-Instruct (`models/debate-or-vote/llama3.1-8b-instruct-nous-d10aef7`), Mistral-7B-Instruct-v0.2 (`models/wynn/mistral-7b-instruct-v0.2-63a8b08`, safetensors SHA-256 equal to HF) |
| Serving | vLLM 0.29 OpenAI server, pinned image; the model's own generation_config defaults where the request sets no temperature |
| Configurations | 3L, 2L+1M, 1L+2M, 3M |
| Seeds | 0–4 |
| Questions | 100 per seed |
| Concurrency | `max_concurrent_calls` raised to 60, a throughput setting only |
| Flags | `wandb.mode=disabled`, `cost_check=False` |

## Deviations

- **Serving:** Together API → local vLLM.
- **Temperature:** the paper says only "default temperature", so the server default is used and logged.

## Scorers

| Scorer | Method |
|---|---|
| Authors' | `src/analysis/evaluation.py`: `parse_gsm8k_answer` (last `{[0-9.,$]*}`) with *string* equality against the last number of the gold solution; `most_frequent` over parsed answers |
| Validated | the last balanced `\boxed{…}`, cleaned to one number, compared numerically. The blind Gemma audit adjudicates (study E). |

## Primary contrast (per configuration, both scorers)

- **Contrast:** after-debate vote − w/o-debate vote (round 2 − round 0).
- **Pooling:** over the 5 seeds.
- **Interval:** bootstrap resampling questions within seed, 20,000 draws, seed 20261117, 95% CI.
- **Verdict changes if:** the sign flips, or significance is gained or lost.

## Secondary

- Per-agent right→wrong and wrong→right flip rates per round, under both scorers.
- Parse-failure and misread rates per round.
- The share of string-equality failures on numerically equal answers (e.g. "18.0" vs "18").

## Budget

About 4 × 5 × 100 × (9 answer calls + 6 summary calls) ≈ 30,000 calls, about 2 GPU-h on two GPUs.

## Run log

- **2026-11-20, smoke test** (1 question, 3L): the pipeline runs end to end and the output was discarded.
- **Server defaults (logged):**
  - Llama: vLLM applies the model's `generation_config.json` (temperature 0.6, top-p 0.9; the request's top-p 0.9 is the same).
  - Mistral-7B-v0.2 has no sampling defaults in its generation_config, so vLLM's default temperature 1.0 applies.
- **Validated extractor:** Llama sometimes writes `boxed{16}` without the backslash. The validated extractor accepts an optional backslash. This was specified before any full-run output.
- **Scheduling:** 3L starts first, on the Llama server only (GPU 3). The mixed configurations and 3M start when a second GPU frees.
- **Deviation (found 2026-11-20 during the run):** the Llama server uses the weights mirror's older Llama-3 chat template, not the official 3.1 one, which differs in its system/date block. It is kept for the whole study for consistency and reported as a deviation.
