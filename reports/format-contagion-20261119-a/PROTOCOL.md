# Format contagion in multi-agent debate: controlled test and blind extraction audit

Version: `format-contagion-v1`, 2026-11-19. Frozen before generation. The user
authorized cheap follow-ups ("keep doing").

## A. Controlled twin test: does a peer's broken answer format cause the reader to break format?

### Setup

- **Stimuli:** round-0 responses from our faithful Debate-or-Vote GSM8K dense
  replication (`reports/debate-vote-gsm8k-20261117-a`).
- **Eligible problems:** all five responses parse with the published extractor
  and end with `{final answer: X}`; 130 problems. Another 14 problems failed the
  manipulation check (a broken peer stayed parseable) and are excluded before
  generation.
- **Reader:** Agent 1, with its own round-0 response.
- **Peers:** Agents 2–5, with the exact Debate-or-Vote dense round-1 prompt.
- **Decoding:** no chat template, T 1.0, top-p 0.9, 512 new tokens, as in the
  original.
- **Model:** Qwen2.5-7B-Instruct, served offline with vLLM.

### Conditions

Only the peers' final-answer markup changes; the content and stated answers are
identical across conditions.

| Condition | Peer change |
|---|---|
| clean | none |
| esc1 / esc4 | 1 or all 4 peers have the last `{final answer: X}` rewritten as `\{final answer: X\}` (escaped braces, +2 characters) |
| loop1 / loop4 | the same, then 20 repeats of ` \{final answer: X\} \} \)` (the loop pattern seen in the debate logs) |

- **Which peer:** for the single-peer conditions, the broken peer is chosen by
  hash per problem.
- **Manipulation check:** the published extractor fails on every broken peer,
  while the robust extractor reads the same answer.
- **Samples:** 8 seeds per problem, shared across conditions; 5,200 calls.

### Outcomes, per reader output

1. **Broken format:** the published extractor returns nothing.
2. **Published-extractor correctness:** what the paper would have scored.
3. **Robust-extractor correctness:** what the reader actually answered.

### Primary contrasts

Each is a problem-level paired difference over the 8 seeds: the exact
sign-flip test, a bootstrap CI (20,000 draws), and Holm over three.

| ID | Contrast | Outcome |
|---|---|---|
| F1 | esc1 − clean | broken format |
| F2 | loop4 − clean | broken format |
| F3 | loop4 − clean | robust correctness |

F3 is predicted to be within ±2 pp: format breaks while reasoning does not.

**Secondary:** every condition against clean on all three outcomes;
dose-response (1 vs 4 broken peers); finish_reason = length rates.

## B. Blind extraction audit

- **Extractor:** Gemma-3-27B (W4A16), a different model family, with greedy
  decoding.
- **Input:** each debate response with no label, question or category, asked
  for the final numeric answer the model committed to.
- **Sample:** 400 responses from the four replication arms, stratified as
  - 240 where the published extractor fails and the robust one recovers;
  - 60 where they agree;
  - 60 where they disagree;
  - 40 where both fail.
- **Measures:**
  - agreement of Gemma with the robust and with the published extractor, per
    stratum;
  - on recovered cases, the accuracy implied by Gemma's reading against the
    robust reading.
- **Pass criterion for using the robust extractor:** at least 95% agreement with
  Gemma on recovered cases.

## Budget

Two GPUs, offline batched, expected under 1 GPU-hour in total. No retries.

## Deviation log

- **2026-11-19, audit B only.** The Gemma extractor container failed at vLLM
  engine start-up: multiprocessing spawn requires a `__main__` guard. This
  happened before any output. The runner was copied to `fc_generate_v1b.py`
  with the code wrapped in `main()` and the guard added; the generation logic
  and parameters are unchanged. The audit was relaunched. Run A (Qwen) used the
  original runner and was unaffected.

## Amendment v2: generality across readers and prompt formatting (2026-11-19, before any v2 output)

**Question:** is format contagion specific to Qwen, or to Debate-or-Vote's choice
of feeding raw prompts without a chat template?

**Stimuli:** the same 130 problems × 5 peer conditions; seeds 0–3 of the frozen
8 (2,600 calls per arm).

**New arms:**
- Qwen2.5-7B with its chat template;
- Mistral-Small-3.1-24B (W4A16) with its chat template;
- Gemma-3-27B (W4A16) with its chat template.

Each arm sends the identical debate message as a single user turn, with T 1.0,
top-p 0.9 and 512 new tokens (`fc_generate_v2.py`, mode `chatgen`).

**Outcomes:** the same three. The robust extractor is unchanged.

**Primary (per arm, Holm over 3 arms × 2):**
- loop4 − clean on broken format, predicted > 0;
- loop4 − clean on robust accuracy, predicted within ±2 pp.

**Secondary:** esc1, esc4 and loop1 against clean; comparison with the raw-prompt
Qwen arm on seeds 0–3.

**Interpretation:** if contagion persists under chat templates and across
families, it is a general multi-agent failure mode. If it vanishes, it is tied
to raw prompting, which is still the protocol Debate-or-Vote used.
- **2026-11-19, v2 Mistral arm only.** vLLM start-up failed while profiling
  Mistral's image processor (PixtralProcessor), before any output. The arm was
  relaunched with `fc_generate_v2b.py`, identical except
  `language_model_only=True`, as in our earlier Mistral qualification. This is a
  text-only serving flag; decoding is unchanged.
- **2026-11-19, v2 Mistral arm only (second fix).** Chat rendering failed
  because Mistral's tokenizer defines no chat template; the template ships in
  `chat_template.json`. `fc_generate_v2c.py` passes that file's template to
  `llm.chat` and is otherwise identical to v2b. No output had been produced.
- **2026-11-19, v2 Mistral arm only (third fix).** Passing the template to
  vLLM's chat path failed again (`MistralCommonBackend` has no
  `get_chat_template`). `fc_generate_v2d.py` instead loads Mistral with
  `tokenizer_mode="hf"` and renders the identical single user message with the
  HF tokenizer's `apply_chat_template`, using `chat_template.json`. This is the
  loader configuration of our earlier Mistral qualification. Decoding is
  unchanged, and no output had been produced.
