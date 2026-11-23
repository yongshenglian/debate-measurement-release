# Measurement validity in multi-agent debate: parser audit (step 1) and re-run plan (steps 2–3)

- **Date:** 2026-11-19.
- **Step 1:** done, no GPU.
- **Steps 2–3:** pre-registration draft for user approval.

## Step 1 results: how fragile is each codebase's answer parser?

**Parsers:** six transcribed from released code (`collaboration/parser_audit/parsers.py`), each
tested on its OWN requested answer format.

**Test cases:** 300 real Qwen debate reasoning texts × 12 variations (`suite.py`, output
`suite.json`). Each cell is the % of answers read correctly; the parenthesis gives the %
unreadable / % misread.

| Variation | Debate-or-Vote `{final answer: X}` | Du et al. `\boxed{X}` | Wynn et al. `\boxed{X}` | DMAD `<answer>` (generation stops at the first `</answer>`) | ReConcile JSON | When Debate Helps |
|---|---|---|---|---|---|---|
| canonical | 99.7 | 99.7 | 99.7 | 100 | 100 | 100 |
| escaped delimiters | **0** (100/0) | 60 (0/40) | **1** (59/40) | **0** (100/0) | — | 100 |
| `\text{X}` / `$X$` inside | **0** | 99.7 | 99.7 | 100 | — | 100 |
| `$X` | **0** | 99.7 | 99.7 | 100 | 100 | 100 |
| thousands separator `1,250` | **0** | 99.7 | 99.7 | 100 | 100 | 100 |
| `X.00` | 99.7 | 99.7 | 99.7 | 100 | 100 | 100 |
| newline inside delimiters | **0.7** | 60 | **1** | 100 | 100 | 100 |
| LaTeX braces after the answer | **0** | **2.7** (0/97) | **2.7** | 100 | **0** | 100 |
| a peer's answer quoted *after* own answer | **0** (misread) | **0** | **0** | 100 | **0** | **0** |
| a peer's answer quoted *before* own answer | 100 | 99.7 | 99.7 | **0** (records the peer's) | **0** | 100 |
| repetition loop, cut at token limit | 100 | 99.7 | 99.7 | 100 | **0** | 100 |
| format ignored ("The final answer is X.") | 0.7 | 60 | 1 | 0 | 0 | 100 |

**How each codebase scores a parse failure:**

| Codebase | A failure is scored as |
|---|---|
| Debate-or-Vote, Wynn et al., DMAD, When Debate Helps | wrong |
| ReConcile | the answer "0" |
| **Du et al.** | **correct** (`compute_accuracy`: `if pred_answer is None: return 1`) |

**How often the breaking patterns occur in real debates** (`natural.py`, output
`natural.json`; Qwen2.5-7B, Debate-or-Vote protocol, % of agent outputs):

| Pattern | Dense round 0 | Dense round 5 | Sparse round 0 | Sparse round 5 |
|---|---|---|---|---|
| escaped `\{final answer` | 4.6 | **19.5** | 4.3 | **17.8** |
| braces after the answer | 3.5 | **12.7** | 2.8 | **9.7** |
| answer loop 5+ times | 1.1 | **33.3** | 1.3 | **28.7** |
| two different stated answers | 0.3 | **7.3** | 0.3 | **8.0** |
| no answer marker | 4.1 | 2.5 | 5.6 | 3.1 |

**Reading:**
- Every parser except When Debate Helps' has a failure mode that real debate outputs trigger.
  These modes **grow with rounds**: agents copy each other's markup, loop, and restate
  others' answers. So parser error is not noise; it is correlated with the variable the
  papers study (rounds of interaction).
- Which way the error pushes depends on the codebase:
  - Debate-or-Vote turns it into apparent decline (confirmed).
  - Last-match parsers (Du, Wynn) misread "peer quoted after own answer" as the peer's
    answer. That looks like conformity.
  - DMAD's stop-at-first-`</answer>` records a quoted peer answer as the agent's own,
    which also looks like conformity.
  - Du et al. score failures as correct, which inflates accuracy whenever extraction fails.

## Step 2: which papers to re-run

| Paper | Influence | At-risk conclusion | Runnable here? | Decision |
|---|---|---|---|---|
| Debate-or-Vote (NeurIPS 2025) | high | "debate degrades over rounds" | yes (Qwen done; Llama-3.1-8B running) | **anchor** |
| DMAD (2601.19921) | recent; builds on Debate-or-Vote | vanilla MAD vs majority vote vs their confidence protocol | yes: open models (Qwen2.5-7B, Llama-3.1-8B), vLLM | **re-run baselines** (their trained models are not runnable) |
| Wynn et al., Talk Isn't Always Cheap (ICML 2025 MAS workshop) | medium | debate can degrade; agents conform | partly: Llama-3.1-8B and Mistral-7B-v0.2 (ungated); GPT-4o-mini is API-only | **re-run the open-model configurations** |
| Du et al. (ICML 2024) | foundational, very high | "debate improves reasoning" (GPT-3.5) | protocol yes; original model no (API, deprecated) | **protocol re-run with open models** (tests the parser, not their numbers) |
| ReConcile | high | — | API-only models | skip |
| When Debate Helps | new | — | parser robust | skip, used as positive control |

## Step 3: pre-registered plan

### Common design

- **Code:** each paper's own released code at a pinned commit, with that paper's
  prompts, decoding, agent counts and rounds.
- **Scoring:** every output is scored twice, with the paper's own parser and with our
  validated extractor. That extractor is the robust extractor, extended per format, and
  is validated per study by a blind Gemma-3-27B audit (≥95% agreement on recovered cases,
  as before).
- **Primary per paper:** the paper's headline contrast (below), estimated under both
  scorers, with paired task-level bootstrap CIs.
- **"Conclusion changes":** the sign flips, or significance at the 95% level is gained or
  lost.
- **Secondary:**
  - per-round parse-failure and misread rates;
  - frequency of each breaking pattern by round;
  - for Du et al., how often a failure is scored as correct.

### Studies

| ID | Paper | Configuration | Headline contrast | Mechanistic check | GPU-h |
|---|---|---|---|---|---|
| A | Debate-or-Vote | Llama-3.1-8B, GSM8K dense, 300 problems, 5 agents × 5 rounds (running, approved) | vote at round 5 − round 0 | — | about 8 (approved) |
| B | DMAD | vanilla MAD, with and without the confidence prompt, on GSM8K; Qwen2.5-7B and Llama-3.1-8B; their 5 agents × 5 rounds, T 1.0, 1,024 tokens, vLLM | MAD round 5 − majority vote; confidence − vanilla | identical seeds with and without the stop token, so stop-at-first-`</answer>` artifacts are measured directly | about 3 |
| C | Wynn et al. | GSM8K, 100 problems as in the paper; 3×Llama, 3×Mistral-7B, 2 Llama + 1 Mistral; 2 rounds; their prompts through a local vLLM server | round 2 − round 0 accuracy, and the right→wrong flip rates they report | — | about 2, plus a 14 GB download |
| D | Du et al. | their GSM8K protocol (3 agents, 2 rounds), Qwen2.5-7B and Llama-3.1-8B | round 2 − round 0 | how much "failure = correct" inflates accuracy | about 1 |
| E | Blind extraction audits | Gemma-3-27B, about 400 stratified outputs per study | — | — | about 1.5 |

- **New GPU beyond the approved Llama run:** about 7.5 GPU-h.
- **Wall time:** about 1–2 days, mostly engineering (adapting each codebase to local
  serving without changing prompts or parsing).

### What would make this high-impact

- **If** two or more of B–D show a headline conclusion that changes under valid
  measurement, the paper becomes "A large share of multi-agent debate's reported dynamics
  are measurement artifacts", with a mechanism (format contagion) and a validated
  protocol.
- **If** B–D are robust, the result is still publishable but narrower: the
  Debate-or-Vote case, plus a parser-validity audit and a recommended protocol.

### Risks

- Adapting API code (Wynn) to local serving may change behaviour. Mitigation: use their
  prompts and parameters verbatim, and record the change.
- DMAD's headline method needs their trained adapters. If released, we add them;
  otherwise we audit only the baselines.
