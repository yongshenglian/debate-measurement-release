# Does a strong model get worse after reading a weaker model's work? (study R)

- **Requested:** by the user on 2026-11-20: "a good llm benefit from an excellent llm. if we do the reverse, will an excellent llm become worse after reading a good llm work?"
- **Frozen:** this protocol, `collaboration/when_debate/reverse_gen.py` and `reverse_analyze.py` are hashed before any full-run output.

## Background

- **Earlier peer studies** (`peer-discrimination`, `peer-persuasion`, `peer-warning`, `self-review`) used Qwen2.5-7B as the reader of Gemma-3-27B-derived peer working, with edited correct and wrong twins.
  - A wrong peer's working lowered Qwen's accuracy.
  - A correct peer's working raised it.
- **The reverse has not been tested:** Gemma-3-27B reading Qwen2.5-7B's working.

## Design

- **Models:**
  - Qwen2.5-7B-Instruct BF16 ("good").
  - Gemma-3-27B-it W4A16 ("excellent"; `models/heterogeneous`).
  - Pinned vLLM 0.29 image.
- **Tasks:** the full GSM8K test set, 1,319 problems (`tasks.jsonl`).
  - Labels are kept in `labels.json` and never enter any prompt.
- **Output contract (two turns per call):**
  1. a free-text solution (≤ 1,024 tokens);
  2. in the same conversation, "State only your final answer as JSON.", with structured output `{"answer": integer}` (guidance backend, no free whitespace, ≤ 16 tokens).
  - Every final answer is the model's own statement. No regex is involved.
  - An invalid output is scored as wrong, and its count is reported.
  - **Change from the first draft, decided on smoke data before any full run:** a one-turn JSON contract (`rationale` + `answer`) made Qwen stop its reasoning early and guess (47.5% correct on 20 smoke tasks, plus whitespace loops). With the two-turn design, Qwen scored 86% and Gemma 95% on 40 smoke tasks, with 0 invalid outputs.
- **Decoding:** T 0.7, top-p 0.95, top-k 20, chat template. Peers are shown their free-text solution verbatim.

### Stage 1 (solve)

- Each model writes two independent solutions per problem.
- Sample 0 is the model's own first answer; sample 1 is used as a same-model peer.

### Stage 2 (review)

Each model reviews its own sample 0 under three conditions, with one seed per task shared across conditions:

| Condition | What the reviewer sees |
|---|---|
| self | its own earlier solution; "Review your earlier solution and give your final answer." |
| peer_qwen | own solution + a Qwen solution labelled only "Another solver's solution" |
| peer_gemma | own solution + a Gemma solution, labelled the same way |

- **Peers are anonymous.** No strength, authority or majority labels (the user's design rule from the peer studies).
- **Which sample is the peer:**
  - When the peer is the same model, it is that model's sample 1: an independent draw, not the reviewer's own text.
  - When the peer is the other model, it is that model's sample 0.
- **Peers are natural, not edited.** Their correctness is whatever the peer model produced.

## Outcomes (`reverse_analyze.py`)

### Primary (the user's question)

- **R1:** Gemma's accuracy with peer_qwen − Gemma's accuracy with self.
  - Paired bootstrap over problems, 20,000 draws, seed 20261117, 95% CI.
  - **Reading:** a CI entirely below 0 means the excellent model gets worse after reading the good model's work. A CI including 0 with both bounds within ±1 pp means no meaningful harm. Otherwise the result is inconclusive.

### Mirror (the forward direction, under identical prompts)

- **R2:** Qwen's accuracy with peer_gemma − Qwen's accuracy with self.

### Mechanism (descriptive, with bootstrap CIs)

- **Corruption:** P(right → wrong | own first answer right), for each reviewer × condition. This is further split by whether the peer was right or wrong.
- **Rescue:** P(wrong → right | own first answer wrong), for each reviewer × condition.
- **Following a wrong peer:** P(final = the peer's wrong answer | own right, peer wrong) — how often a reviewer adopts a wrong peer answer.
  - The key comparison is strong reviewer + weak wrong peer vs weak reviewer + strong wrong peer.
- **Same-model controls:** Gemma reading another Gemma, and Qwen reading another Qwen. These separate "reading any second opinion" from "reading a weaker one".

### Caveats fixed in advance

- **Prompt length:** prompts are not token-matched across conditions, unlike the earlier peer studies. Self prompts are shorter.
- **Natural peers:** peer correctness is not randomized, so it is confounded with task difficulty. The split analyses condition on the reviewer's own first answer, which partly controls for this.

## Budget

- **Calls:** about 5,300 solve calls and 7,900 review calls.
- **Estimate:** about 1.5 GPU-h. **Hard cap:** 3 GPU-h.
