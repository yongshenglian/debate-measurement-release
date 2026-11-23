# Study L: does study K's framing result hold for other judge models?

- **Approved:** by the user on 2026-11-22.
  - The user proposed "we can test qwen27B model, should not take too long, the third option is to test the most recent Inkling 2 Mini from thinking machine".
  - After the plan below, the user replied "go ahead."
- **Motivation:** study K found its framing effects with a single judge, Gemma-3-27B (`reports/framing-rep-20261122-a/RESULTS.md`):
  - neutral labels raise accuracy (K1);
  - the label "Your earlier solution" makes the judge abandon whichever solution carries it (K2, K4, K5);
  - the judge prefers the second block.
- **This study** repeats study K exactly with two other judges, one at a time:

| Judge | Weights | Serving |
|---|---|---|
| `qwen38`: Qwen3.8-27B (Alibaba), dense | `cyankiwi/Qwen3.8-27B-AWQ-INT4` at `63768c10df38c0395e12ef49edac1bd539eaeeea` (community W4A16 quantisation, like our Gemma) | the study K vLLM image (0.29, digest `082ca6f0…`) |
| `inkling`: Inkling 2 Mini (Thinking Machines), MoE 26.4B total, about 3.4B active, looped | `thinkingmachines/Inkling-2-Mini` at `b85c3cfa615fc413cc9af3379f491b963f3b586a` (official NVFP4/MXFP8 checkpoint) | `vllm/vllm-openai:inkling-2-mini`, pinned by digest in `inkling_image.digest` |

- **Inkling runs only if its smoke run works on our RTX A6000 GPUs.**
  - The vLLM recipe lists only Hopper and Blackwell GPUs.
  - If the smoke run fails, Inkling is reported as not run. We do not change its numerics, engine or harness to make it fit.
- **Frozen:**
  - this protocol;
  - `collaboration/when_debate/{l_gen.py,l_launch.sh,l_analyze.py}` and the study K files they import (`k_gen.py`, `m_reverse.py`, `hard_reverse.py`, `reverse_gen.py`);
  - the reused study K inputs (`data/`, `out/qwen_solve.jsonl`).
  - All are hashed in `frozen.sha256` after the smoke runs and before any full-run output.
  - Inkling's image digest is added to the hashes before its full run.

## Smoke runs (2026-11-22, 40 problems; plumbing only, not analysed for outcomes)

- **Inkling 2 Mini: not run.** vLLM refuses to start: "The quantization method inkling2 is not supported for the current GPU. Minimum capability: 90. Current capability: 86."
  - Our RTX A6000s are Ampere (8.6). The official checkpoint needs Hopper or newer.
  - As fixed above, we do not switch engine or numerics. Study L therefore tests one new judge, Qwen3.8-27B.
- **Qwen3.8-27B: works.**
  - Every call returned a valid answer, and no output contained a thinking block.
  - It is much more verbose than Gemma: mean first-turn output was about 1,400 tokens.
  - 30–40% of judge first turns reached the 2,048-token limit (Gemma in study K: 0.1%). These calls still answer in the second, JSON turn, as the harness does for every model.
  - The limit is kept, so the harness is identical to study K. How often it binds is reported (below).
  - Speed: about 140 s for 40 solves and 16 minutes for 168 judge calls, excluding about 3 minutes of start-up.

## Deviation: the time cap was lifted (2026-11-22 23:37 UTC, before any analysis)

- **Why:** the judge phase was running at about 83 minutes per 40-problem chunk.
  - At the 10,200 s cap it would have finished only about 320 of the 650 problems in D.
  - (Judge s0 against peer s0 gives |D| = 650, split 170 / 150 / 175 / 155 problems across the shards.)
- **Decision:** the user was told this, together with the extra cost of about 12–13 GPU-h, and replied "finish all the runs."
- **What was done:** the `timeout` watchdog in each judge container was paused (SIGSTOP) about 2 h 09 min into the judge phase. Each is resumed once its shard has written all its jobs.
- **Unchanged:** the model, prompts, seeds, chunking and analysis. No judge outcome had been analysed or inspected; only line counts were checked.
- **Expected compute:** about 25–28 GPU-h in total, above the 15.3 GPU-h cap.

## Amendment: systems under every framing (2026-11-23, before any study L outcome was analysed)

- **Why:** an external review re-analysed study K post hoc. It found that two framings beat Gemma's own first answer (s0) over all 1,500 problems:
  - `nswap` (neutral labels, own solution second): +1.0 [+0.3, +1.8] pp;
  - `misattr` ("Your earlier solution" on the peer's solution, own solution second): +0.9 [+0.3, +1.6] pp;
  - neither beats vote(g0, g1, g2).
- The user approved testing this on the new judge before its analysis.
- **State of study L when this was written:** the judge phase was still running. Only line counts of the outputs had been checked.
- **Confirmatory hypotheses:** the system accuracies nswap − judge s0 > 0 and misattr − judge s0 > 0.
  - Same system definition as study K: judge accuracy on D averaged over the three seeds, judge s0 elsewhere, over all 1,500 problems.
  - Two-sided sign-flip test on per-problem differences, Holm over these 2.
  - **"Replicates":** same sign and Holm p < .05.
  - **"Does not replicate":** otherwise. It is called inconclusive unless the interval excludes effects above +1 pp.
- **Secondary:** all eight framings against judge s0 and against vote(j0, j1, j2), Holm over 8 within each family.
- **Cost reporting:**
  - logical calls: 2 × 1,500 solves + |D| reviews, against 3 × 1,500 for the vote;
  - every call is 2 inference requests (free text, then the JSON answer);
  - tokens, time and GPU-h are reported separately.
- **Code:** `l_analyze.py`, block "all framings as systems". Its hash is recorded in `frozen_amend1.sha256`.

## What is the same as study K

- **Problems:** the same 1,500 MMLU-Pro problems, gold labels and question format (`reports/framing-rep-20261122-a/data`).
- **The peer is unchanged.** It is Qwen2.5-7B sample 0 from study K's output, reused read-only and not regenerated.
- **Harness:**
  - multiple-choice task line;
  - free text up to 2,048 tokens, then "State only your final answer as JSON." with an integer schema;
  - T 0.7, top-p 0.95, top-k 20.
- **Seeds** are the same: solve 61000000 + 10·tid + sample; judge 67000000 + 10·tid + replicate.
- **Framings:** the same eight, identical prompts. "own" = the judge's sample 0, "other" = Qwen2.5-7B's sample 0.
- **Replicates:** 3 per problem and framing.

## What differs

- **The judge model** and its own solutions (samples 0, 1 and 2). The disagreement set D is judge s0 ≠ peer s0, both valid, so it differs by judge.
- **Thinking is turned off** through each model's chat template, so the judge is tested like Gemma (which has no thinking mode):
  - Qwen3.8: `enable_thinking: false`;
  - Inkling: `reasoning_effort: 0.0` ("Thinking effort level: 0").
  - The smoke run checks that outputs contain no thinking blocks. The rate of first turns cut off at 2,048 tokens is reported.
- **Sampling:** the harness's T 0.7 / top-p 0.95 / top-k 20 is kept for comparability, not each vendor's recommended setting (Inkling recommends T 1.0). This is a stated limitation.

## Outcomes (`l_analyze.py` = `k_analyze.py` with judge/peer renamed)

- **Per judge:** the same five primary contrasts, Holm over five within each judge, with the same tests and bootstrap.
  - K1: neutral − owner, accuracy on D.
  - K2: yours − neutral, keeps own.
  - K3: another − neutral, takes other's answer.
  - K4: yoursol − yours, keeps own.
  - K5: misattr − nswap, takes other's answer.
- **The same secondary outcomes:** every framing, position, rates and systems against vote(j0, j1, j2), plus tokens.
- **Added after the smoke runs, before any full-run output:** the share of first turns that end at the token limit, by framing.
  - If framings differ in how often the judge is cut off, that difference is reported next to K1–K5 as a possible pathway.

### Readings fixed in advance (per judge)

- **"Study K's result generalises to this judge":** the contrast has the same sign as in study K and Holm p < .05.
  - Applies to K1, K2 and K5, where study K's result was significant and signed.
- **"Does not generalise":** the opposite sign with Holm p < .05, or an interval that excludes effects beyond ±3 pp in study K's direction.
- **Otherwise inconclusive.**
- **Across judges:** a result is called general across the judges tested only if it generalises to every judge that ran.

## Budget

- **Per judge:**
  - solves: 1,500 × 3 = 4,500 calls;
  - judge calls: |D| × 24, about 15,000.
- **Estimate** from study K: about 10–12 GPU-h per judge. Inkling activates fewer parameters and may be faster.
- **Cap by construction** in `l_launch.sh`: 3 × 4,800 s + 4 × 10,200 s = 15.3 GPU-h.
  - The solve timeout was raised from 3,000 s after the smoke runs, because Qwen3.8 writes about twice as much as Gemma.
  - The judge's own sample 0 defines D, so a cut-off solve would shrink D.
  - Output is written in chunks of whole problems.
  - After a timeout, only problems with all 24 judge calls enter the analysis, and the shortfall is reported.
- **Order:**
  1. Qwen3.8 first.
  2. Inkling after its smoke run, once Qwen3.8 is done.
  - Both start only after the human-audit blind read (stage A) has released the GPUs.
