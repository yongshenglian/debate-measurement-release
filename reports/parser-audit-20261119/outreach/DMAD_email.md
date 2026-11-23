# Draft: email to the DMAD authors (2601.19921). NOT SENT: the user sends it from their own account.

**Subject:** DMAD (2601.19921): three issues in the released inference code that change the MAD vs majority-vote comparison

Dear authors,

We are auditing measurement pipelines in multi-agent debate papers, and we re-ran the vanilla-MAD baseline from your released code (SpaceHunterInf/DMAD @b64418a).
- **Setup:** GSM8K, 300 test problems, Qwen2.5-7B-Instruct, 5 agents × 5 rounds, T 1.0, your `inference/vllm_debate.py` unmodified.
- **What we found:** three pipeline details that change the headline comparison.

1. **Shared seed.** `vllm_debate.py:284` builds one `SamplingParams(..., seed=42)` and passes it to every prompt in a round.
   - With vLLM's per-request seeding, all agents that receive the same prompt sample the same text: 52% of round-0 agent outputs were identical.
   - The "majority vote" is then largely one sample counted several times.
2. **No generation prompt.** `inference/prompts.py` (lines 173 and 243) calls `apply_chat_template(messages, tokenize=False)` without `add_generation_prompt=True`.
   - The prompt therefore ends after the user turn, with no assistant header.
   - Result: 18% of round-0 answers were unparseable. In later rounds peers demonstrate the `<answer>` format, so parse errors fall to 3–7%.
3. **String matching.** `utils/eval_utils.py:83` (`gsm8k_match`) compares cleaned strings, so "7.00" ≠ "7".

**Effect on the headline contrast** (final-round MAD vote − round-0 majority vote, paired bootstrap, 95% CI):

| Pipeline | Change, pp |
|---|---|
| released code | **+10.0** [+5.3, +15.0] (debate beats voting) |
| per-agent seeds + generation prompt, your scorer | **−5.3** [−8.7, −2.3] |
| same, numeric scoring | −3.9 [−7.0, −1.0] |

Your paper reports −6.1.

**On Llama-3.1-8B-Instruct** (Meta's official chat template), the pattern repeats and is larger:
- The released code gives **+25.0** pp [+18.0, +31.7]. 88% of round-0 agents are identical.
- With the seeds and generation prompt fixed, your scorer gives **−2.0** pp, exactly your reported Llama number.
- However, about half of Llama's round-0 answers end with `<answer>\nN` and no closing tag, and `vllm_debate.py:61` counts those as parse errors. Reading them gives a decline of −11.8 pp [−16.3, −7.3].
- **Your conclusion therefore looks, if anything, understated on Llama.**

So we think your conclusion holds on both models, but the released script gives the opposite sign. We also found that it does not reproduce the paper's baseline numbers.

**Could you tell us which code version produced the paper's numbers?** We'd like our write-up to describe your pipeline accurately. We are happy to share all raw outputs and scripts.

Best regards,
[name, affiliation]
