# Study B: DMAD vanilla-MAD re-run: results

- **Protocol:** `PROTOCOL.md` (with Amendment 1, Deviations 1–2).
- **Analysis:** `collaboration/dmad_rerun/analyze.py` → `analysis.out`, `analysis_out.json`.
- **Data:** GSM8K, 300 questions; 5 agents × 5 rounds (rounds 0–4).
- **CIs:** paired bootstrap, 20,000 draws, 95%.
- **GPU time:** about 5.6 GPU-h.

## Paper (DMAD Table 1, GSM8K)

| Model | Single model | Majority vote | Vanilla MAD |
|---|---|---|---|
| Qwen | 0.908 | 0.908 | 0.847 (−6.1 vs majority vote) |
| Llama | 0.797 | 0.870 | 0.850 (−2.0) |

## Qwen2.5-7B (valid in every arm)

"Last − round 0" is the round-4 vote minus the round-0 vote.

| Arm | Identical round-0 texts | Round-0 parse errors | Round-0 vote, DMAD scorer | Round-0 vote, validated | Last-round vote, DMAD scorer | Last-round vote, validated | Last − round 0, DMAD scorer | Last − round 0, validated |
|---|---|---|---|---|---|---|---|---|
| faithful (DMAD code) | **52%** | **18.0%** | 72.3 | 75.8 | 82.3 | 84.0 | **+10.0** [+5.3, +15.0] | **+8.2** [+2.8, +13.3] |
| nostop | 46% | 17.6% | 72.3 | 75.7 | 83.0 | 86.8 | +10.7 [+6.0, +15.3] | +11.2 [+6.5, +15.8] |
| agentseed | 0% | 18.4% | 85.7 | 92.7 | 86.0 | 91.3 | +0.3 [−2.7, +3.3] | −1.4 [−4.4, +1.5] |
| **fixed** (agentseed + generation prompt) | 0% | **0.7%** | **88.3** | **93.6** | 83.0 | 89.7 | **−5.3** [−8.7, −2.3] | **−3.9** [−7.0, −1.0] |

## Findings

1. **DMAD's released vanilla-MAD script has two pipeline defects. Each one alone can reverse the MAD-vs-vote verdict.**
   - **Shared sampling seed.** One `SamplingParams(seed=42)` serves all agents. Five agents with identical round-0 prompts then sample near-identical text: identical on 52% of Qwen questions, and on 87% for Llama even with its intact assistant header.
     - The round-0 "vote" is effectively one sample.
     - Qwen's paper "majority vote" (0.908) equals its "single model" (0.908), which fits this collapse. That the paper's baseline came from this exact script is our inference, not a fact.
   - **No generation prompt.** `apply_chat_template(..., tokenize=False)` is called without `add_generation_prompt=True`.
     - Qwen must open its own turn. It starts with junk, and 18% of round-0 answers lack `<answer>` tags.
     - Debate prompts show peers' answers wrapped in `<answer>` tags, which repairs the format. This inflates the apparent gain from debate.
   - **With both defects (faithful):** debate looks strongly *beneficial*: +10.0 pp.
   - **With both fixed:** debate is *harmful*, −5.3 / −3.9 pp, matching the paper's −6.1.
   - **The paper's headline (MAD < majority vote) is therefore right for Qwen, but the released code does not produce it.**
     - The released code gives the opposite sign.
     - Its round-0 vote (72%) sits far below the paper's 0.908 baseline.
     - Its last-round vote (82.3) is close to the paper's 0.847.
     - This fits the paper having mixed pipelines: a healthy one for the majority vote, the defective one for MAD. That is an inference; the baseline-generation code is not released.
2. **The scorer has a string-equality bug.** `gsm8k_match` compares cleaned *strings*, so "7.00" ≠ "7" and "$4.00" is wrong.
   - About 4% of round-0 answers in the fixed arm are scored wrong for this reason.
   - Votes split between "7" and "7.00".
   - It lowers every accuracy by about 3–6 pp but does not change any verdict here.
3. **The stop token is a negligible artifact (P4).** Stopping at the first `</answer>` would change the recorded answer in 0.4–1.6% of Qwen outputs per round (nostop arm). Its effect on the faithful-vs-nostop verdict is null.
4. **The decline in the fixed pipeline is real, not a measurement artifact.**
   - Parse errors are under 1%, and the decline holds under the validated scorer.
   - Individual accuracy *rises* (86.6 → 88.6, validated) while the vote falls. Debate makes agents individually better but correlated: the vote loses its independent-error advantage.

## Llama-3.1-8B (withheld)

The weights mirror ships an older chat template that always adds the assistant header (Deviation 2). So the generation-prompt defect never occurred, and `fixed` = `agentseed`.

**Descriptive results:**

| Arm | Identical round-0 texts | Round-0 vote | Last-round vote | Last − round 0 |
|---|---|---|---|---|
| faithful | 87% | 55.0 | 54.7 | −0.3 [−7.0, +6.7] |
| agentseed / fixed | 0% | 80.3 | 78.0 | −2.3 [−6.7, +2.0] (validated −2.6 / −3.4, n.s.) |

**Llama-specific notes:**
- Round-0 parse errors are 32–33% in every arm. Llama often writes `reasoning` / `answer` without angle brackets, or omits the closing tag.
- Debate repairs this to about 3.5% by round 4, because agents copy their peers' tags.
- Individual accuracy rises 52.5 → 74.7 (DMAD scorer) in agentseed. This is mostly format repair.

**Pending (needs approval, about 1.7 GPU-h):** a re-run of the faithful and fixed arms with the official Llama-3.1 template (prepared, hash-verified).

## Llama-3.1-8B with the official chat template (Amendment 2, run 2026-11-20)

- **Runs:** `out/llama-t_faithful.json` and `out/llama-t_fixed.json`. Both exited 0 with 300 problems each; about 0.9 GPU-h.
- **Pre-registered analysis:** `analysis_v2.out`, the same `analyze.py`. It also re-prints the earlier arms unchanged.

### Pre-registered results (final-round vote − round-0 majority vote, pp)

| Arm | Identical round-0 texts | Round-0 DMAD parse errors | DMAD scorer | Validated (pre-registered) |
|---|---|---|---|---|
| faithful (released code) | 88% | 51.7% | **+25.0** [+18.0, +31.7] | +27.3 [+20.3, +34.2] |
| fixed (per-agent seeds + generation prompt) | 0% | 52.0% | −2.0 [−7.3, +3.3] | +0.8 [−4.6, +6.1] |

**The fixed arm still has 52% round-0 parse errors.** Llama often ends its turn right after `<answer>\nN` without writing `</answer>`.
- DMAD's parser needs both tags.
- So does our pre-registered `validated` extractor, which was validated on Qwen.
- The pre-registered validated column is therefore **not valid on Llama**. This is the same lesson as Debate-or-Vote Llama: extractors must be validated per model.

### Post-hoc sensitivity: accept an unclosed final `<answer>` (`collaboration/dmad_rerun/sensitivity_unclosed.py`, output `sensitivity_unclosed.out`)

**Rule:** if `validated` gives nothing, and the text after the last `<answer>` has no closing tag and exactly one number, take that number.

**Check:**
- 573 of the 575 recovered tails are under 25 characters.
- 15 random recovered items were read by hand, blind to labels: all 15 numbers are the committed answer.
- The rule recovers 35–38% of round-0 outputs in the official-template arms, 12–13% in the mirror arms, about 0 in later rounds, and 0 on Qwen.

| Arm | Vote, round 0 → round 4 | Individual, round 0 → round 4 | Final − round 0 |
|---|---|---|---|
| Llama official, faithful | 69.0 → 69.3 | 68.3 → 65.2 | +0.3 [−5.3, +6.0] |
| **Llama official, fixed** | **88.3 → 76.5** | 71.1 → 73.1 | **−11.8 [−16.3, −7.3]** |
| Llama mirror, fixed | 85.3 → 79.7 | 64.2 → 77.4 | −5.6 [−9.8, −1.5] |
| Qwen fixed (unchanged) | 93.6 → 89.7 | 86.7 → 88.6 | −3.9 [−7.0, −1.0] |

### Reading

1. **The released code reverses DMAD's conclusion on Llama even more than on Qwen: +25.0 pp.**
   - Agents are 88% identical, and half of round-0 answers are unreadable.
   - Debate then "repairs" the format: peers show closed `<answer>…</answer>` tags.
2. **The DMAD scorer on the fixed pipeline gives −2.0 pp, exactly the paper's Llama number.** This supports our inference that the paper's numbers come from a seed-fixed pipeline scored with DMAD's own parser.
3. **With answers read validly, the Llama decline is much larger: −11.8 pp, against the −2.0 reported.**
   - DMAD's scorer understates the decline, because it drops correct round-0 answers that lack a closing tag.
   - **The paper's conclusion (MAD < majority vote) holds, and is stronger than reported.**
4. **The prompt template alone moves the valid-reading effect from −5.6 (mirror) to −11.8 pp (official).** The official template adds a date system block.
5. **Mechanism:** individual accuracy is flat to rising (71.1 → 73.1) while the vote falls 11.8 pp. This is correlation, not degradation, as with Qwen.

- **Status:** points 3–5 rest on a post-hoc extractor rule, checked on 15 items by Claude. Before submission they need confirmation: the same blind human or Gemma audit used elsewhere, on a stratified sample.
