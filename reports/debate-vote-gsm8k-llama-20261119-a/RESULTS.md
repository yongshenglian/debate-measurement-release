# Llama-3.1-8B Debate-or-Vote replication (GSM8K, dense): results

- **Run:** 2026-11-20 00:20–08:23 UTC. Stopped by the pre-set 8 h rule (container exit 143 = SIGTERM from `docker stop`).
- **Completed problems:** 197 of 300, in dataset order. The last record is complete.
- **Analysis:** `analysis/rescore_llama.py` → `analysis/rescore_llama.out`.

## 1. Vote change by round 5 (paired bootstrap over problems, 95% CI)

| Extractor | Vote round 0 | Vote round 5 | Change | Individual accuracy round 0 → round 5 |
|---|---|---|---|---|
| Published (authors' last-`{...}`) | 86.8% | 82.0% | **−4.8 pp [−8.9, −0.9]** | 73.2% → 77.0% |
| Published, with the quoted prompt example removed | 86.8% | 83.5% | −3.3 pp [−7.0, +0.4] | 73.7% → 78.5% |
| Robust (validated on Qwen) | 87.1% | 81.2% | −5.9 pp [−10.2, −1.8] | 74.2% → 75.6% |
| Robust, with the quoted prompt example removed | 87.1% | 84.5% | −2.6 pp [−6.3, +0.9] | 74.8% → 78.8% |

- **The paper's Llama numbers:** vote 0.873, decentralized MAD at T=5 0.780 (−9.3 pp).
- **Our round 0 matches** (86.8%). Our published-extractor decline is half the paper's. It's on 197 problems, not 300.

## 2. The Qwen extractor is not valid on Llama

**Llama's format failures differ from Qwen's:**
- **No escaped braces:** 0% in every round, against Qwen's 4.6% → 19.5%.
- **Quoted format example:** Llama quotes the prompt's example `"{final answer: 123}"` back while discussing the format, in 1.6% of outputs at round 0 and 6.9% at round 5.
  - The robust extractor prefers any brace-closed `final answer: N`, so it reads **123**.
  - Wrong 123 reads by round: 11, 25, 39, 55, 53, 51.
  - In the Qwen runs there are none.
- **Bare braces:** Llama usually answers `The final answer is: {40}`, not `{final answer: 40}`. When it also mentions the format, the robust extractor's priority picks the format text over the actual last answer.
- **Loops grow:** 5.6% → 35.5% of outputs. As on Qwen, last-match extraction mostly survives them.

**Consequences:**
- The "robust" row above is **not** a valid measurement for Llama.
- The "+echo" rows are fixes defined *after* looking at the Llama outputs, so they are exploratory.
- About 15–20 disagreements per round remain even after the fix, and the published extractor is usually the right one there.

## 3. Reading

- **Llama's decline is smaller than reported and fragile to scoring.**
  - It is significant only under the published extractor, and then barely: −4.8 pp, with the CI upper bound at −0.9.
  - Removing one clear extraction error (the quoted format example) makes it −2.6 to −3.3 pp and not significant.
  - These CIs still allow a real decline of up to about 6–7 pp.
- **Not shown:** that Llama's decline is *entirely* an extraction artifact, as it was for Qwen (Qwen: −5.4 → +0.4 pp, with a blind-validated extractor).
- **Individual accuracy rises from round 0 to round 1** under every extractor (+3 to +5 pp), as on Qwen. This contradicts the "flat individual accuracy" evidence offered for the martingale account.
- **Format contagion takes a model-specific form.** Qwen copies escaped braces. Llama increasingly loops and quotes the format instruction. A regex validated on one model can fail on another.
  - This strengthens the measurement-validity argument: extractors must be validated per model.
  - It also means a hand-tuned regex cannot settle the Llama question.

## 4. What would settle it

- **Blind Gemma-3-27B extraction** (same audit prompt as the format-contagion study) on all Llama outputs at rounds 0 and 5: 197 × 5 × 2 = 1,970 calls, about 0.5–1 GPU-h.
- This gives a regex-free vote change. It is part of study E in `reports/parser-audit-20261119/PLAN.md` and awaits approval.

## 5. Blind Gemma-3-27B extraction (study E, Llama part): pre-registered result

- **Protocol:** `blind-extraction/PROTOCOL.md`, frozen before any output.
- **Coverage:** 5,910 outputs, all 6 rounds, labels hidden.
- **Outputs:** `blind-extraction/spot.txt` (hand check) and `blind-extraction/main.out`.

### Validity check: failed

The pre-registered hand check read 30 random Gemma-vs-published disagreements, blind to labels.

| Count | Reading |
|---|---|
| 5 | Gemma clearly misread the stated answer |
| 15 | Gemma right; the published extractor read the quoted example "123", a peer's answer or a fraction |
| 10 | genuinely ambiguous |

**The 5 misreads:**
- item 8: a 370 loop read as null;
- item 11: 9830.08 read as 9830;
- items 19 and 27: an intermediate value instead of the stated answer;
- item 25: 15 where the response states `{final answer: 17}`.

**The 10 ambiguous cases** are truncated mid-reasoning, oscillate between answers, or end in loops of a peer's answer. In several truncated cases Gemma inferred an answer the agent never stated (items 3, 7, 15, 17). That is a liberal reading.

**Rule outcome:** 5 > 3, so under the pre-registered rule Gemma is **not** treated as the reference for Llama. The result below is descriptive.

### Descriptive result (Gemma readings)

| Round | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Vote | 87.1 | 85.9 | 86.5 | 86.4 | 85.3 | 84.7 |
| Individual | 77.7 | 80.8 | 81.1 | 82.5 | 81.6 | 81.5 |

- **Vote change by round 5:** −2.4 pp [−6.3, +1.1].
- **Agreement with the regex extractors:** 89–94% per round. It is highest with the "+echo" variants (92–94% from round 1).

### Reading

Three different readings converge, and none of them is validated on Llama:

| Reading | Vote change by round 5 |
|---|---|
| Gemma | −2.4 pp, n.s. |
| robust+echo | −2.6 pp, n.s. |
| published+echo | −3.3 pp, n.s. |
| Published extractor | −4.8 pp, the only significant estimate, with an upper bound of −0.9 |

- **Individual accuracy rises after debate** under every reading.
- **The deeper Llama finding:** by later rounds, a growing share of outputs (loops, oscillation, quoted prompts and peers, truncation) **have no well-defined committed answer**. "Accuracy" of these outputs is partly a choice of the scorer.
- **Status:** the Llama decline is unresolved. The Qwen result (validated) stands.
