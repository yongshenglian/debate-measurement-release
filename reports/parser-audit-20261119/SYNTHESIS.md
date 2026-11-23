# Measurement validity in multi-agent debate: synthesis of studies A–E (2026-11-20)

## The question

Do published "debate helps / hurts" conclusions survive valid measurement? We re-ran four papers with their own code, on open models, and scored every output twice: with the paper's own pipeline and with a validated alternative.

## Results by paper

| Paper | Headline (GSM8K) | What we found | Verdict |
|---|---|---|---|
| **Debate-or-Vote** (NeurIPS 2025), Qwen2.5-7B | vote declines over rounds (−5.4 pp in our faithful run) | Agents copy escaped braces and loops from peers; the published regex can't read them. With blind-validated extraction: +0.4 pp. | **artifact** |
| Debate-or-Vote, Llama-3.1-8B (197/300) | −9.3 pp (paper) | −4.8 pp under the published scorer; −2.4 to −3.3 n.s. under three unvalidated alternatives. Gemma failed its validity check (outputs often have no committed answer). | **unresolved; smaller than reported** |
| **DMAD** (2601.19921), Qwen2.5-7B | vanilla MAD < majority vote (−6.1) | Released code: shared seed (52–87% identical agents), missing generation prompt (18% unparseable at round 0), string-equality scoring ("7.00" ≠ "7"). The released code gives **+10.0 pp** (debate helps). Both pipeline defects fixed: **−5.3 / −3.9 pp**. | **conclusion correct, but the released code gives the opposite sign** |
| DMAD, Llama-3.1-8B (official template, run 2026-11-20) | −2.0 | Released code **+25.0 pp** (88% identical agents; 52% of round-0 answers unparseable). Fixed pipeline, DMAD scorer: −2.0, exactly the paper's number. But half of round-0 answers end with an unclosed `<answer>` tag that DMAD's parser (and our Qwen-validated extractor) drop. Read validly (post-hoc rule, 15/15 spot check): **−11.8 [−16.3, −7.3]**. The chat template alone moves this from −5.6 to −11.8. | **conclusion holds and is understated; released code gives the opposite sign** |
| **Wynn et al.** (2509.05396) | debate often degrades accuracy | Extraction fine (same verdict with the same vote rule). 2L+1M decline replicates (−7.7 to −9.4). 3×Llama (−3.4) and 3×Mistral (+2.8) don't replicate (n.s.). 1L+2M flips with an unreported tie-break rule: first agent wins, −13.0 pp; ties split, +0.6 pp. | **mixed; hidden vote rule** |
| **Du et al.** (ICML 2024) protocol, open models | debate improves (GPT-3.5, 77 → 85) | Flat with Qwen/Llama under both scorers (+0.7 / −0.3). The parser's weak points are never triggered by chat-templated `\boxed` output. | **robust measurement; no gain with these models** |

## Cross-cutting findings

1. **Pipeline choices move results as much as the effects being reported.**
   - The choices: extraction regex, sampling seeds, generation prompt, string vs numeric matching, tie-breaking.
   - Each moved a headline contrast by 5–15 pp in at least one paper, as large as or larger than the reported debate effects (2–10 pp).
   - None of these choices is reported in the papers.
2. **Format contagion biases evaluations in either direction.**
   - Debate homogenizes answer *format* across agents.
   - With a fragile parser that breaks on copied markup (Debate-or-Vote, raw prompting), debate looks harmful.
   - With a strict scorer, debate looks helpful, because peers teach the format:
     - DMAD round-0 parse errors fall from 18–33% to 3–7%;
     - Du/Llama loses `\boxed` in 15% of round-0 answers;
     - DMAD's faithful +10 pp.
   - The bias tracks rounds, the very variable these papers study.
3. **Convergence interacts with the aggregation rule.**
   - Debate reduces disagreement, so any rule that resolves disagreement asymmetrically shifts with rounds even when answers don't:
     - Wynn's first-agent tie-break;
     - DMAD's set-order tie-break;
     - excluding parse failures from votes.
4. **Real effects remain, and are about correlation, not reasoning.**
   - Where measurement is valid and a vote decline remains (DMAD-fixed Qwen; Wynn 2L+1M), *individual* accuracy rises or holds while the vote falls.
   - Debate makes agents better but correlated, or lets a weaker model pull a stronger one (Llama corrupted by Mistral).
5. **Reproducibility of released code is poor.**
   - Wynn's repo is missing the configs its default config needs.
   - DMAD's script doesn't produce its own baseline numbers.
   - Debate-or-Vote doesn't release raw outputs.

## Recommended protocol (the paper's practical contribution)

- Report per-round parse-failure rates.
- Score numerically with a validated, format-tolerant extractor, plus a blind second extractor on a stratified sample.
- Use independent seeds per agent and check round-0 diversity.
- Apply the chat template with the generation prompt.
- Report the tie-break rule, and give results with ties split evenly.
- Report individual accuracy alongside the vote.

## GPU used for A–E (2026-11-20)

| Item | GPU-h |
|---|---|
| Llama Debate-or-Vote (A, approved earlier) | 8 |
| Gemma extraction (E) | 0.6 |
| DMAD (B) | 5.6 |
| Du (D) | 0.5 |
| Wynn (C) | 3.5 |
| **New since approval** | **about 10.2**, against about 8.5 approved |

The overrun came from DMAD arm times. Two secondary `conf` arms were dropped to limit it.

## Follow-up: when debate helps (studies S and R, 2026-11-20)

- **Settings (`reports/debate-settings-20261120-a/RESULTS.md`):** same model, same 300 problems.
  - The DMAD protocol's harm replicates: −6.3 pp pre-registered; −3.8 pp once Gemma fills the no-answer outputs.
  - Changing any one of these removes it: DoV wording, no chat template, or showing answers only.
  - No setting beats the round-0 vote (93–95%). Safe settings lift individual agents to the vote level.
- **Reverse peer (`reports/reverse-peer-20261120-a/RESULTS.md`):**
  - Gemma reading Qwen: −0.2 [−1.1, +0.6]. Qwen reading Gemma: +1.6 [+0.7, +2.5].
  - Strength raises acceptance of a correct peer (88% vs 44%) but not resistance to a wrong one (abandon rates 29% vs 32%).
  - The net effect follows from peer accuracy on disagreements.

- **Hard problems (`reports/debate-hard-20261120-a/RESULTS.md`, MATH Levels 4–5):**
  - With room above the vote (round-0 vote 68%), no debate arm beats it.
  - 20 independent samples beat every debate arm by 3.7–7.0 pp.
  - Reading a stronger peer: +11.5 pp for Qwen. Gemma reading Qwen: −1.5 [−3.3, +0.3].
  - Accept and abandon rates are task-specific; only the accounting identity generalises.

## Follow-up: do the GSM8K findings hold on hard sets? (studies H, I, T and M, 2026-11-20/21)

| Finding | GSM8K | MATH L4–5 | MMLU-Pro |
|---|---|---|---|
| Weak reads strong peer − self | +1.6 | +11.5 | +9.7 |
| Strong reads weak peer − self | −0.2 | −1.5 (n.s.) | −2.2 (n.s.) |
| Strong abandons own right answer for a wrong peer | 29% | 18.5% | 42% |
| Weak abandons own right answer for a wrong peer | 32% | 38% | 55% |
| Best debate arm − compute-matched 20-vote | n/a (ceiling) | −3.7 (answers only) | −0.9 (answers only, tie) |
| Full-reasoning debate − round-0 vote | −6.3 | −1.9 | −1.2 |
| Twins: wrong working − private (Qwen) | −14.1 | −5.1 | — |
| Twins: confident − hedged sign-off | −12.0 | −6.1 | — |
| Independence sentence, net | +3.0 (twins) | +1.8 twins; peers ±2 | peers ±2 |

### What generalises

- **The accounting identity:** net = rescues on disagreements − abandonments on disagreements, within a few problems on every task. The rates themselves must be measured per task.
- **Reading a stronger model's full work helps,** and its working helps beyond its answer (T5).
- **Answers-only debate is the safest debate,** and only reaches compute-matched voting where extra samples saturate.
- **Confident wording is more persuasive.**

### What does not generalise

- **The size of the persuasion harm** (it shrinks about threefold on MATH).
- **The GSM8K pattern that strength helps acceptance but not resistance:** on MATH strength helps resistance; on MMLU-Pro neither model resists.
- **The independence sentence:** a dial that trades abandonment against acceptance. Its net sign follows the peer's relative accuracy.

## Follow-up: does owner framing cost the reader? (study J, 2026-11-22)

- `reports/neutral-judge-20261121-a/RESULTS.md`: the same two disagreeing solutions under owner, neutral and swapped framings.
- **Neutral labels help the strong reader:** Gemma +2.0 / +2.9 / +6.8 pp on disagreements (MMLU-Pro Holm .047), mostly through fewer abandonments.
- **No complementarity captured:** the neutral-judge system is within ±1 pp of Gemma alone and of a 3-call vote on every task.
- **Exploratory:** the label "your earlier solution" makes both models about 5 pp *less* likely to keep the labelled solution, the opposite of judge self-preference. Position bias (up to 15 pp) flips sign between models and tasks, so prompts should be run in both orders.

## Open items

- **DMAD Llama:** done (about 0.9 GPU-h, approved). The unclosed-tag reader is post hoc and needs a blind audit before submission.
- **Llama Debate-or-Vote extraction** has no validated reference. A human-labelled sample of about 200 outputs would settle it at no GPU cost.
- **Possibly contacting the DMAD and Wynn authors** about the code defects, before submission. This is outward-facing, so it is the user's decision.
