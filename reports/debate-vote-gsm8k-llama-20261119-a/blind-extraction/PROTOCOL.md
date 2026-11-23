# Blind model extraction of the Llama-3.1-8B Debate-or-Vote outputs (study E, Llama part)

- **Approved:** by the user on 2026-11-20 ("approve both").
- **Frozen:** this protocol is written before any Gemma output exists.

## Design

- **Extractor:** Gemma-3-27B-it W4A16 (`models/heterogeneous/gemma-3-27b-it-w4a16`).
  - Pinned vLLM image, chat template, greedy decoding, 64 tokens.
  - The system prompt is byte-identical to the format-contagion audit (`fc_audit_prepare.SYSTEM`).
- **Blinding:** the extractor sees only the response text. It sees no question, label, round or agent.
  - Job order is shuffled with seed 20261120.
- **Items:** every agent output of the 197 completed problems, all 6 rounds: 5,910 calls.
  - The primary contrast uses rounds 0 and 5. The other rounds are descriptive.
- **Parsing Gemma's reply:**
  - Take the JSON `answer` and round it to 1 decimal.
  - Null, unparseable or non-numeric replies count as no answer.

## Primary analysis

- **Contrast:** vote accuracy at round 5 minus round 0, with Gemma's readings as the agents' answers.
- **Voting:** the published plurality vote. Ties are split; no-answer agents are excluded; a problem with no answers counts as 0.
- **Interval:** paired bootstrap over problems, 20,000 draws, seed 20261117, 95% percentile CI.
- **Reading:**
  - CI excludes 0 below zero: the decline is real under a regex-free reading.
  - CI includes 0 and the upper bound is above −2 pp: the decline is not established.

## Secondary analysis

- Individual accuracy by round.
- Gemma's agreement with each regex extractor (published, published+echo, robust, robust+echo) by round.
- On disagreements, which extractor matches Gemma.

## Validity check

- **Spot check:** 30 random items where Gemma disagrees with the published extractor are read by hand (by Claude), blind to the label, before the vote analysis is run.
- **Failure rule:** if more than 3 of the 30 show Gemma clearly misreading the committed answer, Gemma is not treated as the reference. The result is then reported as unresolved.
