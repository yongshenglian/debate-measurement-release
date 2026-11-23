# Draft outline: "Debate doesn't degrade; parsing does. Format contagion in multi-agent LLM debate"

Status: working outline, 2026-11-19. Numbers marked [v2] are filled in when the
generality arms finish.

## Claim in one sentence

In faithful replications of a NeurIPS 2025 benchmark study, the reported decline
of multi-agent debate over rounds comes from answer-format failures that spread
between agents, not from worse reasoning. Correctly scored, debate neither
helps nor hurts the vote, and individual accuracy rises.

## Contributions

1. **Re-scoring.** We re-score four replication arms (GSM8K and arithmetic,
   dense and sparse) with a blind-validated extractor.
   - The vote decline by round 5 disappears: GSM8K −5.4 → +0.4 pp; arithmetic
     −16.3 → −2.0 pp ns.
   - Individual accuracy rises about 5 pp in round 1.
   - This contradicts the flat-accuracy evidence offered for the martingale
     account, while the "debate ≈ voting" conclusion survives.
2. **Validation.** An independent model (Gemma-3-27B), blind to labels, agrees
   with the robust extractor on 99.6% of recovered answers.
3. **Causal test.** A controlled twin experiment holds peer content and stated
   answers fixed and changes only peer markup. Broken peer markup causes reader
   markup to break: +5.3 pp from one escaped-brace peer, +91.6 pp from four
   looping peers. Actual accuracy is unchanged (0.0 pp [−1.1, +1.1]).
4. **Natural-trace evidence.** Within the same task and round, a clean agent
   next to a broken neighbour is +16.5 pp [12.9, 20.3] more likely to break
   format.
5. **Generality (v2).** Contagion is model- and protocol-dependent.
   - Qwen raw: +92 pp (4 looping peers). Qwen chat template: +15 pp.
     Mistral-24B: +10 pp. Gemma-27B: +1 pp.
   - Reasoning is unaffected in every arm.
   - Framing: a protocol-specific measurement artifact that explains a published
     trend, plus a contagion mechanism that is strongest for small models
     prompted raw. Not a universal failure mode.
6. **Recommendations for multi-agent evaluation:**
   - report extraction-failure rates per round;
   - score with validated, format-tolerant extractors or structured outputs;
   - audit with a blind second extractor;
   - avoid raw (template-free) prompting.

## Sections

1. **Introduction.** "Debate hurts with more rounds" is an influential claim
   (Debate-or-Vote; Talk Isn't Always Cheap). Extraction is a known nuisance;
   nobody has measured how much of the round-wise trend it explains, or whether
   it is contagious.
2. **Background.**
   - Debate-or-Vote protocol: martingale theory, `{final answer: X}` format,
     last-`{...}` regex.
   - Related: extraction as a confound in voting (2606.16710); error cascades;
     CORBA (contagious repetition attacks); When Debate Helps (recovery and
     damage decomposition).
3. **Re-scoring replications.** Method, extractor, Table 1 (published vs robust
   by round, arm), Figure 1 (round curves).
4. **Validation.** Blind audit, Table 2.
5. **Format contagion.**
   - natural traces (dose-response, within-task);
   - twin experiment (Table 3);
   - generality across readers and templates (Table 4) [v2].
6. **Discussion.**
   - What survives of debate-or-vote.
   - Measurement hygiene for MAS.
   - Contagion as a failure mode distinct from persuasion: it hits availability
     and formatting, not belief.
7. **Limitations.**
   - our replication, not the authors' raw outputs (not released);
   - our round-0 published-extractor accuracy is 79.7% vs their 87.1%;
   - one reader in the re-scoring;
   - a fixed loop pattern in the twin test.

## Open items before submission

- Ask the Debate-or-Vote authors for raw outputs, or rerun their Llama-3.1-8B
  configuration (needs a download and GPU approval).
- Check whether Wynn et al.'s parser, the last `{digits}`, shows the same effect
  on their released code.
- Choose a venue: a short paper or comment (e.g. NeurIPS/ICLR workshop, TMLR, or
  ACL Findings).
