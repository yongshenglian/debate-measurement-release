# Explicit one-agent GSM8K arm

Status: complete and strictly audited.

## Execution

- Container: `debate-vote-gsm8k-20261117-a-single-gpu3`
- Exit: 0; OOM: false
- Start: `2026-11-17T03:25:31.986199434Z`
- Finish: `2026-11-17T04:03:12.473824711Z`
- Wall time: 2,260.49 seconds (37 minutes 40.49 seconds)
- GPU 3 after exit: 15 MiB, 0% utilization; no hardware thermal slowdown was observed.
- Model/source/runtime pins are those in `collaboration/debate_vote_replication/PROTOCOL.md`.
- Invocation: one agent, zero debate rounds, 300 GSM8K test questions, default seed 42 and published generation settings.

## Strict audit

Authoritative artifact: `single-audit-v2.json`.

- History rows: 300/300
- Logged top-level questions: 300/300
- Exact question order and labels versus pinned GSM8K cache: pass
- Schema and stored-versus-recomputed correctness: pass, zero problems
- Correct: 243/300 = **81.0%**
- Empty answer extractions: 27/300 = **9.0%**
- History SHA-256: `7e43100f95ff811c160610b0013ae1c3a4b6dd39c5476e6d626919fac27765d7`
- Audit SHA-256: `0b48da5b149f957dcf51ed17eefc2834e24f187f83192a647947a3efdab724bd`

The first strict-log audit file, `single-audit.json`, is retained but is superseded. It parsed 299 questions because tqdm carriage-return output prefixed question 264 inside the same Docker log record. Raw inspection confirmed that question was present. A tested parser now recognizes the benchmark marker after progress text while still requiring exact agreement with all 300 pinned questions.

## Interpretation boundary

This explicit batch-size-one arm is a practical one-call control. It is **not** interchangeable with the paper's reported 87.13% “single-agent” value, which is computed from the individual outputs of five agents generated together in the round-zero five-agent batch. Batch shape and random-number consumption differ. The paper-aligned individual estimate and equal-sampling majority vote must come from the completed dense/sparse round-zero records.

The 9% empty-extraction rate is a material diagnostic, but it does not prove that all empty extractions contain semantically correct answers. Any relaxed or semantic rescoring must be preregistered, independently validated, and reported separately from the official extractor.
