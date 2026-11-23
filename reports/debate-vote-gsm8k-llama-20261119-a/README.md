# Debate-or-Vote GSM8K replication with Llama-3.1-8B-Instruct (dense arm)

- **Approved by the user** on 2026-11-19 ("you can run the Llama replication";
  mirror option 2).
- **Purpose:** test whether the published Llama-3.1-8B round-wise decline
  (GSM8K vote 0.8733 → Decentralized MAD T=5 0.7800) is, like Qwen's in our
  replication, an extraction artifact.
- **Setup:** identical to the Qwen dense arm (`reports/debate-vote-gsm8k-20261117-a`).
  - Same container image, environment, author source commit and dataset.
  - Same arguments (5 agents, 300 problems, 5 rounds, seed 42), with only
    `--model llama3.1-8b` changed.
- **Weights:** `models/debate-or-vote/llama3.1-8b-instruct-nous-d10aef7`, from the
  NousResearch mirror at revision d10aef7999a2.
  - All 4 safetensors match Meta's official SHA-256 for
    meta-llama/Llama-3.1-8B-Instruct@0e9e39f (`PROVENANCE.json`).
  - config, generation config, tokenizer.json and special tokens are
    byte-identical.
  - **Known deviation:** `tokenizer_config.json` differs (an older chat template).
    Debate-or-Vote does not apply chat templates.
- **Analysis plan (frozen now):** run `reports/peer-theory-20261119/rescore.py`
  (published vs robust extractor) on the history, and report round-wise vote
  and individual accuracy with paired bootstrap CIs against round 0.
- **Budget:** about 7–9 GPU-h on one A6000, with a timeout of 24 h.
- 2026-11-20: the first launch exited after problem 1 because `out/history/` was not pre-created (the Qwen run's was). Relaunched from scratch with the folder created; no data was kept from the first launch.
- **2026-11-20 00:28 UTC: stopping rule, fixed before any results were inspected.**
  - Speed: about 3.5 min per problem (Llama outputs loop to 512 tokens), so 300
    problems would take about 17 GPU-h, against the approved estimate of 7–9.
  - Rule: stop the container 8.0 h after its start (2026-11-20T00:20:34Z) or at 300
    problems, whichever is first.
  - Analysis: all completed problems, in dataset order, with the same frozen analysis.
  - The number of problems is set by the budget, not by the results.
