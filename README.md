# Data and code release: measurement, sampling and review in multi-agent debate

This repository holds the code, prompts, frozen protocols with their hashes, raw model outputs and analysis outputs for the paper
*Evaluating Multi-Agent Debate: Measurement Artifacts, Independent-Sampling Baselines, and Peer-Review Framing* (under review).

## Layout

- `reports/<study>/`: one folder per study.
  - `PROTOCOL.md`: the protocol, including amendments and deviations, with dates.
  - `frozen*.sha256`: hashes taken before generation.
  - `RESULTS.md`: a summary of the results.
  - `*.out`: analysis outputs.
  - `out*/`: raw model outputs.
- `collaboration/`: generation and analysis code.
  - `collaboration/wynn_rerun/talk-aint-cheap/` is a copy of the authors' MIT-licensed code, included unchanged for the re-run.
- `verify_hashes.py`: checks every hash file (see "Hash check" below).

| Paper label | Folder | What |
|---|---|---|
| Re-run | `debate-vote-gsm8k-20261117-a` | Debate-or-Vote, Qwen2.5-7B |
| Re-run | `debate-vote-gsm8k-llama-20261119-a` | Debate-or-Vote, Llama-3.1-8B (stopped early; 197 of 300 problems) |
| Re-run | `dmad-rerun-20261120-a` | DMAD, Qwen and Llama |
| Re-run | `wynn-rerun-20261120-a` | Wynn et al. |
| Re-run | `du-rerun-20261120-a` | Du et al. |
| Re-run analysis | `peer-theory-20261119` | Released versus validated extractor on the Debate-or-Vote outputs |
| Audit | `format-contagion-20261119-a` | Markup experiment and blind extraction audit |
| Audit | `parser-audit-20261119` | Parser stress test (CPU) |
| GSM8K | `debate-settings-20261120-a`, `self-review-20261119-a` | Debate settings; private revision |
| R | `reverse-peer-20261120-a` | Peer review, both readers |
| H | `debate-hard-20261120-a` | Debate on MATH |
| I | `debate-indep-20261121-a` | Independence instruction |
| T | `twins-hard-20261121-a` | Matched twins, MATH |
| M | `mmlupro-20261121-a` | Debate and peers, MMLU-Pro |
| J | `neutral-judge-20261121-a` | Framing with frozen solutions |
| K | `framing-rep-20261122-a` | Framing replication |
| L | `framing-gen-20261122-a` | Framing with a Qwen3.8-27B judge |
| P | `private-revision-20261123-a` | Private revision, both judges |
| — | `revision-reanalysis-20261122` | Re-analyses for the revision (no new generation) |

## Not yet included

- **The reader audit (`human-audit-20261122`) and the reader-label packet for the Llama Debate-or-Vote re-run.** Both are withheld while a human labelling round is in progress, so that labellers cannot see the automatic readings. They will be added when it is complete.
- **Inference caches, smoke runs, container metadata and model weights.**
  - The models are public: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Gemma-3-27B-it, Mistral-7B-Instruct and Qwen3.8-27B (AWQ INT4).
  - The exact image digests are recorded in each protocol.

## Running the code

Scripts were run on the original machine, and many contain absolute paths beginning `/home/lian/swarmworld/research/`. That prefix corresponds to the root of this repository. Replace it, or create a symlink, to re-run them.

## Hash check

Run `python3 verify_hashes.py` from the repository root.

- Every hashed file is included.
- Some hash files are cumulative: an amendment appends a new hash for a file it changes, and the latest entry is the one checked.

Files that no longer match the hash they were frozen with:

- **`PROTOCOL.md` in `dmad-rerun`, `format-contagion`, `wynn-rerun` and `framing-gen`.** Each has amendment or deviation sections that were appended after freezing. Each section is dated and states whether it was written before or after the outputs it concerns. For `framing-gen`, the amended files are re-hashed in `frozen_amend1.sha256`.
- **`collaboration/when_debate/l_analyze.py` (study L).** It was amended during the run, before any outcome was inspected. It is re-hashed in `frozen_amend1.sha256`.
- **`collaboration/when_debate/reverse_mech.py` (study H).** It computes HR4, the pooled accept and abandon rates, which study H registered as descriptive. It was edited after the freeze, at analysis time, one second before `reports/debate-hard-20261120-a/peer/mech.out` was written. Study H's protocol calls it unchanged, which is wrong.
  - The earlier version was not kept, so the change cannot be shown.
  - Treat HR4 in `mech.out` as not covered by the freeze.
  - HR1–HR3 come from `reverse_analyze.py`, which matches its hash.
  - The accept and abandon rates shown in the paper are recomputed independently in `revision-reanalysis-20261122`.

## Labels produced by AI

- No label in this release is presented as a human label.
- Where the paper uses reference labels from LLM readers, they are described as such in the paper and in the study's `RESULTS.md`.
