# Debate-or-Vote published-baseline replication

Frozen before inference on 2026-11-16. Reproduce the key open-weight result from Choi, Zhu, and Li, *Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?* (NeurIPS 2025 Spotlight), before further custom collaboration experiments.

## Source and runtime

- Official code commit: `82c929ea773d534cfb3fb0ddc4b7d14d245ab549`.
- Model: `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- Authors' environment: Python 3.10.16, Torch 2.7.1+cu128, Transformers 4.55.2, Datasets 4.0.0, Accelerate 1.8.1.
- Generation: five agents, 512 new tokens, sampling, temperature 1.0, top-p 0.9, default seed 42.
- Preserve the authors' unusual prompt behavior: raw user content without the Qwen chat template for homogeneous agents.

## Stages

1. Integrity smoke test: ten deterministic synthetic-arithmetic questions, one fully connected run. Confirm parsing, histories, and round scoring.
2. Published arithmetic workload: 100 questions. Run the authors' fully connected, sparse, and centralized five-agent commands, plus a one-agent baseline. The five independent round-zero outputs are the majority-vote condition; later rounds are communicating conditions.
3. Proceed to the published 300-question GSM8K workload only after arithmetic is operationally valid. Pin the dataset revision and document that the paper did not publish its dataset commit.

Primary reproduction targets for Qwen2.5-7B-Instruct arithmetic are single-agent `0.8140`, majority vote `0.9900`, and best MAD `0.8400`. For GSM8K they are single-agent `0.8713`, majority vote `0.9400`, and best MAD `0.9033`.

## Interpretation

Do not call majority voting collaboration. Report three contrasts separately: majority vote minus single agent (independent sampling/aggregation), debate minus the same round-zero vote (communication), and debate minus single agent (combined scaling plus communication). Keep calls, generated tokens, wall time, and GPU-hours visible. A replication can reproduce the qualitative ordering without exactly matching stochastic point estimates.
