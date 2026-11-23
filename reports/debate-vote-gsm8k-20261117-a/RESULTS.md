# Debate-or-Vote GSM8K replication results

Completed 2026-11-17. All four arms contain exactly 300 ordered GSM8K test
problems. Every scientific container exited zero without OOM. The history/source,
final Docker-log/source, and task-paired v4 audits all passed.

## Main result

| Arm | Round-0 individual | Round-0 readout | Best later round | Later − round 0 |
|---|---:|---:|---:|---:|
| dense, five agents | 79.67% (1195/1500) | 91.67% (275/300) | 91.00% (round 1) | −0.67 pp, paired 95% CI [−3.33, 2.00] |
| sparse, five agents | 81.20% (1218/1500) | 91.33% (274/300) | 92.00% (round 1) | +0.67 pp, paired 95% CI [−1.67, 3.00] |
| centralized | 82.00% (246/300) | 82.00% (246/300) | 79.67% (round 1) | −2.33 pp, paired 95% CI [−7.00, 2.33] |
| explicit one-call control | 81.00% (243/300) | 81.00% | — | — |

The round-zero vote improved on the average individual output by 12.00 percentage
points in dense and 10.13 points in sparse. Additional communication did not
clearly improve either arm's own round-zero vote. Sparse round 1 was two tasks
better, but its paired interval includes zero; dense round 1 was two tasks worse.
Later rounds degraded both arms.

By round 5, dense fell to 86.67% (−5.00 pp, paired 95% CI [−9.00,
−1.00]) and sparse fell to 86.00% (−5.33 pp, paired 95% CI [−8.67,
−2.00]). Centralized generation fell monotonically to 67.00% at round 5
(−15.00 pp, paired 95% CI [−21.00, −9.00]).

## Correction versus corruption

Dense round 1 corrected eight initially wrong votes but corrupted ten initially
correct votes. Sparse round 1 corrected seven and corrupted five. These nearly
cancelled, explaining why a point estimate alone is misleading. At round 5,
dense corrected eleven and corrupted twenty-six; sparse corrected five and
corrupted twenty-one.

Initial candidate coverage was already 96.00% in dense and 96.33% in sparse.
Only 13 dense and 15 sparse tasks had a correct proposal that the round-zero vote
missed. Later debate therefore had limited recoverable headroom and much larger
exposure to damaging an initially correct readout. Dense generated only one new
correct answer beyond initial coverage (rounds 4–5); sparse generated one in round
1 and none later.

Empty final-answer extractions increased with repeated communication. Dense rose
from 153/1500 at round zero to 306/1500 at round 5; sparse rose from 132/1500 to
300/1500. Centralized rose from 27/300 to 77/300. This is part of the observed
protocol behavior, not an excluded infrastructure failure.

## Relation to the published result

Published Qwen2.5-7B anchors are 87.13% mean individual accuracy, 94.00% round-zero
vote, and 90.33% best debate. This run is lower on individual and vote accuracy,
but it reproduces the important qualitative ordering: independent voting provides
a large gain over an individual output, while multi-round debate provides no clear
gain over its own vote and longer debate becomes harmful. Exact stochastic
trajectory or score reproduction was not expected.

The explicit 81.00% batch-size-one arm is not interchangeable with the
paper-aligned mean of five round-zero outputs because batch shape and random-number
consumption differ.

## Interpretation limits

Dense, sparse, and centralized arms were launched in separate stochastic
processes. Only 1/300 dense/sparse and 2/300 dense/centralized round-zero response
sets were byte-identical. Cross-arm topology comparisons are therefore
descriptive. The defensible communication contrasts are within each arm against
its own round zero.

This replication does not show that communities are universally worse than
individual agents. It shows that, for this model, benchmark, and published
protocol, the strong gain comes from independent sampling plus voting rather than
additional rounds of text exchange.

## Integrity and cost

- All 1,200 history rows and all 1,200 logged questions match the pinned ordered
  dataset prefix.
- All four containers exited zero; none was OOM-killed.
- Scientific container durations: dense 7.064 h, sparse 9.225 h, centralized
  10.335 h, explicit single 0.628 h; total 27.252 allocated GPU-hours.
- Authoritative analysis: `paired-audit-v4.json`.
- Authoritative source receipts: `source-history-audit.json` and
  `final-log-source-audit.json`.

## Frozen gate decision

The qualitative baseline gate passes. Voting clearly beats the mean individual,
and communication does not clearly beat its own round-zero vote. Proceed only to
the bounded causal calibration after adding a true private-revision control,
retesting the amended six-cell schedule, and completing the Qwen2.5-7B BF16/FP8
runtime smoke. This is permission to calibrate, not a confirmatory result or a
general novelty claim.
