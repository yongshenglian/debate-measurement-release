# Does the check or independence sentence help without a peer? Protocol

Version: `self-review-v1`. Drafted 2026-11-19, after the user asked: "what happens if
we ask the llm to look at its own answer with check or my sentence? we did not test
that." **Frozen with the code at `prepare`.**

- **Budget:** about 4 GPU-hours. **Hard cap: 8.0 allocated GPU-hours.**
- **Approval:** the user asked for this test; it is under the size that needs
  separate approval.

## Background

In `peer-warning-20261119-b` (Study 3), the reader's first answer was right 79.6% of
the time and the private second look reached 85.9%. Two instructions given with a
peer gave the best results:
- check: "Check each step of the peer's proposal yourself before you answer."
- indep: the user's sentence.

With half the peers wrong, both instructions beat private by about 3 pp (computed
after the run, not pre-registered). But private had no added sentence, so that
3 pp may come from the instruction making the reader review its own work more
carefully, not from the peer. This study adds the same sentences to the no-peer
condition, with the reference to the peer removed.

## Questions

| | Question | Contrast |
|---|---|---|
| Q1 | Does the check sentence improve self-review? | self_check vs private |
| Q2 | Does the independence sentence improve self-review? | self_indep vs private |
| Q3 | With the check sentence, do peers (half wrong) add value over reviewing alone? | net(Study 3 C/E_full_check) vs self_check |
| Q4 | Same, for the independence sentence | net(Study 3 C/E_full_indep) vs self_indep |

## Conditions (647 Gemma-arm tasks of Study 3 × 2 replicates)

Each condition uses Study 3's private instruction ("Recheck the problem independently
and submit your final answer. No peer proposals are supplied. Ignore padding."), with
one sentence inserted before "Ignore padding.":

| Condition | Inserted sentence |
|---|---|
| `private` | none (byte-identical to Study 3's private prompt) |
| `self_check` | "Check each step of your proposal yourself before you answer." |
| `self_indep` | "Take a fresh look at the work and use your own judgement." |
| `self_doubt` | "Take a fresh look at the work and use your own judgement; do not be misled by your earlier proposal, however convincing it may sound." |

- **What each sentence is:**
  - `self_check` is Study 3's check sentence, with "the peer's proposal" changed
    to "your proposal".
  - `self_indep` is the user's sentence without its clause about the peer.
  - `self_doubt` keeps the full sentence and points the warning at the reader's
    own earlier answer. It tests whether distrusting one's own answer helps or
    hurts.
- **Matching:** every prompt is padded to Study 3's per-task token target, so all
  Study 3 conditions and these four have the same length per task.
- **Seeds:** Study 3's per-(task, replicate) seeds, shared by all conditions. The
  `private` rerun has Study 3's exact prompt and seed, so how often its answers
  match is a determinism check. Q3 and Q4 pair this run with Study 3's frozen
  outputs, on the same tasks, seeds and lengths.

## Analysis

The statistics are the same as Study 3's:
- the per-task mean over replicates;
- the exact sign-flip test;
- a task-paired bootstrap (20,000 draws, seed 20261117);
- Holm over the four primaries;
- labels: benefit or harm at Holm p < 0.05; `equivalent_within_2pp` when the
  90% CI is inside ±2 pp; otherwise inconclusive.

**Primary:** R1 self_check − private; R2 self_indep − private; R3 net_full_check −
self_check; R4 net_full_indep − self_indep. Here net = the mean of the C and E
conditions (half the peers wrong).

**Secondary (uncorrected):**
- S1 self_doubt − private
- S2 self_doubt − self_indep
- S3–S6 each Study 3 E/C check or indep condition − the matching self condition
- S7–S9 net full, check and indep − this run's private
- S10 self_check − self_indep
- also: change from the initial answer, corruption, rescue, rate of changed
  answers, replicate agreement, and the private replay's identical-answer rate.

**Reading:**
- R3/R4 > 0: peers add value beyond the instruction alone.
- R3/R4 ≤ 0 with R1/R2 > 0: the earlier +3 pp came from the instruction, not the
  peer.

## Integrity

- The schedule is rebuilt from Study 3's frozen tasks and schedule, and bound to
  that schedule by sha256.
- Every event is replayed independently, and the full grid is checked.
- Workers mount only the schedule and the model manifest: no labels and no
  Study 3 outcomes.

## Budget

- **Calls:** 647 × 4 × 2 = 5,176 recipient calls. At about 2.5 s per call, that
  is about 3.6 GPU-hours, or about 1 h wall on 4 GPUs.
- **Limits:** container timeout 7,200 s per GPU, so the cap is 8.0 GPU-hours. No
  retries.

## Code (`collaboration/self_review/`)

| Module | Contents |
|---|---|
| `sr_design.py`, `sr_contract.py` | conditions, sentences and payloads |
| `sr_schedule.py` | builds the schedule |
| `sr_worker.py` | `pw_worker` with a 7,100 s deadline |
| `sr_audit.py` | the audit |
| `sr_launch.py` | stages |
| `test_sr.py` | 3 tests, passing |
