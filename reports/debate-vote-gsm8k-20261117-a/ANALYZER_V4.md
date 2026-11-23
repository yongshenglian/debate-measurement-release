# Final analyzer amendment before completion

While all three debate arms were still running, a read-only check of their first 20 common tasks showed that separate GPU processes did not reproduce an identical round-zero proposal pool despite the shared seed. Dense versus sparse had one task with byte-identical response dictionaries; dense versus centralized had two. This is not a run failure: stochastic generation occurs separately in each process. It changes the valid contrast.

Consequences:

- Every communication effect is computed within arm against that arm's own round-zero outcome.
- Dense-versus-sparse and centralized comparisons are separate stochastic run comparisons, not frozen-proposal causal contrasts.
- The proposed precision experiment must generate and freeze one initial proposal pool before crossing recipient conditions.

`analyze_v4.py` supersedes `v3`. It also corrects the descriptive best-round label so round zero cannot be called a communication round, and adds for dense/sparse:

- initial proposal coverage and all-wrong count;
- recoverable headroom;
- recovery of an initially surfaced but vote-losing correct answer;
- new correctness when no initial agent was correct;
- damage to a correct round-zero vote;
- an exact algebraic identity check that observed gain equals recovery plus new correctness minus damage.

Two protocol tests pass. The `v4` completion watcher was verified live before the `v3` auxiliary watcher was deliberately stopped. Its exit 137 records that stop; no inference container was changed. The authoritative final analysis artifact is `paired-audit-v4.json`.
