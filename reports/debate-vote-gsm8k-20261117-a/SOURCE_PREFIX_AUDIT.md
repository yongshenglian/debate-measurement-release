# Live source-prefix audit

During inference on 2026-11-16 local time, the saved histories and flushed top-level question logs were independently checked against the pinned offline `openai/gsm8k` cache revision `740312add88f781978c0658806c59bc2815b9866`.

The verifier reproduced the authors' test-set construction: convert the 1,319-row test split to pandas, shuffle with `sample(frac=1, random_state=0)`, reset the index, and select the first 300 rows. The canonical ordered `(index, question, answer)` manifest has SHA-256:

`2f0dc48003c4268db1bba1917eaf212d30fe6d23d57ad3aeb75508f7c9a05202`

At the audited live prefixes (dense 17 histories/18 logged questions, sparse 15/16, centralized 12/13, single 210/210), every saved label and every logged top-level question matched the expected ordered prefix. A one-question lead in active debate logs is expected because the next question is printed before its completed history record is rewritten. Earlier log lag was also observed from stdout buffering. Therefore strict log-count equality is reserved for post-exit audit.

This is a live-prefix integrity result, not an accuracy result. A detached watcher will run strict 300-row history/label validation. Exact 300-question log matching remains a post-exit check.
