# Audit-watcher correction

On 2026-11-16 local time, while all inference arms were still running, a partial-history schema check found that `analyze_v2.py` incorrectly assumed centralized MAD stores five scored `final_answers` per round. The pinned author source at commit `82c929ea773d534cfb3fb0ddc4b7d14d245ab549` shows that centralized mode generates five `responses` but deliberately passes only agent zero to `evaluate`, so `final_answers` and `final_answer_iscorr` each have length one.

No inference configuration, container, history record, prompt, seed, or result was changed. The first completion watcher had not produced an audit because every arm was far below 300 records. It was stopped after a corrected watcher was verified live; its container exit code 137 records the deliberate stop, not an OOM or scientific-run failure.

The replacement `analyze_v3.py` encodes these arm-specific shapes:

- dense and sparse: five responses, five scored outputs;
- centralized: five responses, one scored output;
- explicit single: one response, one scored output.

Three protocol-specific tests pass, including acceptance of the author-defined centralized representation and rejection of a malformed centralized output count. A read-only audit of all records present at replacement time found zero shape violations. The authoritative completion artifact is `paired-audit-v3.json`; `paired-audit-v2.json` must not be used if it appears.
