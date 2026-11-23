import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / d) for d in ("kv_communication", "peer_evidence", "peer_strength",
                                               "peer_clean", "peer_discrimination", "peer_persuasion",
                                               "peer_warning")] + [str(HERE)]

import pw_design  # noqa: E402
from pw_contract import INSTRUCTIONS as PW, payload as pw_payload  # noqa: E402
import sr_audit  # noqa: E402
from sr_contract import INSTRUCTIONS, payload  # noqa: E402
import sr_design as design  # noqa: E402
from test_pp import OWN, TASK  # noqa: E402


def test_sentences_and_payloads():
    for text in design.SENTENCES.values():
        assert not re.search(r"\d", text)
        assert "peer" not in text
    assert INSTRUCTIONS["private"] == PW["private"]
    for name, sentence in design.SENTENCES.items():
        assert INSTRUCTIONS[name] == PW["private"].replace(" Ignore padding.",
                                                          sentence + " Ignore padding.")
    assert pw_design.CHECK.replace("the peer's proposal", "your proposal") == design.SELF_CHECK
    assert pw_design.INDEPENDENT.startswith(design.SELF_INDEP[:-1] + ";")
    assert pw_design.INDEPENDENT.replace("the peer", "your earlier proposal") == design.SELF_DOUBT
    base = pw_payload(TASK, OWN, None, "private")
    for name in design.CONDITIONS:
        value = payload(TASK, OWN, name)
        assert {k: v for k, v in value.items() if k != "instruction"} == {
            k: v for k, v in base.items() if k != "instruction"}
    assert payload(TASK, OWN, "private") == base


def test_seeds_are_study3s():
    assert design.recipient_seed("t", 1) == pw_design.recipient_seed("t", 1)
    jobs = design.jobs_for("t")
    assert len(jobs) == 4 * design.REPLICATES
    assert sorted(design.job_order("t", jobs)) == sorted(jobs)


def test_analysis_on_synthetic_answers(monkeypatch):
    monkeypatch.setattr(sr_audit, "paired", lambda a, b, k: {
        "estimate": float(sum(x - y for x, y in zip(a, b))) / len(a),
        "sign_flip_exact_p": 1.0, "bootstrap": {"ci95": [-1, 1], "ci90": [-1, 1]}})
    labels = {"a": 1, "b": 2}
    own = {"a": 1, "b": 0}
    final = {t: {"private": [labels[t], 9], "self_check": [labels[t]] * 2,
                 "self_indep": [9, 9], "self_doubt": [labels[t], None]} for t in labels}
    s3 = {t: {"private": [labels[t], 9], "C_full_check": [labels[t]] * 2, "E_full_check": [9, 9],
              "C_full_indep": [labels[t]] * 2, "E_full_indep": [labels[t]] * 2,
              "C_full": [labels[t]] * 2, "E_full": [9, 9]} for t in labels}
    r = sr_audit.analyze(final, s3, labels, own)
    assert r["accuracy"]["self_check"]["rate"] == 1 and r["accuracy"]["private"]["rate"] == 0.5
    assert r["primary"]["R1_self_check_minus_private"]["estimate"] == 0.5
    assert r["primary"]["R3_net_full_check_minus_self_check"]["estimate"] == -0.5
    assert r["primary"]["R4_net_full_indep_minus_self_indep"]["estimate"] == 1
    assert r["private_replay_identical"]["rate"] == 1
    assert r["invalid_generations"] == {"self_doubt": 2}
    assert r["corruption_initially_right"]["self_indep"]["rate"] == 1
    assert r["rescue_initially_wrong"]["self_check"]["rate"] == 1
    assert r["changed_answer"]["self_check"]["rate"] == 0.5
