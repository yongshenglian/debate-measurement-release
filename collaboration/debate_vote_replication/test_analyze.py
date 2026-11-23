import json

from analyze import bootstrap_difference, summarize


def write_arm(root, arm, correctness):
    folder = root / arm / "out" / "history"
    folder.mkdir(parents=True)
    rows = []
    for index, (before, after) in enumerate(correctness):
        rounds = {}
        for name, correct in (("0", before), ("1", after)):
            rounds[name] = {"answer": index, "debate_answer_iscorr": correct,
                            "final_answer_iscorr": [correct],
                            "final_answers": [index if correct else ""]}
        rows.append(rounds)
    (folder / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_summary_uses_paired_transitions(tmp_path):
    pairs = [(False, True), (True, False), (True, True), (False, False)]
    for arm in ("dense", "sparse", "central", "single"):
        write_arm(tmp_path, arm, pairs)
    value = summarize(tmp_path, 4)
    transitions = value["arms"]["dense"]["rounds"]["1"]["communication_transitions"]
    assert transitions == {"wrong_to_right": 1, "right_to_wrong": 1,
                           "right_to_right": 1, "wrong_to_wrong": 1}
    assert value["arms"]["dense"]["rounds"]["1"]["paired_effect"]["difference"] == 0


def test_bootstrap_preserves_pairs():
    value = bootstrap_difference([(False, True)] * 8, draws=100, seed=1)
    assert value == {"difference": 1.0, "ci95": [1.0, 1.0]}
