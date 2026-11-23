import json

from analyze_v2 import bootstrap_difference, summarize


def write_arm(root, arm, correctness, rounds, agents):
    folder = root / arm / "out" / "history"
    folder.mkdir(parents=True)
    rows = []
    for index, values in enumerate(correctness):
        row = {}
        for round_name, correct in zip(rounds, values):
            row[round_name] = {
                "answer": index,
                "debate_answer_iscorr": correct,
                "final_answer_iscorr": [correct] * agents,
                "final_answers": [index if correct else ""] * agents,
                "responses": ["response"] * agents,
            }
        rows.append(row)
    (folder / "history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_summary_validates_protocol_and_transitions(tmp_path):
    six_round_values = [
        [False, True, True, True, True, True],
        [True, False, True, True, True, True],
        [True, True, True, True, True, True],
        [False, False, False, False, False, False],
    ]
    for arm in ("dense", "sparse", "central"):
        write_arm(tmp_path, arm, six_round_values, [str(i) for i in range(6)], 5)
    write_arm(tmp_path, "single", [[values[0]] for values in six_round_values], ["0"], 1)
    value = summarize(tmp_path, 4)
    transitions = value["arms"]["dense"]["rounds"]["1"]["communication_transitions"]
    assert transitions == {
        "wrong_to_right": 1,
        "right_to_wrong": 1,
        "right_to_right": 1,
        "wrong_to_wrong": 1,
    }
    assert value["arms"]["dense"]["rounds"]["1"]["paired_effect"]["difference"] == 0


def test_bootstrap_preserves_pairs():
    value = bootstrap_difference([(False, True)] * 8, draws=100, seed=1)
    assert value == {"difference": 1.0, "ci95": [1.0, 1.0]}
