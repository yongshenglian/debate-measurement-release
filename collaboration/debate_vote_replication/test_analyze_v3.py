import json

from analyze_v3 import bootstrap_difference, summarize, validate


def write_arm(root, arm, correctness, rounds, outputs, responses=None):
    responses = outputs if responses is None else responses
    folder = root / arm / "out" / "history"
    folder.mkdir(parents=True)
    rows = []
    for index, values in enumerate(correctness):
        row = {}
        for round_name, correct in zip(rounds, values):
            row[round_name] = {
                "answer": index,
                "debate_answer_iscorr": correct,
                "final_answer_iscorr": [correct] * outputs,
                "final_answers": [index if correct else ""] * outputs,
                "responses": ["response"] * responses,
            }
        rows.append(row)
    (folder / "history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def fixture(root):
    values = [
        [False, True, True, True, True, True],
        [True, False, True, True, True, True],
        [True, True, True, True, True, True],
        [False, False, False, False, False, False],
    ]
    rounds = [str(i) for i in range(6)]
    for arm in ("dense", "sparse"):
        write_arm(root, arm, values, rounds, 5)
    write_arm(root, "central", values, rounds, 1, 5)
    write_arm(root, "single", [[row[0]] for row in values], ["0"], 1)


def test_protocol_shapes_and_transitions(tmp_path):
    fixture(tmp_path)
    value = summarize(tmp_path, 4)
    transitions = value["arms"]["dense"]["rounds"]["1"]["communication_transitions"]
    assert transitions == {
        "wrong_to_right": 1,
        "right_to_wrong": 1,
        "right_to_right": 1,
        "wrong_to_wrong": 1,
    }
    assert value["arms"]["central"]["rounds"]["0"]["individual_total"] == 4
    assert "round0_vote_minus_round0_individual" not in value["contrasts"]["central"]


def test_bootstrap_preserves_pairs():
    value = bootstrap_difference([(False, True)] * 8, draws=100, seed=1)
    assert value == {"difference": 1.0, "ci95": [1.0, 1.0]}


def test_rejects_wrong_central_output_shape(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "central" / "out" / "history" / "history.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["0"]["final_answers"].append(0)
    histories = {
        arm: [json.loads(line) for line in (tmp_path / arm / "out" / "history" / "history.jsonl").read_text().splitlines()]
        for arm in ("dense", "sparse", "central", "single")
    }
    histories["central"] = rows
    try:
        validate(histories, 4)
    except ValueError as error:
        assert "central row 0 round 0" in str(error)
    else:
        raise AssertionError("invalid centralized output shape was accepted")
