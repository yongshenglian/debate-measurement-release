from audit_single import audit_rows


def test_audit_rows_recomputes_correctness_and_empty_extractions():
    rows = [
        {"0": {"answer": 2, "responses": ["x"], "final_answers": [2.0],
               "final_answer_iscorr": [True], "debate_answer": 2.0,
               "debate_answer_iscorr": True}},
        {"0": {"answer": 3, "responses": ["y"], "final_answers": [""],
               "final_answer_iscorr": [False], "debate_answer": "",
               "debate_answer_iscorr": False}},
    ]
    result = audit_rows(rows, [2, 3])
    assert result == {"correct": 1, "empty_extractions": 1, "problems": []}


def test_audit_rows_rejects_inconsistent_score():
    rows = [
        {"0": {"answer": 2, "responses": ["x"], "final_answers": [2.0],
               "final_answer_iscorr": [False], "debate_answer": 2.0,
               "debate_answer_iscorr": True}},
    ]
    result = audit_rows(rows, [2])
    assert result["problems"] == ["row 0: individual correctness"]
