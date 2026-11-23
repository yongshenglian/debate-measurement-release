from log_questions import questions_from_log, QUESTION_SUFFIX


def test_parses_plain_and_progress_prefixed_questions():
    text = (
        "Question:  First" + QUESTION_SUFFIX + "\n"
        "\ufffd\ufffdQuestion:  Second" + QUESTION_SUFFIX + "\n"
        "{'content': 'Question: not a top-level record'}\n"
    )
    assert questions_from_log(text) == ["First", "Second"]


def test_ignores_nonbenchmark_question_marker():
    assert questions_from_log("response says Question:  maybe") == []
