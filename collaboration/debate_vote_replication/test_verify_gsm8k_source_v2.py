from verify_gsm8k_source_v2 import extract_answer, questions_from_log, QUESTION_SUFFIX


def test_extract_answer_matches_author_format():
    assert extract_answer("work\n#### 1,234") == 1234
    assert extract_answer("work\n#### -7") == -7


def test_question_log_parser_ignores_embedded_prompt_text():
    text = (
        "noise\nQuestion:  First question" + QUESTION_SUFFIX + " \n"
        + "\ufffd\ufffdQuestion:  Progress-prefixed question" + QUESTION_SUFFIX + "\n"
        "{'content': 'Question: not a top-level line'}\n"
        "Question:  Second question" + QUESTION_SUFFIX + "\n"
    )
    assert questions_from_log(text) == [
        "First question",
        "Progress-prefixed question",
        "Second question",
    ]
