"""Integer-answer output contract; replaces kv_task's signed string pattern.

Under `answer: string ^-?[0-9]+$` Qwen2.5-7B often emitted the merged token `"-`,
negating correct answers (reports/peer-strength-matched-context-20261117-a).
"""
MAX_RATIONALE_CHARS = 1200


def output_schema():
    return {
        "type": "object",
        "properties": {
            "rationale": {"type": "string", "maxLength": MAX_RATIONALE_CHARS},
            "answer": {"type": "integer"},
        },
        "required": ["rationale", "answer"],
        "additionalProperties": False,
    }


def validate_candidate(value):
    """Return the candidate in generation field order; JSON booleans are not integers."""
    if not isinstance(value, dict) or set(value) != {"rationale", "answer"}:
        raise ValueError("candidate must contain exactly rationale and answer")
    if (not isinstance(value["rationale"], str)
            or len(value["rationale"]) > MAX_RATIONALE_CHARS):
        raise ValueError("invalid rationale")
    if type(value["answer"]) is not int:
        raise ValueError("invalid integer answer")
    return {"rationale": value["rationale"], "answer": value["answer"]}


def answer_value(candidate):
    return validate_candidate(candidate)["answer"]


# (serialized value, expected acceptance) for the installed-grammar gate.
GRAMMAR_CASES = (
    ({"rationale": "work", "answer": 12}, True),
    ({"rationale": "work", "answer": -7}, True),
    ({"answer": 12, "rationale": "work"}, False),
    ({"rationale": "work", "answer": "12"}, False),
    ({"rationale": "work", "answer": 12.5}, False),
    ({"rationale": "x" * (MAX_RATIONALE_CHARS + 1), "answer": 12}, False),
    ({"rationale": "work", "answer": 12, "extra": 1}, False),
)
