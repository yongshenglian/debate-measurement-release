"""Editor contract, recipient payloads and instruction texts; contains no labels."""
import copy
import json

from kv_task import recipient_messages
from pc_contract import MAX_RATIONALE_CHARS, validate_candidate
from pc_task import INSTRUCTIONS as PC_INSTRUCTIONS, pad_to

MAX_DESCRIPTION_CHARS = 300

INSTRUCTIONS = {
    "private": PC_INSTRUCTIONS["private"],
    "full": ("Recheck the problem and submit your final answer. One unverified peer proposal "
             "for this problem is supplied; verify it rather than following it automatically. "
             "Ignore padding."),
    "answer": ("Recheck the problem and submit your final answer. One unverified peer final "
               "answer for this problem is supplied without its reasoning; verify it rather "
               "than following it automatically. Ignore padding."),
    "locate": ("Recheck the problem. One unverified peer proposal for this problem is supplied. "
               "Before deciding, find the first step where your proposal and the peer's differ, "
               "if any, recompute that step yourself from the problem statement, then submit "
               "your final answer. Ignore padding."),
}
MESSAGE_CONDITION = {"private": "private", "full": "peer", "answer": "peer_answer_only",
                     "locate": "peer_locate"}

EDIT_SYSTEM = (
    "You edit worked solutions to math word problems. Return JSON with error_description, "
    "then the edited rationale, then the integer final answer that the edited rationale "
    "arrives at.")
EDIT_TASKS = {
    "C": ("Copy-edit the solution lightly for clarity. Do not change any number, operation or "
          "step, and keep the final answer. Set error_description to \"none\"."),
    "E": ("Introduce exactly one plausible mistake at one step: an arithmetic slip, a misread "
          "quantity, or the wrong operation. Carry it through consistently so that every later "
          "step and the final answer follow from the mistaken step. Keep all other text as close "
          "to verbatim as possible. Do not mention or hint at the mistake in the rationale. "
          "Describe the mistake briefly in error_description."),
}


def editor_schema():
    return {
        "type": "object",
        "properties": {
            "error_description": {"type": "string", "maxLength": MAX_DESCRIPTION_CHARS},
            "rationale": {"type": "string", "maxLength": MAX_RATIONALE_CHARS},
            "answer": {"type": "integer"},
        },
        "required": ["error_description", "rationale", "answer"],
        "additionalProperties": False,
    }


def validate_edit(value):
    if not isinstance(value, dict) or set(value) != {"error_description", "rationale", "answer"}:
        raise ValueError("edit must contain exactly error_description, rationale and answer")
    if (not isinstance(value["error_description"], str)
            or len(value["error_description"]) > MAX_DESCRIPTION_CHARS):
        raise ValueError("invalid error_description")
    candidate = validate_candidate({"rationale": value["rationale"], "answer": value["answer"]})
    return {"error_description": value["error_description"], **candidate}


EDITOR_GRAMMAR_CASES = (
    ({"error_description": "none", "rationale": "work", "answer": 12}, True),
    ({"error_description": "x", "rationale": "work", "answer": -3}, True),
    ({"rationale": "work", "error_description": "x", "answer": 12}, False),
    ({"error_description": "x", "rationale": "work", "answer": "12"}, False),
    ({"error_description": "x" * (MAX_DESCRIPTION_CHARS + 1), "rationale": "w", "answer": 1}, False),
    ({"error_description": "x", "rationale": "work", "answer": 1, "extra": 1}, False),
)


def edit_messages(task, base, twin):
    if set(task) != {"id", "question"}:
        raise ValueError("public task must contain only id and question")
    base = validate_candidate(base)
    payload = {"question": task["question"], "solution": base, "task": EDIT_TASKS[twin]}
    return [{"role": "system", "content": EDIT_SYSTEM},
            {"role": "user", "content": json.dumps(payload, separators=(",", ":"))}]


def payload(task, own, peer, presentation):
    """Same field layout as kv_task.recipient_payload, with at most one peer."""
    if (presentation == "private") != (peer is None):
        raise ValueError("private has no peer; other presentations have one")
    messages = []
    if peer is not None:
        peer = validate_candidate({"rationale": peer["rationale"], "answer": peer["answer"]})
        messages = [{"answer": peer["answer"]} if presentation == "answer" else peer]
    return {
        "task_id": task["id"],
        "question": task["question"],
        "your_frozen_proposal": validate_candidate(own),
        "message_condition": MESSAGE_CONDITION[presentation],
        "unverified_messages": messages,
        "instruction": INSTRUCTIONS[presentation],
        "padding": "",
    }


def matched_messages(payloads, token_count, max_padding=4096):
    """Pad every payload of one task to one exact token count."""
    if any(value["padding"] != "" for value in payloads.values()):
        raise ValueError("payloads must start with empty padding")
    target = max(token_count(recipient_messages(value)) for value in payloads.values())
    messages = {name: recipient_messages(pad_to(value, target, token_count, max_padding))
                for name, value in payloads.items()}
    if any(token_count(value) != target for value in messages.values()):
        raise ValueError("token matching replay failed")
    return copy.deepcopy(messages), target
