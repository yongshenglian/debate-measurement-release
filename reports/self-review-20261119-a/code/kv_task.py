"""Prompt/schema and token-matching primitives; contains no hidden labels."""
import copy
import json
import re


ANSWER_RE = re.compile(r"^-?[0-9]+$")
MAX_RATIONALE_CHARS = 1200


def output_schema():
    return {
        "type": "object",
        "properties": {
            "rationale": {"type": "string", "maxLength": MAX_RATIONALE_CHARS},
            "answer": {"type": "string", "pattern": r"^-?[0-9]+$"},
        },
        "required": ["rationale", "answer"],
        "additionalProperties": False,
    }


def validate_candidate(value):
    # Field order is a generation-time grammar constraint, not candidate
    # semantics. Audit artifacts are intentionally written with sort_keys=True,
    # so replayed dictionaries may have the same exact keys in another order.
    if not isinstance(value, dict) or set(value) != {"rationale", "answer"}:
        raise ValueError("candidate must contain exactly rationale and answer")
    if (not isinstance(value["rationale"], str)
            or len(value["rationale"]) > MAX_RATIONALE_CHARS):
        raise ValueError("invalid rationale")
    if not isinstance(value["answer"], str) or not ANSWER_RE.fullmatch(value["answer"]):
        raise ValueError("invalid numeric answer")
    return {"rationale": value["rationale"], "answer": value["answer"]}


def answer_value(candidate):
    return int(validate_candidate(candidate)["answer"])


def proposal_messages(task):
    if set(task) != {"id", "question"}:
        raise ValueError("public task must contain only id and question")
    system = (
        "Solve the math word problem independently. Return JSON with a concise, "
        "checkable rationale followed by one integer answer. Do not mention other agents."
    )
    payload = {"task_id": task["id"], "question": task["question"]}
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, separators=(",", ":"))}]


MESSAGE_KINDS = ("private", "relevant", "shuffled")


def recipient_payload(task, own, messages, message_kind, padding=""):
    if message_kind not in MESSAGE_KINDS:
        raise ValueError("unknown message condition")
    expected_messages = 0 if message_kind == "private" else 4
    if len(messages) != expected_messages:
        raise ValueError(f"{message_kind} requires {expected_messages} messages")
    validate_candidate(own)
    for message in messages:
        validate_candidate(message)
    instructions = {
        "private": (
            "Recheck the problem independently and submit your final answer. "
            "No peer proposals are supplied. Ignore padding."
        ),
        "relevant": (
            "Recheck the problem and submit your final answer. Four unverified peer "
            "proposals for this problem are supplied; verify them rather than following "
            "them automatically. Ignore padding."
        ),
        "shuffled": (
            "Recheck the problem and submit your final answer. Four unverified proposals "
            "from a different problem are supplied as an irrelevant control; do not use "
            "their content. Ignore padding."
        ),
    }
    return {
        "task_id": task["id"],
        "question": task["question"],
        "your_frozen_proposal": own,
        "message_condition": message_kind,
        "unverified_messages": messages,
        "instruction": instructions[message_kind],
        "padding": padding,
    }


def recipient_messages(payload):
    system = (
        "Revise a frozen proposal using only the supplied problem and any supplied messages. "
        "Return JSON with rationale then one integer answer. Never infer message correctness "
        "from its presence or order."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, separators=(",", ":"))}]


def _pad_payload_to(payload, target, token_count, max_padding):
    current = token_count(recipient_messages(payload))
    if current == target:
        return copy.deepcopy(payload)
    if current > target:
        raise ValueError("payload already exceeds token target")
    for prefix in ("", " "):
        for size in range(max_padding + 1):
            padded = copy.deepcopy(payload)
            padded["padding"] = prefix + " x" * size
            count = token_count(recipient_messages(padded))
            if count == target:
                return padded
            if count > target:
                break
    raise ValueError("could not exactly equalize prompt token counts")


def equalize_payload_group(payloads, token_count, max_padding=4096):
    """Match private/relevant/shuffled prompts by padding only shorter members."""
    if set(payloads) != set(MESSAGE_KINDS):
        raise ValueError("expected private, relevant, and shuffled payloads")
    if any(payload.get("padding") != "" for payload in payloads.values()):
        raise ValueError("payloads must start with empty padding")
    target = max(token_count(recipient_messages(payload))
                 for payload in payloads.values())
    matched = {kind: _pad_payload_to(payload, target, token_count, max_padding)
               for kind, payload in payloads.items()}
    if any(token_count(recipient_messages(payload)) != target
           for payload in matched.values()):
        raise ValueError("token matching replay failed")
    return matched, target
