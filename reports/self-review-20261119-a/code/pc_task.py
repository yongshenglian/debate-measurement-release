"""Recipient payloads for the seven conditions; contains no hidden labels.

Instruction texts and payload layout are those of kv_task.recipient_payload and
pe_task.answer_only_payload; only the answer type (integer) differs.
"""
import copy

from kv_task import _pad_payload_to, recipient_messages
from pc_contract import validate_candidate
import pc_design as design

INSTRUCTIONS = {
    "private": (
        "Recheck the problem independently and submit your final answer. "
        "No peer proposals are supplied. Ignore padding."
    ),
    "relevant": (
        "Recheck the problem and submit your final answer. Four unverified peer "
        "proposals for this problem are supplied; verify them rather than following "
        "them automatically. Ignore padding."
    ),
    "answer_only": (
        "Recheck the problem and submit your final answer. Four unverified peer final "
        "answers for this problem are supplied without their reasoning; verify them rather "
        "than following them automatically. Ignore padding."
    ),
    "shuffled": (
        "Recheck the problem and submit your final answer. Four unverified proposals "
        "from a different problem are supplied as an irrelevant control; do not use "
        "their content. Ignore padding."
    ),
}
PADDING_WINDOW = 8


def payload(task, own, peers, kind):
    if kind not in INSTRUCTIONS:
        raise ValueError("unknown message condition")
    if len(peers) != (0 if kind == "private" else 4):
        raise ValueError(f"{kind} requires {0 if kind == 'private' else 4} messages")
    peers = [validate_candidate(peer) for peer in peers]
    if kind == "answer_only":
        peers = [{"answer": peer["answer"]} for peer in peers]
    return {
        "task_id": task["id"],
        "question": task["question"],
        "your_frozen_proposal": validate_candidate(own),
        "message_condition": kind,
        "unverified_messages": peers,
        "instruction": INSTRUCTIONS[kind],
        "padding": "",
    }


def condition_payloads(task, own, qwen_peers, shuffled_peers, strong_peers):
    """Both peer sources use identical wording; only the proposals differ."""
    return {
        "private": payload(task, own, [], "private"),
        "qwen_full": payload(task, own, qwen_peers, "relevant"),
        "qwen_answers": payload(task, own, qwen_peers, "answer_only"),
        "shuffled": payload(task, own, shuffled_peers, "shuffled"),
        "strong_full": payload(task, own, strong_peers, "relevant"),
        "strong_answers": payload(task, own, strong_peers, "answer_only"),
    }


def pad_to(value, target, token_count, max_padding=4096):
    """Exact token-count padding: try sizes near the expected one, then a full scan."""
    current = token_count(recipient_messages(value))
    if current == target:
        return copy.deepcopy(value)
    if current > target:
        raise ValueError("payload already exceeds token target")
    guess = target - current
    sizes = sorted(range(max(0, guess - PADDING_WINDOW),
                         min(max_padding, guess + PADDING_WINDOW) + 1),
                   key=lambda size: (abs(size - guess), size))
    for prefix in ("", " "):
        for size in sizes:
            padded = copy.deepcopy(value)
            padded["padding"] = prefix + " x" * size
            if token_count(recipient_messages(padded)) == target:
                return padded
    return _pad_payload_to(value, target, token_count, max_padding)


def matched_messages(payloads, token_count, max_padding=4096):
    """Pad the six matched payloads to one exact count; add the unpadded private prompt.

    Returns (messages by condition, matched target, unpadded private count).
    Raises ValueError when exact matching is impossible.
    """
    if set(payloads) != set(design.MATCHED):
        raise ValueError("expected the six matched payloads")
    if any(value["padding"] != "" for value in payloads.values()):
        raise ValueError("payloads must start with empty padding")
    target = max(token_count(recipient_messages(value)) for value in payloads.values())
    messages = {kind: recipient_messages(pad_to(value, target, token_count, max_padding))
                for kind, value in payloads.items()}
    if any(token_count(value) != target for value in messages.values()):
        raise ValueError("token matching replay failed")
    messages["private_unpadded"] = recipient_messages(copy.deepcopy(payloads["private"]))
    return messages, target, token_count(messages["private_unpadded"])
