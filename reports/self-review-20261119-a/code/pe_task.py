"""Recipient payloads for the five conditions; contains no hidden labels."""
import copy

from kv_task import (_pad_payload_to, recipient_messages, recipient_payload,
                     validate_candidate)
import pe_design as design

ANSWER_ONLY_INSTRUCTION = (
    "Recheck the problem and submit your final answer. Four unverified peer final "
    "answers for this problem are supplied without their reasoning; verify them rather "
    "than following them automatically. Ignore padding."
)


def answer_only_payload(task, own, peers):
    """Same field order as kv_task.recipient_payload; peers reduced to answers."""
    if len(peers) != 4:
        raise ValueError("answer_only requires 4 messages")
    validate_candidate(own)
    answers = [{"answer": validate_candidate(peer)["answer"]} for peer in peers]
    return {
        "task_id": task["id"],
        "question": task["question"],
        "your_frozen_proposal": own,
        "message_condition": "answer_only",
        "unverified_messages": answers,
        "instruction": ANSWER_ONLY_INSTRUCTION,
        "padding": "",
    }


def condition_payloads(task, own, relevant, shuffled):
    return {
        "private": recipient_payload(task, own, [], "private"),
        "relevant": recipient_payload(task, own, relevant, "relevant"),
        "answer_only": answer_only_payload(task, own, relevant),
        "shuffled": recipient_payload(task, own, shuffled, "shuffled"),
    }


def matched_messages(payloads, token_count, max_padding=4096):
    """Pad the four matched payloads to one exact token count; add unpadded private.

    Returns (messages by condition, matched target, unpadded private count).
    Raises ValueError when exact matching is impossible.
    """
    if set(payloads) != set(design.MATCHED):
        raise ValueError("expected the four matched payloads")
    if any(payload["padding"] != "" for payload in payloads.values()):
        raise ValueError("payloads must start with empty padding")
    target = max(token_count(recipient_messages(p)) for p in payloads.values())
    padded = {kind: _pad_payload_to(payload, target, token_count, max_padding)
              for kind, payload in payloads.items()}
    messages = {kind: recipient_messages(payload) for kind, payload in padded.items()}
    if any(token_count(value) != target for value in messages.values()):
        raise ValueError("token matching replay failed")
    messages["private_unpadded"] = recipient_messages(copy.deepcopy(payloads["private"]))
    return messages, target, token_count(messages["private_unpadded"])
