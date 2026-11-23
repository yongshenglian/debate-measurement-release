"""Recipient instructions, peers and payloads for the peer-warning study (no labels)."""
from pc_contract import validate_candidate
from pd_contract import INSTRUCTIONS as PD_INSTRUCTIONS, matched_messages  # noqa: F401
from pp_contract import peer_for  # noqa: F401
import pw_design as design

_TAIL = " Ignore padding."
if not PD_INSTRUCTIONS["full"].endswith(_TAIL):
    raise ValueError("unexpected v1 peer instruction")
_BASE = PD_INSTRUCTIONS["full"][:-len(_TAIL)]
INSTRUCTIONS = {"private": PD_INSTRUCTIONS["private"],
                "neutral": PD_INSTRUCTIONS["full"],
                "warn": _BASE + design.WARNING + _TAIL,
                "check": _BASE + design.CHECK + _TAIL,
                "indep": _BASE + design.INDEPENDENT + _TAIL}


def payload(task, own, peer, instruction):
    """v1 field layout; only the instruction text differs among peer conditions."""
    if (instruction == "private") != (peer is None):
        raise ValueError("only the private instruction has no peer")
    return {"task_id": task["id"], "question": task["question"],
            "your_frozen_proposal": validate_candidate(own),
            "message_condition": "private" if peer is None else "peer",
            "unverified_messages": [] if peer is None else [validate_candidate(peer)],
            "instruction": INSTRUCTIONS[instruction], "padding": ""}


def peers_for(record):
    """condition -> peer message or None."""
    out = {}
    for name in design.conditions_for(record):
        source, variant, _ = design.CONDITIONS[name]
        if source is None:
            out[name] = None
        elif source in ("advC", "advE"):
            out[name] = peer_for(record["adv"][source[-1]], variant)
        else:
            out[name] = peer_for(record["gemma"][source], variant)
    return out
