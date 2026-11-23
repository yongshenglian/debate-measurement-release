"""Self-review instructions and payloads (Study 3's private layout; no labels, no peer)."""
from pw_contract import INSTRUCTIONS as PW_INSTRUCTIONS, payload as pw_payload
import sr_design as design

_TAIL = " Ignore padding."
if not PW_INSTRUCTIONS["private"].endswith(_TAIL):
    raise ValueError("unexpected private instruction")
_BASE = PW_INSTRUCTIONS["private"][:-len(_TAIL)]
INSTRUCTIONS = {name: _BASE + sentence + _TAIL for name, sentence in design.SENTENCES.items()}


def payload(task, own, name):
    """Study 3's private payload with only the instruction text changed."""
    value = pw_payload(task, own, None, "private")
    value["instruction"] = INSTRUCTIONS[name]
    return value
