"""Pure deterministic design for the peer-persuasion study (what in a peer's working persuades)."""
import hashlib
import json
import random
import re

VERSION = "peer-persuasion-v1"
SOURCE_RUN = "peer-discrimination-20261118-a"
ADV_FILE = "datasets/gsm8k_adv/test.json"
ADV_SHA256 = "ebfa36e87d64e22ef79564a9b0f524c9967516d59cf555b672c704c814a1b3ad"
ADV_REVISION = "987b7e9646a668e9a493ab0ae5b545acb7632def"
N_GPUS = 4
EDITOR_FAMILY = "gemma"
SLOTS = ("C", "E")
EDIT_KINDS = ("key", "asserted")
MAX_EDIT_ATTEMPTS = 2
MAX_RATIONALE_CHARS = 1200

# Digit-free templates (asserted in tests).
CONFIDENT = " I double-checked every step, and this answer is definitely correct."
HEDGED = " I am not completely sure about one of these steps."
STATED = "The answer is {}."
STATED_REPEATS = 5

# condition -> (peer source, variant, presentation); source None = no peer.
GEMMA_CONDITIONS = {
    "private": (None, None, "private"),
    "anchor_C": ("C", "anchor", "anchor"), "anchor_E": ("E", "anchor", "anchor"),
    "C_stated1": ("C", "stated1", "full"), "E_stated1": ("E", "stated1", "full"),
    "E_stated5": ("E", "stated5", "full"),
    "C_full": ("C", "full", "full"), "E_full": ("E", "full", "full"),
    "C_confident": ("C", "confident", "full"), "E_confident": ("E", "confident", "full"),
    "C_hedged": ("C", "hedged", "full"), "E_hedged": ("E", "hedged", "full"),
    "C_key": ("C", "key", "full"), "E_key": ("E", "key", "full"),
    "C_asserted": ("C", "asserted", "full"), "E_asserted": ("E", "asserted", "full"),
}
ADV_CONDITIONS = {
    "adv_C_full": ("advC", "full", "full"), "adv_E_full": ("advE", "full", "full"),
    "adv_E_stated1": ("advE", "stated1", "full"),
}
CONDITIONS = {**GEMMA_CONDITIONS, **ADV_CONDITIONS}

# Stimulus rules.
KEY_MAX_LENGTH_RATIO = 0.6
ADV_MIN_LENGTH_RATIO, ADV_MAX_LENGTH_RATIO = 0.8, 1.25
ADV_MIN_SIMILARITY = 0.6
REVIEW_SAMPLE = 40
REVIEW_MIN_PASS = 32
MAX_UNMATCHED_FRACTION = 0.02
MAX_PROMPT_TOKENS = 4096

# Frozen v1 twin-quality flag (reports/peer-discrimination-20261118-a/exploratory/pd_explore.py).
SELF_FLAG = re.compile(r"\b(error|mistake|incorrect|wrong|must be|should be|doesn't make sense|"
                       r"does not make sense|not possible|impossible|wait)\b", re.I)
DECIMAL = re.compile(r"\d+\.\d+")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def flag_E(e):
    """Mechanical E-twin quality flags (self-flag, unstated rounding, truncation)."""
    text = e["rationale"]
    reasons = []
    if SELF_FLAG.search(text):
        reasons.append("self_flag")
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    last = nums[-1].replace(",", "") if nums else ""
    if DECIMAL.fullmatch(last) and float(last) != float(e["answer"]):
        reasons.append("unstated_rounding")
    if len(text) >= 1190:
        reasons.append("truncated")
    return reasons


def canonical(token):
    value = float(token.replace(",", ""))
    return str(int(value)) if value == int(value) else repr(round(value, 6))


def numbers(text):
    return [canonical(token) for token in NUMBER.findall(text)]


def key_values(c, e):
    """(C value, E value) at the first differing number: the error's location."""
    cn, en = numbers(c["rationale"]), numbers(e["rationale"])
    for a, b in zip(cn, en):
        if a != b:
            return {"C": a, "E": b}
    m = min(len(cn), len(en))
    if len(cn) != len(en):
        return {"C": cn[m] if len(cn) > m else str(c["answer"]),
                "E": en[m] if len(en) > m else str(e["answer"])}
    return {"C": str(c["answer"]), "E": str(e["answer"])}


def seed_for(*parts):
    payload = json.dumps((VERSION,) + parts, separators=(",", ":"), sort_keys=True)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)


def gpu_for(position):
    return position % N_GPUS


def recipient_seed(task_id):
    return seed_for(task_id, "recipient-generation")


def edit_seed(task_id, slot, kind, attempt):
    return seed_for(task_id, "edit", slot, kind, attempt)


def adv_pick(task_id, rows):
    return random.Random(seed_for(task_id, "adv-row")).choice(rows)


def conditions_for(record):
    names = []
    if record.get("gemma"):
        names += list(GEMMA_CONDITIONS)
    if record.get("adv"):
        names += (["private"] if "private" not in names else []) + list(ADV_CONDITIONS)
    return names


def condition_order(task_id, names):
    return random.Random(seed_for(task_id, "condition-order")).sample(names, len(names))


def review_sample(task_ids, kind):
    ordered = sorted(task_ids)
    return sorted(random.Random(seed_for("manual-review", kind)).sample(
        ordered, min(REVIEW_SAMPLE, len(ordered))))
