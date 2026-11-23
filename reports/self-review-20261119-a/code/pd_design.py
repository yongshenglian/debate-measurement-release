"""Pure deterministic design for the single-peer discrimination study."""
import hashlib
import json
import random

VERSION = "peer-discrimination-v1"
SOURCE_RUNS = ("peer-clean-integer-20261118-a", "peer-clean-integer-20261118-b")
N_GPUS = 4
EDITOR_FAMILY = "gemma"
TWINS = ("C", "E")
MAX_EDIT_ATTEMPTS = 2
ARM_I = ("private", "C_full", "E_full", "C_answer", "E_answer", "C_locate", "E_locate")
ARM_R = ("RC_full", "RW_full", "RC_locate", "RW_locate")
# condition -> (peer slot, presentation); slot None = no peer.
CONDITIONS = {
    "private": (None, "private"),
    "C_full": ("C", "full"), "E_full": ("E", "full"),
    "C_answer": ("C", "answer"), "E_answer": ("E", "answer"),
    "C_locate": ("C", "locate"), "E_locate": ("E", "locate"),
    "RC_full": ("RC", "full"), "RW_full": ("RW", "full"),
    "RC_locate": ("RC", "locate"), "RW_locate": ("RW", "locate"),
}
MIN_LENGTH_RATIO, MAX_LENGTH_RATIO = 0.8, 1.25
MIN_SIMILARITY = 0.6
REVIEW_SAMPLE = 40
REVIEW_MIN_PASS = 32
MAX_UNMATCHED_FRACTION = 0.02


def seed_for(*parts):
    payload = json.dumps((VERSION,) + parts, separators=(",", ":"), sort_keys=True)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)


def gpu_for(position):
    return position % N_GPUS


def recipient_seed(task_id):
    return seed_for(task_id, "recipient-generation")


def base_order(task_id, keys):
    """Hash-seeded order in which strong proposals are tried as the twin base."""
    return random.Random(seed_for(task_id, "base-order")).sample(sorted(keys), len(keys))


def pick(task_id, role, indices):
    """Hash-seeded choice of one peer index for the real-mistake arm."""
    return random.Random(seed_for(task_id, role)).choice(sorted(indices))


def edit_seed(task_id, twin, attempt):
    return seed_for(task_id, "edit", twin, attempt)


def conditions_for(task):
    """Conditions run for a stimulus record (arm membership decided at freeze)."""
    names = []
    if task.get("twins"):
        names += list(ARM_I)
    if task.get("real"):
        names += (["private"] if "private" not in names else []) + list(ARM_R)
    return names


def condition_order(task_id, names):
    return random.Random(seed_for(task_id, "condition-order")).sample(names, len(names))


def review_sample(task_ids):
    ordered = sorted(task_ids)
    return sorted(random.Random(seed_for("manual-review")).sample(
        ordered, min(REVIEW_SAMPLE, len(ordered))))
