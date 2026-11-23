"""Pure deterministic design for the clean integer-contract peer study."""
import hashlib
import json
import random

VERSION = "peer-clean-integer-v1"
# Which GSM8K shuffled-test positions to use: (first position, count).
TASK_SETS = {"main": (396, 640), "reserved": (1036, 283)}
TASK_SET = "reserved"
N_AGENTS = 5
N_GPUS = 4
SHUFFLE_OFFSET = 13
FAMILIES = ("gemma", "mistral")
MODEL_DIRS = {"gemma": "gemma-3-27b-it-w4a16", "mistral": "mistral-small-3.1-24b-w4a16"}
STRONG_SLOTS = (("gemma", 0), ("gemma", 1), ("mistral", 0), ("mistral", 1))
# GPU -> (family, task-position parity) for the strong proposal phase.
STRONG_LAYOUT = {0: ("gemma", 0), 1: ("gemma", 1), 2: ("mistral", 0), 3: ("mistral", 1)}
CONDITIONS = ("private_unpadded", "private", "qwen_full", "qwen_answers", "shuffled",
              "strong_full", "strong_answers")
MATCHED = CONDITIONS[1:]
PEER_SOURCE = {"private_unpadded": None, "private": None, "qwen_full": "qwen",
               "qwen_answers": "qwen", "shuffled": "shuffled", "strong_full": "strong",
               "strong_answers": "strong"}
MAX_INELIGIBLE_FRACTION = 0.05
MAX_UNMATCHED_FRACTION = 0.02


def seed_for(*parts):
    payload = json.dumps((VERSION,) + parts, separators=(",", ":"), sort_keys=True)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)


def task_ids(task_set=None):
    first, count = TASK_SETS[task_set or TASK_SET]
    return [f"gsm8k-shuffled-{index}" for index in range(first, first + count)]


def gpu_for(task_id):
    return task_ids().index(task_id) % N_GPUS


def recipient_index(task_id):
    return seed_for(task_id, "recipient-index") % N_AGENTS


def recipient_seed(task_id):
    return seed_for(task_id, "recipient-generation")


def peer_indices(recipient):
    if recipient not in range(N_AGENTS):
        raise ValueError("invalid recipient")
    return [index for index in range(N_AGENTS) if index != recipient]


def slot_key(family, slot):
    return f"{family}{slot}"


def shuffled_task(task_id, eligible):
    """Ring offset over eligible tasks in frozen order; never the task itself."""
    ordered = [task for task in task_ids() if task in set(eligible)]
    if task_id not in ordered or len(ordered) <= SHUFFLE_OFFSET:
        raise ValueError("task not eligible or ring too small")
    return ordered[(ordered.index(task_id) + SHUFFLE_OFFSET) % len(ordered)]


def strong_order(task_id):
    """Label-blind presentation order of the four strong peers."""
    return random.Random(seed_for(task_id, "strong-order")).sample(STRONG_SLOTS, 4)


def condition_order(task_id):
    return random.Random(seed_for(task_id, "condition-order")).sample(CONDITIONS,
                                                                      len(CONDITIONS))


def qwen_schedule():
    workers = {gpu: [] for gpu in range(N_GPUS)}
    for position, task_id in enumerate(task_ids()):
        for agent in range(N_AGENTS):
            workers[(position * N_AGENTS + agent) % N_GPUS].append({
                "task_id": task_id, "family": "qwen", "agent": agent,
                "seed": seed_for(task_id, "qwen-proposal", agent)})
    return workers


def strong_schedule():
    workers = {}
    for gpu, (family, parity) in STRONG_LAYOUT.items():
        workers[gpu] = [{"task_id": task, "family": family, "slot": slot,
                         "seed": seed_for(task, "strong-proposal", family, slot)}
                        for position, task in enumerate(task_ids()) if position % 2 == parity
                        for slot in (0, 1)]
    return workers
