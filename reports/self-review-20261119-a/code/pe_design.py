"""Pure deterministic design for the matched-context peer-evidence study."""
import hashlib
import json
import random


VERSION = "peer-evidence-matched-context-v1"
N_TASKS = 640
FIRST_POSITION = 396
RESERVED_POSITIONS = range(1036, 1319)
N_AGENTS = 5
N_GPUS = 4
SHUFFLE_OFFSET = 13
CONDITIONS = ("private_unpadded", "private", "relevant", "answer_only", "shuffled")
MATCHED = ("private", "relevant", "answer_only", "shuffled")
MAX_INELIGIBLE_FRACTION = 0.05
MAX_UNMATCHED_FRACTION = 0.02


def seed_for(*parts):
    payload = json.dumps((VERSION,) + parts, separators=(",", ":"), sort_keys=True)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)


def task_ids():
    """Pinned GSM8K test shuffled positions 396..1035, disjoint from 0..395."""
    return [f"gsm8k-shuffled-{index}"
            for index in range(FIRST_POSITION, FIRST_POSITION + N_TASKS)]


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


def shuffled_task(task_id, eligible):
    """Ring offset over eligible tasks in frozen order; never the task itself."""
    ordered = [task for task in task_ids() if task in set(eligible)]
    if task_id not in ordered or len(ordered) <= SHUFFLE_OFFSET:
        raise ValueError("task not eligible or ring too small")
    return ordered[(ordered.index(task_id) + SHUFFLE_OFFSET) % len(ordered)]


def condition_order(task_id):
    return random.Random(seed_for(task_id, "condition-order")).sample(CONDITIONS,
                                                                      len(CONDITIONS))


def proposal_schedule():
    workers = {gpu: [] for gpu in range(N_GPUS)}
    for position, task_id in enumerate(task_ids()):
        for agent in range(N_AGENTS):
            workers[(position * N_AGENTS + agent) % N_GPUS].append({
                "task_id": task_id, "agent": agent,
                "seed": seed_for(task_id, "proposal", agent)})
    return workers
