"""Pure deterministic design for the peer-warning study (recipient-side warning vs peer hedge)."""
import hashlib
import json
import random

from pp_design import (ADV_FILE, ADV_REVISION, ADV_SHA256, CONFIDENT, HEDGED,  # noqa: F401
                       MAX_PROMPT_TOKENS, MAX_RATIONALE_CHARS, MAX_UNMATCHED_FRACTION, N_GPUS,
                       adv_pick, canonical, flag_E, numbers)

VERSION = "peer-warning-v1.1"
SOURCE_RUN = "peer-discrimination-20261118-a"
REPLICATES = 2          # recipient samples per Gemma-arm condition (seed shared across conditions)
ADV_REPLICATES = 1      # gsm8k_adv conditions use replicate 0 only

# Recipient-side sentences, inserted before "Ignore padding." in the v1 peer instruction.
# Digit-free (asserted in tests).
WARNING = " The peer's proposal may contain a mistake in one of its steps."
CHECK = " Check each step of the peer's proposal yourself before you answer."
# User-proposed (2026-11-19): resist persuasion rather than flag a possible mistake.
INDEPENDENT = (" Take a fresh look at the work and use your own judgement; do not be misled by"
               " the peer, however convincing it may sound.")

# condition -> (peer source, peer variant, recipient instruction); source None = no peer.
GEMMA_CONDITIONS = {
    "private": (None, None, "private"),
    "C_full": ("C", "full", "neutral"), "E_full": ("E", "full", "neutral"),
    "C_hedged": ("C", "hedged", "neutral"), "E_hedged": ("E", "hedged", "neutral"),
    "C_confident": ("C", "confident", "neutral"), "E_confident": ("E", "confident", "neutral"),
    "C_full_warn": ("C", "full", "warn"), "E_full_warn": ("E", "full", "warn"),
    "C_hedged_warn": ("C", "hedged", "warn"), "E_hedged_warn": ("E", "hedged", "warn"),
    "C_confident_warn": ("C", "confident", "warn"), "E_confident_warn": ("E", "confident", "warn"),
    "C_full_check": ("C", "full", "check"), "E_full_check": ("E", "full", "check"),
    "C_full_indep": ("C", "full", "indep"), "E_full_indep": ("E", "full", "indep"),
    "C_confident_indep": ("C", "confident", "indep"),
    "E_confident_indep": ("E", "confident", "indep"),
}
ADV_CONDITIONS = {
    "adv_C_full": ("advC", "full", "neutral"), "adv_E_full": ("advE", "full", "neutral"),
    "adv_C_full_warn": ("advC", "full", "warn"), "adv_E_full_warn": ("advE", "full", "warn"),
}
CONDITIONS = {**GEMMA_CONDITIONS, **ADV_CONDITIONS}


def seed_for(*parts):
    payload = json.dumps((VERSION,) + parts, separators=(",", ":"), sort_keys=True)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)


def gpu_for(position):
    return position % N_GPUS


def recipient_seed(task_id, replicate):
    """One seed per (task, replicate), shared by every condition of that replicate."""
    return seed_for(task_id, "recipient-generation", replicate)


def conditions_for(record):
    names = []
    if record.get("gemma"):
        names += list(GEMMA_CONDITIONS)
    if record.get("adv"):
        names += (["private"] if "private" not in names else []) + list(ADV_CONDITIONS)
    return names


def jobs_for(record):
    """(condition, replicate) pairs run for one task."""
    gemma = bool(record.get("gemma"))
    out = []
    for name in conditions_for(record):
        in_gemma = name in GEMMA_CONDITIONS and gemma
        for replicate in range(REPLICATES if in_gemma else ADV_REPLICATES):
            out.append((name, replicate))
    return out


def job_order(task_id, jobs):
    return random.Random(seed_for(task_id, "job-order")).sample(jobs, len(jobs))
