"""Pure deterministic design for the self-review study: Study 3's sentences with no peer."""
import hashlib
import json
import random

import pw_design as pw

VERSION = "self-review-v1"
SOURCE_RUN = "peer-warning-20261119-b"
REPLICATES = pw.REPLICATES
N_GPUS = pw.N_GPUS
MAX_PROMPT_TOKENS = pw.MAX_PROMPT_TOKENS

# Inserted before " Ignore padding." in the private instruction. Digit-free (asserted in tests).
# Self-directed versions of Study 3's check and independence sentences.
SELF_CHECK = " Check each step of your proposal yourself before you answer."
SELF_INDEP = " Take a fresh look at the work and use your own judgement."
# The user's full sentence, with "the peer" replaced by the reader's own earlier proposal.
SELF_DOUBT = (" Take a fresh look at the work and use your own judgement; do not be misled by"
              " your earlier proposal, however convincing it may sound.")
SENTENCES = {"private": "", "self_check": SELF_CHECK, "self_indep": SELF_INDEP,
             "self_doubt": SELF_DOUBT}
CONDITIONS = tuple(SENTENCES)


def seed_for(*parts):
    payload = json.dumps((VERSION,) + parts, separators=(",", ":"), sort_keys=True)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)


def recipient_seed(task_id, replicate):
    """Study 3's seed for (task, replicate), so every condition shares Study 3's random numbers."""
    return pw.recipient_seed(task_id, replicate)


def jobs_for(task_id):
    return [(name, r) for name in CONDITIONS for r in range(REPLICATES)]


def job_order(task_id, jobs):
    return random.Random(seed_for(task_id, "job-order")).sample(jobs, len(jobs))
