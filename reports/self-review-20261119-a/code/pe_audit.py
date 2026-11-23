"""Independent replay audit and the frozen paired analysis."""
from collections import Counter
import json
from math import comb
from pathlib import Path
import random

from kv_io import save, sha
from kv_task import answer_value
import pe_design as design
from pe_freeze import build_recipient_schedule, freeze_pool, load_events, replay_worker

BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20261117
ALPHA = 0.05
EQUIVALENCE_MARGIN = 0.05
MAX_INVALID_FRACTION = 0.02
PRIMARY = {"P1_relevant_minus_private": ("relevant", "private"),
           "P2_relevant_minus_shuffled": ("relevant", "shuffled")}
SECONDARY = {"S1a_answer_only_minus_private": ("answer_only", "private"),
             "S1b_relevant_minus_answer_only": ("relevant", "answer_only"),
             "S2_private_minus_private_unpadded": ("private", "private_unpadded"),
             "S3_shuffled_minus_private": ("shuffled", "private")}


def bootstrap(values, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                       for _ in range(draws))

    def pick(q):
        return estimates[min(draws - 1, int(q * draws))]
    return {"method": "task-paired percentile bootstrap", "draws": draws, "seed": seed,
            "ci95": [pick(0.025), pick(0.975)], "ci90": [pick(0.05), pick(0.95)]}


def mcnemar_exact(a_only, b_only):
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvalues):
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def contrast(a, b):
    diffs = [x - y for x, y in zip(a, b)]
    a_only, b_only = diffs.count(1), diffs.count(-1)
    return {"estimate": sum(diffs) / len(diffs), "a_only_correct": a_only,
            "b_only_correct": b_only, "mcnemar_exact_p": mcnemar_exact(a_only, b_only),
            "bootstrap": bootstrap(diffs)}


def primary_label(result):
    low, high = result["bootstrap"]["ci90"]
    if result["holm_p"] < ALPHA:
        return "benefit" if result["estimate"] > 0 else "harm"
    if -EQUIVALENCE_MARGIN < low and high < EQUIVALENCE_MARGIN:
        return "equivalent_within_5pp"
    return "inconclusive"


def vote(answers):
    """Plurality; ties go to the answer appearing earliest in agent order."""
    present = [a for a in answers if a is not None]
    counts = Counter(present)
    return max(counts, key=lambda value: (counts[value], -present.index(value)))


def peer_bin(count):
    return "0" if count == 0 else "1-2" if count <= 2 else "3-4"


def analyze(recipient_tasks, labels, pool, outcomes, jobs_by_task):
    """outcomes[task][condition] = candidate dict or None (invalid generation)."""
    n = len(recipient_tasks)
    correct = {c: [] for c in design.CONDITIONS}
    invalid = Counter()
    changed = Counter()
    adopted = Counter()
    vote_revised = Counter()
    transitions = {c: Counter() for c in design.CONDITIONS}
    wrong_by_peers = {c: {b: Counter() for b in ("0", "1-2", "3-4")}
                      for c in design.CONDITIONS}
    initial_correct = vote5_correct = 0
    per_task = {}
    for task in recipient_tasks:
        label = labels[task]
        jobs = jobs_by_task[task]
        recipient = jobs["private"]["recipient"]
        answers = [answer_value(pool[task][str(i)]) for i in range(design.N_AGENTS)]
        own = answers[recipient]
        before = own == label
        peers_correct = sum(answers[i] == label for i in design.peer_indices(recipient))
        initial_correct += before
        vote5_correct += vote(answers) == label
        row = {"initial_correct": before, "peers_correct": peers_correct, "conditions": {}}
        for condition in design.CONDITIONS:
            candidate = outcomes[task][condition]
            final = None if candidate is None else answer_value(candidate)
            ok = final == label
            correct[condition].append(int(ok))
            invalid[condition] += candidate is None
            changed[condition] += final is not None and final != own
            source = jobs[condition]["message_source_task"]
            peer_answers = ([] if source is None else
                            [answer_value(pool[source][str(i)])
                             for i in design.peer_indices(recipient)])
            took = final is not None and final != own and final in peer_answers
            adopted[condition] += took
            revised = list(answers)
            revised[recipient] = final
            vote_revised[condition] += vote(revised) == label
            key = ("right" if before else "wrong") + "_to_" + ("right" if ok else "wrong")
            transitions[condition][key] += 1
            if not before:
                wrong_by_peers[condition][peer_bin(peers_correct)]["n"] += 1
                wrong_by_peers[condition][peer_bin(peers_correct)]["wrong_to_right"] += ok
            row["conditions"][condition] = {"answer": final, "correct": ok,
                                            "valid": candidate is not None,
                                            "peer_answer_adopted": took}
        per_task[task] = row

    primary = {name: contrast(correct[a], correct[b]) for name, (a, b) in PRIMARY.items()}
    adjusted = holm({name: r["mcnemar_exact_p"] for name, r in primary.items()})
    for name, result in primary.items():
        result["holm_p"] = adjusted[name]
        result["label"] = primary_label(result)
    secondary = {name: contrast(correct[a], correct[b]) for name, (a, b) in SECONDARY.items()}
    return {
        "tasks": n,
        "correct": {c: sum(v) for c, v in correct.items()},
        "accuracy": {c: sum(v) / n for c, v in correct.items()},
        "invalid_generations": {c: invalid[c] for c in design.CONDITIONS},
        "invalid_flag_over_2pct": {c: invalid[c] > MAX_INVALID_FRACTION * n
                                   for c in design.CONDITIONS},
        "answer_changed": {c: changed[c] for c in design.CONDITIONS},
        "peer_answer_adopted": {c: adopted[c] for c in design.CONDITIONS},
        "recipient_initial_correct": initial_correct,
        "frozen_pool_vote_correct": vote5_correct,
        "vote_with_revised_recipient_correct": {c: vote_revised[c] for c in design.CONDITIONS},
        "transitions": {c: dict(t) for c, t in transitions.items()},
        "initially_wrong_by_correct_peers": {
            c: {b: dict(v) for b, v in bins.items()} for c, bins in wrong_by_peers.items()},
        "primary": primary, "secondary": secondary, "per_task": per_task,
    }


def audit(run, tokenizer):
    run = Path(run)
    data = run / "data"
    labels = {item["id"]: item["answer"]
              for item in json.loads((run / "hidden" / "hidden.json").read_text())}
    proposal_schedule = json.loads((data / "proposal-schedule.json").read_text())
    events = {gpu: load_events(run / f"proposal-gpu{gpu}") for gpu in range(design.N_GPUS)}
    pool, eligible, ineligible = freeze_pool(proposal_schedule, events, tokenizer)
    frozen = json.loads((data / "proposal-audit.json").read_text())
    if frozen["pool"] != pool or frozen["ineligible_tasks"] != ineligible:
        raise ValueError("re-derived pool differs from frozen proposal audit")
    schedule = json.loads((data / "recipient-schedule.json").read_text())
    if schedule["proposal_audit_sha256"] != sha(data / "proposal-audit.json"):
        raise ValueError("recipient schedule not bound to proposal audit")
    public = json.loads((data / "public.json").read_text())
    rebuilt = build_recipient_schedule(public, pool, eligible, tokenizer)
    if any(rebuilt[key] != schedule[key] for key in rebuilt):
        raise ValueError("re-derived recipient schedule differs from frozen schedule")
    outcomes, jobs_by_task = {}, {}
    for gpu in range(design.N_GPUS):
        jobs = schedule["workers"][str(gpu)]
        candidates = replay_worker("recipient", gpu, jobs,
                                   load_events(run / f"recipient-gpu{gpu}"), tokenizer)
        for job, candidate in zip(jobs, candidates):
            if job["condition"] in outcomes.setdefault(job["task_id"], {}):
                raise ValueError("duplicate task condition")
            outcomes[job["task_id"]][job["condition"]] = candidate
            jobs_by_task.setdefault(job["task_id"], {})[job["condition"]] = job
    tasks = schedule["recipient_tasks"]
    if (set(outcomes) != set(tasks)
            or any(set(v) != set(design.CONDITIONS) for v in outcomes.values())):
        raise ValueError("paired recipient grid incomplete")
    result = analyze(tasks, labels, pool, outcomes, jobs_by_task)
    result.update({"version": design.VERSION, "integrity_passed": True,
                   "eligible_count": len(eligible), "ineligible_tasks": ineligible,
                   "unmatched_tasks": schedule["unmatched_tasks"],
                   "hidden_sha256": sha(run / "hidden" / "hidden.json"),
                   "recipient_schedule_sha256": sha(data / "recipient-schedule.json")})
    return result


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/model", local_files_only=True)
    result = audit(Path("/run"), tokenizer)
    save(Path("/output/audit.json"), result)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("per_task", "ineligible_tasks", "unmatched_tasks")},
                     indent=2))
