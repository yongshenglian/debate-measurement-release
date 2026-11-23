"""Replay both proposal phases, freeze the pools, and build matched recipient prompts.

Label-blind: this step never reads hidden labels.
"""
import json
from pathlib import Path

from kv_io import save, sha
from kv_runtime import prompt_ids
from pc_contract import output_schema
import pc_design as design
from pc_runtime import validate_failed_record, validate_record
from pc_task import condition_payloads, matched_messages

QWEN_SLOTS = tuple(str(agent) for agent in range(design.N_AGENTS))
STRONG_KEYS = tuple(design.slot_key(f, s) for f, s in design.STRONG_SLOTS)


def replay_worker(phase, gpu, jobs, events, tokenizer):
    """Independently replay one worker's events; return candidates (None = invalid)."""
    if len(events) != len(jobs):
        raise ValueError(f"{phase} gpu {gpu} event count mismatch")
    candidates = []
    for index, (job, event) in enumerate(zip(jobs, events)):
        if (event["gpu"] != gpu or event["phase"] != phase or event["call_index"] != index
                or event["job"] != job or event["schema"] != output_schema()
                or event["record"]["messages"] != job["messages"]):
            raise ValueError(f"{phase} gpu {gpu} call {index}: schedule/prompt/schema mismatch")
        if "input_tokens" in job and event["record"]["input_tokens"] != job["input_tokens"]:
            raise ValueError("input token count mismatch")
        if event["candidate"] is None:
            if not event.get("generation_failure"):
                raise ValueError("missing generation-failure receipt")
            validate_failed_record(event["record"], tokenizer)
        elif (validate_record(event["record"], tokenizer) != event["candidate"]
              or event.get("generation_failure")):
            raise ValueError("candidate replay mismatch")
        candidates.append(event["candidate"])
    return candidates


def load_events(folder):
    status = json.loads((folder / "status.json").read_text())
    if status["status"] != "completed":
        raise ValueError(f"{folder.name} did not complete")
    return [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]


def freeze_pools(schedules, events, tokenizers):
    """schedules/events keyed by phase ('qwen', 'strong'); tokenizers keyed by family."""
    pools = {"qwen": {task: {} for task in design.task_ids()},
             "strong": {task: {} for task in design.task_ids()}}
    for phase in ("qwen", "strong"):
        for gpu in range(design.N_GPUS):
            jobs = schedules[phase]["workers"][str(gpu)]
            families = {job["family"] for job in jobs}
            if len(families) != 1 or (phase == "qwen") != (families == {"qwen"}):
                raise ValueError(f"{phase} schedule family drift")
            replayed = replay_worker(phase, gpu, jobs, events[phase][gpu],
                                     tokenizers[families.pop()])
            for job, candidate in zip(jobs, replayed):
                key = (str(job["agent"]) if phase == "qwen"
                       else design.slot_key(job["family"], job["slot"]))
                if key in pools[phase][job["task_id"]]:
                    raise ValueError(f"duplicate {phase} slot")
                pools[phase][job["task_id"]][key] = candidate
    for phase, keys in (("qwen", QWEN_SLOTS), ("strong", STRONG_KEYS)):
        if any(set(slots) != set(keys) for slots in pools[phase].values()):
            raise ValueError(f"{phase} pool incomplete")
    eligible = [task for task in design.task_ids()
                if all(value is not None for phase in pools
                       for value in pools[phase][task].values())]
    return pools["qwen"], pools["strong"], eligible


def build_recipient_schedule(public, qwen_pool, strong_pool, eligible, tokenizer):
    tasks = {item["id"]: item for item in public}
    if list(tasks) != design.task_ids():
        raise ValueError("public tasks do not match frozen order")

    def count(messages):
        return len(prompt_ids(tokenizer, messages))

    workers = {str(gpu): [] for gpu in range(design.N_GPUS)}
    recipient_tasks, unmatched, tokens = [], [], {}
    for task_id in eligible:
        recipient = design.recipient_index(task_id)
        peers = design.peer_indices(recipient)
        source = design.shuffled_task(task_id, eligible)
        order = [design.slot_key(f, s) for f, s in design.strong_order(task_id)]
        own = qwen_pool[task_id][str(recipient)]
        try:
            messages, target, natural = matched_messages(condition_payloads(
                tasks[task_id], own,
                [qwen_pool[task_id][str(i)] for i in peers],
                [qwen_pool[source][str(i)] for i in peers],
                [strong_pool[task_id][key] for key in order]), count)
        except ValueError:
            unmatched.append(task_id)
            continue
        recipient_tasks.append(task_id)
        tokens[task_id] = {"matched": target, "private_unpadded": natural}
        for condition in design.condition_order(task_id):
            kind = design.PEER_SOURCE[condition]
            workers[str(design.gpu_for(task_id))].append({
                "task_id": task_id, "family": "qwen", "condition": condition,
                "recipient": recipient, "seed": design.recipient_seed(task_id),
                "peer_source": kind,
                "message_source_task": (None if kind is None else
                                        source if kind == "shuffled" else task_id),
                "strong_order": order, "messages": messages[condition],
                "input_tokens": natural if condition == "private_unpadded" else target,
            })
    return {"version": design.VERSION, "task_set": design.TASK_SET,
            "eligible_tasks": eligible, "unmatched_tasks": unmatched,
            "recipient_tasks": recipient_tasks, "input_tokens": tokens, "workers": workers,
            "physical_calls": sum(len(jobs) for jobs in workers.values())}


def gates(eligible, unmatched):
    n = len(design.task_ids())
    checks = {
        "ineligible_at_most_5pct": (n - len(eligible)) <= design.MAX_INELIGIBLE_FRACTION * n,
        "unmatched_at_most_2pct": len(unmatched) <= design.MAX_UNMATCHED_FRACTION * len(eligible),
    }
    return {"checks": checks, "passed": all(checks.values())}


def load_tokenizers():
    from transformers import AutoTokenizer
    tokenizers = {family: AutoTokenizer.from_pretrained(f"/models/{family}", local_files_only=True)
                  for family in design.FAMILIES}
    tokenizers["qwen"] = AutoTokenizer.from_pretrained("/model", local_files_only=True)
    return tokenizers


def load_pools(run, tokenizers):
    data = run / "data"
    schedules = {phase: json.loads((data / f"{phase}-schedule.json").read_text())
                 for phase in ("qwen", "strong")}
    events = {phase: {gpu: load_events(run / f"{phase}-gpu{gpu}")
                      for gpu in range(design.N_GPUS)} for phase in ("qwen", "strong")}
    return freeze_pools(schedules, events, tokenizers)


def main():
    run = Path("/run")
    tokenizers = load_tokenizers()
    qwen_pool, strong_pool, eligible = load_pools(run, tokenizers)
    public = json.loads((run / "data" / "public.json").read_text())
    recipient = build_recipient_schedule(public, qwen_pool, strong_pool, eligible,
                                         tokenizers["qwen"])
    decision = gates(eligible, recipient["unmatched_tasks"])
    audit = {"version": design.VERSION, "task_set": design.TASK_SET,
             "passed": decision["passed"], "gates": decision,
             "schedule_sha256": {phase: sha(run / "data" / f"{phase}-schedule.json")
                                 for phase in ("qwen", "strong")},
             "eligible_count": len(eligible),
             "ineligible_tasks": [t for t in design.task_ids() if t not in set(eligible)],
             "unmatched_tasks": recipient["unmatched_tasks"],
             "qwen_pool": qwen_pool, "strong_pool": strong_pool,
             "worker_events_sha256": {f"{phase}-gpu{gpu}":
                                      sha(run / f"{phase}-gpu{gpu}" / "events.jsonl")
                                      for phase in ("qwen", "strong")
                                      for gpu in range(design.N_GPUS)}}
    out = Path("/output")
    save(out / "pool-audit.json", audit)
    if decision["passed"]:
        recipient["pool_audit_sha256"] = sha(out / "pool-audit.json")
        save(out / "recipient-schedule.json", recipient)
    print(json.dumps({"passed": decision["passed"], "gates": decision["checks"],
                      "eligible": len(eligible), "unmatched": len(recipient["unmatched_tasks"]),
                      "physical_calls": recipient["physical_calls"]}))
    if not decision["passed"]:
        raise SystemExit("freeze gate failed; diagnose before the recipient phase")


if __name__ == "__main__":
    main()
