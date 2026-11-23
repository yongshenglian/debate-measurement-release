"""Replay the proposal phase, freeze the pool, and build matched recipient prompts.

Label-blind: this step never reads hidden labels.
"""
import json
from pathlib import Path

from kv_io import save, sha
from kv_runtime import prompt_ids, validate_failed_record, validate_record
from kv_task import output_schema
import pe_design as design
from pe_task import condition_payloads, matched_messages


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


def freeze_pool(schedule, events_by_gpu, tokenizer):
    pool = {task: {} for task in design.task_ids()}
    for gpu in range(design.N_GPUS):
        jobs = schedule["workers"][str(gpu)]
        for job, candidate in zip(jobs, replay_worker("proposal", gpu, jobs,
                                                      events_by_gpu[gpu], tokenizer)):
            slot = str(job["agent"])
            if slot in pool[job["task_id"]]:
                raise ValueError("duplicate proposal slot")
            pool[job["task_id"]][slot] = candidate
    if any(set(slots) != {str(a) for a in range(design.N_AGENTS)} for slots in pool.values()):
        raise ValueError("proposal pool incomplete")
    eligible = [task for task, slots in pool.items()
                if all(value is not None for value in slots.values())]
    ineligible = [task for task in design.task_ids() if task not in set(eligible)]
    return pool, eligible, ineligible


def build_recipient_schedule(public, pool, eligible, tokenizer):
    tasks = {item["id"]: item for item in public}
    if list(tasks) != design.task_ids():
        raise ValueError("public tasks do not match frozen order")

    def count(messages):
        return len(prompt_ids(tokenizer, messages))

    workers = {str(gpu): [] for gpu in range(design.N_GPUS)}
    recipient_tasks, unmatched, tokens = [], [], {}
    for task_id in eligible:
        recipient = design.recipient_index(task_id)
        source = design.shuffled_task(task_id, eligible)
        peers = design.peer_indices(recipient)
        own = pool[task_id][str(recipient)]
        relevant = [pool[task_id][str(i)] for i in peers]
        shuffled = [pool[source][str(i)] for i in peers]
        try:
            messages, target, natural = matched_messages(
                condition_payloads(tasks[task_id], own, relevant, shuffled), count)
        except ValueError:
            unmatched.append(task_id)
            continue
        recipient_tasks.append(task_id)
        tokens[task_id] = {"matched": target, "private_unpadded": natural}
        for condition in design.condition_order(task_id):
            workers[str(design.gpu_for(task_id))].append({
                "task_id": task_id, "condition": condition, "recipient": recipient,
                "seed": design.recipient_seed(task_id),
                "message_source_task": (task_id if condition in ("relevant", "answer_only")
                                        else source if condition == "shuffled" else None),
                "messages": messages[condition],
                "input_tokens": target if condition != "private_unpadded" else natural,
            })
    return {"version": design.VERSION, "eligible_tasks": eligible,
            "unmatched_tasks": unmatched, "recipient_tasks": recipient_tasks,
            "input_tokens": tokens, "workers": workers,
            "physical_calls": sum(len(jobs) for jobs in workers.values())}


def gates(eligible, unmatched):
    n = design.N_TASKS
    checks = {
        "ineligible_at_most_5pct": (n - len(eligible)) <= design.MAX_INELIGIBLE_FRACTION * n,
        "unmatched_at_most_2pct": len(unmatched) <= design.MAX_UNMATCHED_FRACTION * len(eligible),
    }
    return {"checks": checks, "passed": all(checks.values())}


def main():
    from transformers import AutoTokenizer
    run = Path("/run")
    data = run / "data"
    tokenizer = AutoTokenizer.from_pretrained("/model", local_files_only=True)
    schedule = json.loads((data / "proposal-schedule.json").read_text())
    events = {gpu: load_events(run / f"proposal-gpu{gpu}") for gpu in range(design.N_GPUS)}
    pool, eligible, ineligible = freeze_pool(schedule, events, tokenizer)
    public = json.loads((data / "public.json").read_text())
    recipient = build_recipient_schedule(public, pool, eligible, tokenizer)
    decision = gates(eligible, recipient["unmatched_tasks"])
    audit = {"version": design.VERSION, "passed": decision["passed"], "gates": decision,
             "proposal_schedule_sha256": sha(data / "proposal-schedule.json"),
             "eligible_count": len(eligible), "ineligible_tasks": ineligible,
             "unmatched_tasks": recipient["unmatched_tasks"], "pool": pool,
             "worker_events_sha256": {gpu: sha(run / f"proposal-gpu{gpu}" / "events.jsonl")
                                      for gpu in range(design.N_GPUS)}}
    out = Path("/output")
    save(out / "proposal-audit.json", audit)
    if decision["passed"]:
        recipient["proposal_audit_sha256"] = sha(out / "proposal-audit.json")
        save(out / "recipient-schedule.json", recipient)
    print(json.dumps({"passed": decision["passed"], "gates": decision["checks"],
                      "eligible": len(eligible), "unmatched": len(recipient["unmatched_tasks"]),
                      "physical_calls": recipient["physical_calls"]}))
    if not decision["passed"]:
        raise SystemExit("freeze gate failed; diagnose before phase 2")


if __name__ == "__main__":
    main()
