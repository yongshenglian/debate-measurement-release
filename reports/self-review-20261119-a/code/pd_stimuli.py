"""Replay editor events, validate twins (label-aware stimulus construction), then either
schedule edit retries or freeze the stimuli and build matched recipient prompts."""
import difflib
import json
from pathlib import Path

from kv_io import save, sha
from kv_runtime import prompt_ids
from pc_freeze import load_events
import pd_design as design
from pd_contract import edit_messages, editor_schema, matched_messages, payload
from pd_runtime import validate_editor_record, validate_failed_editor_record


def replay_events(phase, gpu, jobs, events, tokenizer, schema, validate, validate_failed):
    """Independently replay one worker's events; return candidates (None = invalid)."""
    if len(events) != len(jobs):
        raise ValueError(f"{phase} gpu {gpu} event count mismatch")
    out = []
    for index, (job, event) in enumerate(zip(jobs, events)):
        if (event["gpu"] != gpu or event["phase"] != phase or event["call_index"] != index
                or event["job"] != job or event["schema"] != schema
                or event["record"]["messages"] != job["messages"]):
            raise ValueError(f"{phase} gpu {gpu} call {index}: schedule/prompt/schema mismatch")
        if "input_tokens" in job and event["record"]["input_tokens"] != job["input_tokens"]:
            raise ValueError("input token count mismatch")
        if event["candidate"] is None:
            if not event.get("generation_failure"):
                raise ValueError("missing generation-failure receipt")
            validate_failed(event["record"], tokenizer)
        elif (validate(event["record"], tokenizer) != event["candidate"]
              or event.get("generation_failure")):
            raise ValueError("candidate replay mismatch")
        out.append(event["candidate"])
    return out


def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def pair_ok(c, e):
    ratio = len(e["rationale"]) / max(1, len(c["rationale"]))
    return (design.MIN_LENGTH_RATIO <= ratio <= design.MAX_LENGTH_RATIO
            and similarity(c["rationale"], e["rationale"]) >= design.MIN_SIMILARITY)


def select_twins(attempts, label):
    """attempts[twin] = list of edit dicts or None, in attempt order.

    Returns (twins or None, twins needing another attempt, exclusion reason)."""
    c_ok = [i for i, edit in enumerate(attempts["C"]) if edit and edit["answer"] == label]
    c = attempts["C"][c_ok[0]] if c_ok else None
    e_ok = [i for i, edit in enumerate(attempts["E"])
            if edit and edit["answer"] != label and (c is None or pair_ok(c, edit))]
    retry = [twin for twin, ok in (("C", c_ok), ("E", e_ok))
             if not ok and len(attempts[twin]) < design.MAX_EDIT_ATTEMPTS]
    if c_ok and e_ok:
        return ({"C": {**c, "attempt": c_ok[0]},
                 "E": {**attempts["E"][e_ok[0]], "attempt": e_ok[0]}}, [], None)
    reason = "no_valid_C" if not c_ok else "no_valid_E"
    return None, retry, reason


def edit_jobs(entries, public, twins_needed, attempt):
    tasks = {item["id"]: item for item in public}
    return [{"task_id": entry["task_id"], "family": design.EDITOR_FAMILY, "twin": twin,
             "attempt": attempt, "seed": design.edit_seed(entry["task_id"], twin, attempt),
             "messages": edit_messages(tasks[entry["task_id"]], entry["base"]["proposal"], twin)}
            for entry in entries for twin in twins_needed.get(entry["task_id"], ())]


def spread(jobs):
    workers = {str(gpu): [] for gpu in range(design.N_GPUS)}
    for index, job in enumerate(jobs):
        workers[str(index % design.N_GPUS)].append(job)
    return workers


def collect_edits(run, tokenizer, phases):
    """attempts[task][twin] = list ordered by attempt of edit dict or None."""
    attempts = {}
    for phase in phases:
        schedule = json.loads((run / "data" / f"{phase}-schedule.json").read_text())
        for gpu, jobs in schedule["workers"].items():
            if not jobs:
                continue
            edits = replay_events(phase, int(gpu), jobs, load_events(run / f"{phase}-gpu{gpu}"),
                                  tokenizer, editor_schema(), validate_editor_record,
                                  validate_failed_editor_record)
            for job, edit in zip(jobs, edits):
                slot = attempts.setdefault(job["task_id"], {"C": [], "E": []})[job["twin"]]
                if len(slot) != job["attempt"]:
                    raise ValueError("edit attempts out of order")
                slot.append(edit)
    return attempts


def build_recipient_schedule(public, entries, stimuli, tokenizer):
    tasks = {item["id"]: item for item in public}

    def count(messages):
        return len(prompt_ids(tokenizer, messages))

    workers = {str(gpu): [] for gpu in range(design.N_GPUS)}
    recipient_tasks, unmatched, tokens = [], [], {}
    for position, entry in enumerate(entries):
        task_id = entry["task_id"]
        record = stimuli[task_id]
        names = design.conditions_for(record)
        if not names:
            continue
        peers = {}
        if record["twins"]:
            peers.update({k: record["twins"][k] for k in ("C", "E")})
        if record["real"]:
            peers.update(record["real"])
        payloads = {}
        for name in names:
            slot, presentation = design.CONDITIONS[name]
            payloads[name] = payload(tasks[task_id], entry["own"],
                                     None if slot is None else peers[slot], presentation)
        try:
            messages, target = matched_messages(payloads, count)
        except ValueError:
            unmatched.append(task_id)
            continue
        recipient_tasks.append(task_id)
        tokens[task_id] = target
        for name in design.condition_order(task_id, names):
            slot = design.CONDITIONS[name][0]
            workers[str(design.gpu_for(position))].append({
                "task_id": task_id, "family": "qwen", "condition": name,
                "seed": design.recipient_seed(task_id),
                "peer_answer": None if slot is None else peers[slot]["answer"],
                "messages": messages[name], "input_tokens": target})
    return {"version": design.VERSION, "recipient_tasks": recipient_tasks,
            "unmatched_tasks": unmatched, "input_tokens": tokens, "workers": workers,
            "physical_calls": sum(len(jobs) for jobs in workers.values())}


def freeze_stimuli(entries, labels, attempts, retry_done):
    stimuli, retry, excluded = {}, {}, {}
    for entry in entries:
        task_id = entry["task_id"]
        twins = None
        if entry["base"] is not None:
            twins, needed, reason = select_twins(attempts[task_id], labels[task_id])
            if twins is None:
                if needed and not retry_done:
                    retry[task_id] = needed
                else:
                    excluded[task_id] = reason
        stimuli[task_id] = {"twins": twins, "real": entry["real"]}
    return stimuli, retry, excluded


def main():
    from transformers import AutoTokenizer
    run = Path("/run")
    data = run / "data"
    gemma = AutoTokenizer.from_pretrained("/models/gemma", local_files_only=True)
    entries = json.loads((data / "tasks.json").read_text())["tasks"]
    public = json.loads((data / "public.json").read_text())
    labels = {i["id"]: i["answer"] for i in json.loads((run / "hidden/hidden.json").read_text())}
    retry_done = (data / "retry-schedule.json").exists()
    phases = ("editor", "retry") if retry_done else ("editor",)
    attempts = collect_edits(run, gemma, phases)
    stimuli, retry, excluded = freeze_stimuli(entries, labels, attempts, retry_done)
    out = Path("/output")
    if retry:
        jobs = edit_jobs([e for e in entries if e["task_id"] in retry], public, retry, 1)
        save(out / "retry-schedule.json", {"version": design.VERSION, "phase": "retry",
                                           "workers": spread(jobs), "physical_calls": len(jobs)})
        print(json.dumps({"stage": "retry_needed", "retry_tasks": len(retry),
                          "retry_calls": len(jobs)}))
        return
    if not retry_done:
        save(out / "retry-schedule.json", {"version": design.VERSION, "phase": "retry",
                                           "workers": spread([]), "physical_calls": 0})
    qwen = AutoTokenizer.from_pretrained("/model", local_files_only=True)
    schedule = build_recipient_schedule(public, entries, stimuli, qwen)
    arm_i = [t for t, s in stimuli.items() if s["twins"]]
    review = [{"task_id": t, "question": next(p["question"] for p in public if p["id"] == t),
               "label": labels[t], "C": stimuli[t]["twins"]["C"], "E": stimuli[t]["twins"]["E"]}
              for t in design.review_sample(arm_i)]
    summary = {"version": design.VERSION, "arm_I_tasks": len(arm_i),
               "arm_R_tasks": sum(bool(s["real"]) for s in stimuli.values()),
               "excluded_from_arm_I": excluded,
               "edit_attempts_used": {"C1": sum(s["twins"]["C"]["attempt"] == 1 for s in stimuli.values() if s["twins"]),
                                      "E1": sum(s["twins"]["E"]["attempt"] == 1 for s in stimuli.values() if s["twins"])},
               "unmatched": schedule["unmatched_tasks"], "physical_calls": schedule["physical_calls"],
               "passed": len(schedule["unmatched_tasks"])
               <= design.MAX_UNMATCHED_FRACTION * len(schedule["recipient_tasks"] + schedule["unmatched_tasks"])}
    save(out / "stimuli.json", {"version": design.VERSION, "stimuli": stimuli, "summary": summary})
    save(out / "review-sample.json", review)
    schedule["stimuli_sha256"] = sha(out / "stimuli.json")
    save(out / "recipient-schedule.json", schedule)
    print(json.dumps({"stage": "final", **{k: v for k, v in summary.items()
                                           if k != "excluded_from_arm_I"},
                      "excluded": len(excluded)}))


if __name__ == "__main__":
    main()
