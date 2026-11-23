"""Recipient schedule padded to Study 3's per-task token targets, with Study 3's seeds (CPU)."""
import json
from pathlib import Path

from kv_io import save, sha
from kv_runtime import prompt_ids
from kv_task import recipient_messages
from pc_task import pad_to
import sr_design as design
from sr_contract import payload


def build_recipient_schedule(public, entries, source, tokenizer):
    """source: Study 3's frozen recipient schedule."""
    tasks = {item["id"]: item for item in public}

    def count(messages):
        return len(prompt_ids(tokenizer, messages))

    private = {}
    for jobs in source["workers"].values():
        for job in jobs:
            if job["condition"] == "private":
                private[(job["task_id"], job["replicate"])] = job
    workers = {str(gpu): [] for gpu in range(design.N_GPUS)}
    recipient_tasks = []
    for entry in entries:
        task_id = entry["task_id"]
        if not entry.get("gemma") or task_id not in source["input_tokens"]:
            continue
        target = source["input_tokens"][task_id]
        messages = {}
        for name in design.CONDITIONS:
            padded = pad_to(payload(tasks[task_id], entry["own"], name), target, count,
                            design.MAX_PROMPT_TOKENS)
            messages[name] = recipient_messages(padded)
            if count(messages[name]) != target:
                raise ValueError("token matching failed")
        gpu = str(len(recipient_tasks) % design.N_GPUS)
        recipient_tasks.append(task_id)
        for name, replicate in design.job_order(task_id, design.jobs_for(task_id)):
            old = private[(task_id, replicate)]
            seed = design.recipient_seed(task_id, replicate)
            if seed != old["seed"] or old["input_tokens"] != target:
                raise ValueError("seed or target differs from Study 3")
            if name == "private" and messages[name] != old["messages"]:
                raise ValueError("private prompt differs from Study 3")
            workers[gpu].append({
                "task_id": task_id, "family": "qwen", "condition": name, "replicate": replicate,
                "seed": seed, "peer_answer": None, "messages": messages[name],
                "input_tokens": target})
    return {"version": design.VERSION, "recipient_tasks": recipient_tasks,
            "unmatched_tasks": [], "workers": workers,
            "physical_calls": sum(len(jobs) for jobs in workers.values())}


def main():
    from transformers import AutoTokenizer
    source = Path("/source")
    entries = json.loads((source / "tasks.json").read_text())["tasks"]
    public = json.loads((source / "public.json").read_text())
    old = json.loads((source / "recipient-schedule.json").read_text())
    qwen = AutoTokenizer.from_pretrained("/model", local_files_only=True)
    schedule = build_recipient_schedule(public, entries, old, qwen)
    schedule["source_schedule_sha256"] = sha(source / "recipient-schedule.json")
    schedule["passed"] = len(schedule["recipient_tasks"]) == 647
    save(Path("/output/recipient-schedule.json"), schedule)
    print(json.dumps({"recipient_tasks": len(schedule["recipient_tasks"]),
                      "physical_calls": schedule["physical_calls"],
                      "per_gpu": {g: len(j) for g, j in schedule["workers"].items()},
                      "passed": schedule["passed"]}))


if __name__ == "__main__":
    main()
