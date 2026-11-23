"""Build the token-matched recipient schedule from the frozen tasks (CPU, pinned image)."""
import json
from pathlib import Path

from kv_io import save, sha
from kv_runtime import prompt_ids
import pw_design as design
from pw_contract import matched_messages, payload, peers_for


def build_recipient_schedule(public, entries, tokenizer):
    tasks = {item["id"]: item for item in public}

    def count(messages):
        return len(prompt_ids(tokenizer, messages))

    workers = {str(gpu): [] for gpu in range(design.N_GPUS)}
    recipient_tasks, unmatched, tokens = [], [], {}
    for position, entry in enumerate(entries):
        task_id = entry["task_id"]
        names = design.conditions_for(entry)
        if not names:
            continue
        peers = peers_for(entry)
        payloads = {name: payload(tasks[task_id], entry["own"], peers[name],
                                  design.CONDITIONS[name][2]) for name in names}
        try:
            messages, target = matched_messages(payloads, count, design.MAX_PROMPT_TOKENS)
            if target > design.MAX_PROMPT_TOKENS:
                raise ValueError("over the prompt limit")
        except ValueError:
            unmatched.append(task_id)
            continue
        recipient_tasks.append(task_id)
        tokens[task_id] = target
        for name, replicate in design.job_order(task_id, design.jobs_for(entry)):
            peer = peers[name]
            workers[str(design.gpu_for(position))].append({
                "task_id": task_id, "family": "qwen", "condition": name, "replicate": replicate,
                "seed": design.recipient_seed(task_id, replicate),
                "peer_answer": peer["answer"] if peer else None,
                "messages": messages[name], "input_tokens": target})
    return {"version": design.VERSION, "recipient_tasks": recipient_tasks,
            "unmatched_tasks": unmatched, "input_tokens": tokens, "workers": workers,
            "physical_calls": sum(len(jobs) for jobs in workers.values())}


def main():
    from transformers import AutoTokenizer
    data = Path("/run/data")
    entries = json.loads((data / "tasks.json").read_text())["tasks"]
    public = json.loads((data / "public.json").read_text())
    qwen = AutoTokenizer.from_pretrained("/model", local_files_only=True)
    schedule = build_recipient_schedule(public, entries, qwen)
    schedule["tasks_sha256"] = sha(data / "tasks.json")
    total = len(schedule["recipient_tasks"]) + len(schedule["unmatched_tasks"])
    schedule["passed"] = len(schedule["unmatched_tasks"]) <= design.MAX_UNMATCHED_FRACTION * total
    save(Path("/output/recipient-schedule.json"), schedule)
    targets = list(schedule["input_tokens"].values())
    print(json.dumps({"recipient_tasks": len(schedule["recipient_tasks"]),
                      "unmatched": schedule["unmatched_tasks"],
                      "physical_calls": schedule["physical_calls"],
                      "per_gpu": {g: len(j) for g, j in schedule["workers"].items()},
                      "tokens": [min(targets), max(targets), sum(targets) / len(targets)],
                      "passed": schedule["passed"]}))


if __name__ == "__main__":
    main()
