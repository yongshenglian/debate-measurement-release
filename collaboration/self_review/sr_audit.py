"""Independent replay audit and the frozen analysis for the self-review study."""
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

from kv_io import save, sha
from pc_contract import output_schema
from pc_freeze import load_events
from pc_runtime import validate_failed_record, validate_record
from pd_stimuli import replay_events
from pe_audit import holm
from pw_audit import label, paired, rate
import sr_design as design
from sr_schedule import build_recipient_schedule

# Names without a prefix are this run's conditions; "s3:" names are Study 3's.
PRIMARY = {"R1_self_check_minus_private": (("self_check",), ("private",)),
           "R2_self_indep_minus_private": (("self_indep",), ("private",)),
           # Does a peer pool with half the peers wrong add value beyond the instruction alone?
           "R3_net_full_check_minus_self_check": (("s3:C_full_check", "s3:E_full_check"),
                                                  ("self_check",)),
           "R4_net_full_indep_minus_self_indep": (("s3:C_full_indep", "s3:E_full_indep"),
                                                  ("self_indep",))}
SECONDARY = {"S1_self_doubt_minus_private": (("self_doubt",), ("private",)),
             "S2_self_doubt_minus_self_indep": (("self_doubt",), ("self_indep",)),
             "S3_E_full_check_minus_self_check": (("s3:E_full_check",), ("self_check",)),
             "S4_C_full_check_minus_self_check": (("s3:C_full_check",), ("self_check",)),
             "S5_E_full_indep_minus_self_indep": (("s3:E_full_indep",), ("self_indep",)),
             "S6_C_full_indep_minus_self_indep": (("s3:C_full_indep",), ("self_indep",)),
             "S7_net_full_minus_private": (("s3:C_full", "s3:E_full"), ("private",)),
             "S8_net_full_check_minus_private": (("s3:C_full_check", "s3:E_full_check"),
                                                 ("private",)),
             "S9_net_full_indep_minus_private": (("s3:C_full_indep", "s3:E_full_indep"),
                                                 ("private",)),
             "S10_self_check_minus_self_indep": (("self_check",), ("self_indep",))}


def analyze(final, study3, labels, own):
    """final[task][condition] and study3[task][condition]: answers by replicate (None = invalid)."""
    tasks = sorted(final)
    reps = range(design.REPLICATES)
    right = {t for t in tasks if own[t] == labels[t]}

    def answers(t, name):
        return study3[t][name[3:]] if name.startswith("s3:") else final[t][name]

    def score(t, names):
        cells = [int(answers(t, n)[r] == labels[t]) for n in names for r in reps]
        return Fraction(sum(cells), len(cells))

    def pair(a, b):
        return paired([score(t, a) for t in tasks], [score(t, b) for t in tasks],
                      max(len(a), len(b)) * len(reps))

    result = {"tasks": len(tasks), "replicates": design.REPLICATES,
              "initially_right": len(right),
              "initial_accuracy": len(right) / len(tasks)}
    cells = {n: [(t, r) for t in tasks for r in reps] for n in design.CONDITIONS}
    result["accuracy"] = {n: rate([int(final[t][n][r] == labels[t]) for t, r in cells[n]])
                          for n in design.CONDITIONS}
    result["accuracy_study3"] = {n: rate([int(study3[t][n][r] == labels[t]) for t in tasks
                                          for r in reps])
                                 for n in ("private", "C_full", "E_full", "C_full_check",
                                           "E_full_check", "C_full_indep", "E_full_indep")}
    primary = {name: pair(a, b) for name, (a, b) in PRIMARY.items()}
    for name, adjusted in holm({n: r["sign_flip_exact_p"] for n, r in primary.items()}).items():
        primary[name]["holm_p"] = adjusted
        primary[name]["label"] = label(primary[name])
    result["primary"] = primary
    result["secondary"] = {name: pair(a, b) for name, (a, b) in SECONDARY.items()}
    initial = [Fraction(int(t in right)) for t in tasks]
    result["versus_initial_answer"] = {
        n: paired([score(t, (n,)) for t in tasks], initial, len(reps)) for n in design.CONDITIONS}
    result["corruption_initially_right"] = {
        n: rate([int(final[t][n][r] != labels[t]) for t in tasks if t in right for r in reps])
        for n in design.CONDITIONS}
    result["rescue_initially_wrong"] = {
        n: rate([int(final[t][n][r] == labels[t]) for t in tasks if t not in right for r in reps])
        for n in design.CONDITIONS}
    result["changed_answer"] = {n: rate([int(final[t][n][r] != own[t]) for t in tasks for r in reps])
                                for n in design.CONDITIONS}
    result["replicate_agreement"] = {
        n: sum(len({final[t][n][r] == labels[t] for r in reps}) == 1 for t in tasks) / len(tasks)
        for n in design.CONDITIONS}
    # Same prompt and seed as Study 3's private condition: how often is the answer identical?
    result["private_replay_identical"] = rate([int(final[t]["private"][r] == study3[t]["private"][r])
                                               for t in tasks for r in reps])
    result["invalid_generations"] = dict(Counter(n for t in tasks for n, fs in final[t].items()
                                                 for f in fs if f is None))
    return result


def audit(run, tokenizer):
    run = Path(run)
    source, data = run / "source", run / "data"
    labels = {i["id"]: i["answer"] for i in json.loads((run / "hidden/hidden.json").read_text())}
    study3 = json.loads((run / "hidden/study3-per-task.json").read_text())
    entries = json.loads((source / "tasks.json").read_text())["tasks"]
    public = json.loads((source / "public.json").read_text())
    old = json.loads((source / "recipient-schedule.json").read_text())
    schedule = json.loads((data / "recipient-schedule.json").read_text())
    if schedule["source_schedule_sha256"] != sha(source / "recipient-schedule.json"):
        raise ValueError("schedule not bound to Study 3's schedule")
    rebuilt = build_recipient_schedule(public, entries, old, tokenizer)
    if any(json.loads(json.dumps(rebuilt[key])) != schedule[key] for key in rebuilt):
        raise ValueError("re-derived recipient schedule differs from frozen schedule")
    final = {}
    for gpu, jobs in schedule["workers"].items():
        candidates = replay_events("recipient", int(gpu), jobs,
                                   load_events(run / f"recipient-gpu{gpu}"), tokenizer,
                                   output_schema(), validate_record, validate_failed_record)
        for job, candidate in zip(jobs, candidates):
            slots = final.setdefault(job["task_id"], {}).setdefault(job["condition"], {})
            if job["replicate"] in slots:
                raise ValueError("duplicate task condition replicate")
            slots[job["replicate"]] = None if candidate is None else candidate["answer"]
    if set(final) != set(schedule["recipient_tasks"]):
        raise ValueError("recipient task set incomplete")
    expected = dict(Counter(n for n, _ in design.jobs_for("")))
    for t, by_name in final.items():
        if {n: len(v) for n, v in by_name.items()} != expected:
            raise ValueError(f"recipient grid incomplete for {t}")
        if study3[t]["label"] != labels[t]:
            raise ValueError("Study 3 label mismatch")
    final = {t: {n: [v[r] for r in sorted(v)] for n, v in by_name.items()}
             for t, by_name in final.items()}
    own = {e["task_id"]: e["own"]["answer"] for e in entries}
    result = analyze(final, {t: study3[t]["final"] for t in final}, labels, own)
    result.update({"version": design.VERSION, "integrity_passed": True,
                   "recipient_calls": schedule["physical_calls"],
                   "hidden_sha256": sha(run / "hidden/hidden.json"),
                   "recipient_schedule_sha256": sha(data / "recipient-schedule.json"),
                   "per_task": {t: {"own": own[t], "label": labels[t], "final": final[t]}
                                for t in final}})
    return result


if __name__ == "__main__":
    from transformers import AutoTokenizer
    qwen = AutoTokenizer.from_pretrained("/model", local_files_only=True)
    result = audit(Path("/run"), qwen)
    save(Path("/output/audit.json"), result)
    print(json.dumps({k: v for k, v in result.items() if k != "per_task"}, indent=2))
