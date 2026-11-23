"""Independent replay audit and the frozen analysis for the peer-warning study."""
from collections import Counter
from fractions import Fraction
from math import comb
import json
from pathlib import Path

from kv_io import save, sha
from pc_contract import output_schema
from pc_freeze import load_events
from pc_runtime import validate_failed_record, validate_record
from pd_stimuli import replay_events
from pe_audit import ALPHA, bootstrap, holm, mcnemar_exact
import pw_design as design
from pw_schedule import build_recipient_schedule

EQUIVALENCE_MARGIN = 0.02
PRIMARY = {"P1_E_full_warn_minus_E_full": ("E_full_warn", "E_full"),
           "P2_C_full_warn_minus_C_full": ("C_full_warn", "C_full"),
           "P3_E_confident_warn_minus_E_confident": ("E_confident_warn", "E_confident"),
           "P4_E_full_warn_minus_E_hedged": ("E_full_warn", "E_hedged"),
           "P5_E_full_indep_minus_E_full": ("E_full_indep", "E_full"),
           "P6_C_full_indep_minus_C_full": ("C_full_indep", "C_full")}
SECONDARY = {"S1_E_hedged_minus_E_full": ("E_hedged", "E_full"),
             "S2_E_confident_minus_E_full": ("E_confident", "E_full"),
             "S3_C_confident_warn_minus_C_confident": ("C_confident_warn", "C_confident"),
             "S4_E_hedged_warn_minus_E_hedged": ("E_hedged_warn", "E_hedged"),
             "S5_C_hedged_warn_minus_C_hedged": ("C_hedged_warn", "C_hedged"),
             "S6_E_full_check_minus_E_full": ("E_full_check", "E_full"),
             "S7_C_full_check_minus_C_full": ("C_full_check", "C_full"),
             "S8_E_full_warn_minus_E_full_check": ("E_full_warn", "E_full_check"),
             "S9_C_hedged_minus_C_full": ("C_hedged", "C_full"),
             "S10_E_full_minus_private": ("E_full", "private"),
             "S11_C_full_minus_private": ("C_full", "private"),
             "S12_E_confident_indep_minus_E_confident": ("E_confident_indep", "E_confident"),
             "S13_C_confident_indep_minus_C_confident": ("C_confident_indep", "C_confident"),
             "S14_E_full_indep_minus_E_full_warn": ("E_full_indep", "E_full_warn"),
             "S15_C_full_indep_minus_C_full_warn": ("C_full_indep", "C_full_warn")}
# Net effect if half of all peers are wrong: mean of the C and E conditions.
NET = {"N1_full_warn_minus_full": (("C_full_warn", "E_full_warn"), ("C_full", "E_full")),
       "N2_confident_warn_minus_confident": (("C_confident_warn", "E_confident_warn"),
                                             ("C_confident", "E_confident")),
       "N3_full_check_minus_full": (("C_full_check", "E_full_check"), ("C_full", "E_full")),
       "N4_full_indep_minus_full": (("C_full_indep", "E_full_indep"), ("C_full", "E_full")),
       "N5_confident_indep_minus_confident": (("C_confident_indep", "E_confident_indep"),
                                              ("C_confident", "E_confident"))}
ADV_SECONDARY = {"A1_adv_E_full_warn_minus_adv_E_full": ("adv_E_full_warn", "adv_E_full"),
                 "A2_adv_C_full_warn_minus_adv_C_full": ("adv_C_full_warn", "adv_C_full"),
                 "A4_adv_E_full_minus_private": ("adv_E_full", "private"),
                 "A5_adv_C_full_minus_private": ("adv_C_full", "private")}
ADV_NET = {"A3_net_adv_full_warn_minus_adv_full": (("adv_C_full_warn", "adv_E_full_warn"),
                                                   ("adv_C_full", "adv_E_full"))}
PRESENTATIONS = ("full", "hedged", "confident", "full_warn", "hedged_warn", "confident_warn",
                 "full_check", "full_indep", "confident_indep")


def sign_flip_exact(units):
    """Exact two-sided sign-flip p for integer per-task differences (= exact McNemar for 0/±1)."""
    observed = abs(sum(units))
    if observed == 0:
        return 1.0
    counts = {0: 1}
    for u, m in Counter(abs(x) for x in units if x).items():
        nxt = Counter()
        for j in range(m + 1):
            shift, ways = u * (2 * j - m), comb(m, j)
            for total, w in counts.items():
                nxt[total + shift] += w * ways
        counts = nxt
    hits = sum(ways for total, ways in counts.items() if abs(total) >= observed)
    return float(Fraction(hits, sum(counts.values())))


def paired(a, b, denominator):
    """a, b: per-task scores in multiples of 1/denominator."""
    diffs = [x - y for x, y in zip(a, b)]
    units = [round(d * denominator) for d in diffs]
    if any(abs(u - d * denominator) > 1e-9 for u, d in zip(units, diffs)):
        raise ValueError("scores are not on the stated grid")
    return {"estimate": float(sum(diffs)) / len(diffs), "tasks": len(diffs),
            "a_better_tasks": sum(u > 0 for u in units), "b_better_tasks": sum(u < 0 for u in units),
            "sign_flip_exact_p": sign_flip_exact(units),
            "bootstrap": bootstrap([float(d) for d in diffs])}


def label(result):
    low, high = result["bootstrap"]["ci90"]
    if result["holm_p"] < ALPHA:
        return "benefit" if result["estimate"] > 0 else "harm"
    if -EQUIVALENCE_MARGIN < low and high < EQUIVALENCE_MARGIN:
        return "equivalent_within_2pp"
    return "inconclusive"


def rate(cells):
    k, n = sum(cells), len(cells)
    return {"n": n, "k": k, "rate": k / n if n else None}


def analyze(stimuli, own, labels, final, peer):
    """final[task][condition] = list of answers by replicate (None = invalid)."""
    tasks_g = sorted(t for t in final if stimuli[t]["gemma"])
    tasks_a = sorted(t for t in final if stimuli[t]["adv"])
    right = {t for t in final if own[t] == labels[t]}

    def ok(t, name, replicates):
        return [int(final[t][name][r] == labels[t]) for r in replicates]

    def score(t, names, replicates):
        cells = [c for n in names for c in ok(t, n, replicates)]
        return Fraction(sum(cells), len(cells))

    g_reps, a_reps = range(design.REPLICATES), range(design.ADV_REPLICATES)

    def pair(tasks, a, b, reps, k=1):
        a = a if isinstance(a, tuple) else (a,)
        b = b if isinstance(b, tuple) else (b,)
        return paired([score(t, a, reps) for t in tasks], [score(t, b, reps) for t in tasks],
                      k * len(reps))

    result = {"tasks_gemma": len(tasks_g), "tasks_adv": len(tasks_a),
              "replicates_gemma": design.REPLICATES, "replicates_adv": design.ADV_REPLICATES,
              "initially_right": {"gemma": sum(t in right for t in tasks_g),
                                  "adv": sum(t in right for t in tasks_a)}}
    result["accuracy_gemma"] = {n: rate([c for t in tasks_g for c in ok(t, n, g_reps)])
                                for n in design.GEMMA_CONDITIONS}
    result["accuracy_adv"] = {n: rate([c for t in tasks_a for c in ok(t, n, a_reps)])
                              for n in ("private",) + tuple(design.ADV_CONDITIONS)}

    primary = {name: pair(tasks_g, a, b, g_reps) for name, (a, b) in PRIMARY.items()}
    for name, adjusted in holm({n: r["sign_flip_exact_p"] for n, r in primary.items()}).items():
        primary[name]["holm_p"] = adjusted
        primary[name]["label"] = label(primary[name])
    result["primary"] = primary
    # Sensitivity: replicate 0 alone with the earlier studies' exact McNemar.
    sens = {}
    for name, (a, b) in PRIMARY.items():
        d = [x - y for x, y in zip([ok(t, a, [0])[0] for t in tasks_g],
                                   [ok(t, b, [0])[0] for t in tasks_g])]
        sens[name] = {"estimate": sum(d) / len(d), "mcnemar_exact_p": mcnemar_exact(d.count(1), d.count(-1))}
    result["primary_replicate0_mcnemar"] = sens
    result["secondary"] = {name: pair(tasks_g, a, b, g_reps) for name, (a, b) in SECONDARY.items()}
    result["net_half_wrong"] = {name: pair(tasks_g, a, b, g_reps, 2) for name, (a, b) in NET.items()}
    result["secondary_adv"] = {name: pair(tasks_a, a, b, a_reps) for name, (a, b) in ADV_SECONDARY.items()}
    result["secondary_adv"].update({name: pair(tasks_a, a, b, a_reps, 2)
                                    for name, (a, b) in ADV_NET.items()})

    # Outcomes by the recipient's initial correctness (cells = task x replicate).
    def cells(tasks, name, reps, test):
        return [int(test(t, name, final[t][name][r])) for t in tasks for r in reps]

    for arm, tasks, reps, names in (("gemma", tasks_g, g_reps, tuple(design.GEMMA_CONDITIONS)),
                                    ("adv", tasks_a, a_reps, ("private",) + tuple(design.ADV_CONDITIONS))):
        r_tasks = [t for t in tasks if t in right]
        w_tasks = [t for t in tasks if t not in right]
        result[f"corruption_initially_right_{arm}"] = {
            n: rate(cells(r_tasks, n, reps, lambda t, n, f: f != labels[t]))
            for n in names if n == "private" or "E_" in n}
        result[f"rescue_initially_wrong_{arm}"] = {
            n: rate(cells(w_tasks, n, reps, lambda t, n, f: f == labels[t]))
            for n in names if n == "private" or "C_" in n}
        adoption = {}
        for n in names:
            if n == "private":
                continue
            eligible = [t for t in w_tasks if peer[t][n] != own[t]]
            adoption[n] = rate(cells(eligible, n, reps, lambda t, n, f: f == peer[t][n]))
        result[f"adoption_initially_wrong_{arm}"] = adoption
        source = {}
        for n in names:
            if "E_" not in n:
                continue
            count = Counter()
            for t in tasks:
                for r in reps:
                    f = final[t][n][r]
                    count["peer_answer" if f == peer[t][n] else "own_answer" if f == own[t]
                          else "correct" if f == labels[t] else "invalid" if f is None
                          else "other"] += 1
            source[n] = dict(count)
        result[f"final_answer_source_with_wrong_peer_{arm}"] = source
    adoption = result["adoption_initially_wrong_gemma"]
    result["discrimination_initially_wrong"] = {
        p: adoption[f"C_{p}"]["rate"] - adoption[f"E_{p}"]["rate"] for p in PRESENTATIONS
        if adoption[f"C_{p}"]["rate"] is not None and adoption[f"E_{p}"]["rate"] is not None}

    # Test-retest: replicate agreement in correctness per Gemma condition.
    result["replicate_agreement_gemma"] = {
        n: sum(len(set(ok(t, n, g_reps))) == 1 for t in tasks_g) / len(tasks_g)
        for n in design.GEMMA_CONDITIONS} if design.REPLICATES > 1 else None
    result["adv_corruption_by_position_warn"] = {}
    a_right = [t for t in tasks_a if t in right]
    for n in ("adv_E_full", "adv_E_full_warn"):
        by = {}
        for t in a_right:
            by.setdefault(stimuli[t]["adv"]["error_type"].split("_")[0], []).append(
                int(final[t][n][0] != labels[t]))
        result["adv_corruption_by_position_warn"][n] = {k: rate(v) for k, v in sorted(by.items())}

    negatives = Counter(n for t in final for n, fs in final[t].items() for f in fs
                        if labels[t] < 0)
    revisions_negative = Counter(n for t in final for n, fs in final[t].items()
                                 for f in fs if f is not None and f < 0)
    cells_per = Counter(n for t in final for n, fs in final[t].items() for f in fs)
    result["contract_check"] = {"labels_negative": sum(labels[t] < 0 for t in final),
                                "revisions_negative": dict(revisions_negative),
                                "flag": any(v - negatives[n] > 0.02 * cells_per[n]
                                            for n, v in revisions_negative.items())}
    result["invalid_generations"] = dict(Counter(n for t in final for n, fs in final[t].items()
                                                 for f in fs if f is None))
    result["recipient_calls"] = sum(cells_per.values())
    return result


def audit(run, tokenizer):
    run = Path(run)
    data = run / "data"
    labels = {i["id"]: i["answer"] for i in json.loads((run / "hidden/hidden.json").read_text())}
    entries = json.loads((data / "tasks.json").read_text())["tasks"]
    public = json.loads((data / "public.json").read_text())
    schedule = json.loads((data / "recipient-schedule.json").read_text())
    if schedule["tasks_sha256"] != sha(data / "tasks.json"):
        raise ValueError("recipient schedule not bound to tasks")
    rebuilt = build_recipient_schedule(public, entries, tokenizer)
    if any(json.loads(json.dumps(rebuilt[key])) != schedule[key] for key in rebuilt):
        raise ValueError("re-derived recipient schedule differs from frozen schedule")
    stimuli = {e["task_id"]: e for e in entries}
    final, peer = {}, {}
    for gpu, jobs in schedule["workers"].items():
        candidates = replay_events("recipient", int(gpu), jobs,
                                   load_events(run / f"recipient-gpu{gpu}"), tokenizer,
                                   output_schema(), validate_record, validate_failed_record)
        for job, candidate in zip(jobs, candidates):
            slots = final.setdefault(job["task_id"], {}).setdefault(job["condition"], {})
            if job["replicate"] in slots:
                raise ValueError("duplicate task condition replicate")
            slots[job["replicate"]] = None if candidate is None else candidate["answer"]
            peer.setdefault(job["task_id"], {})[job["condition"]] = job["peer_answer"]
    if set(final) != set(schedule["recipient_tasks"]):
        raise ValueError("recipient task set incomplete")
    for t, by_name in final.items():
        expected = Counter(n for n, _ in design.jobs_for(stimuli[t]))
        if {n: len(v) for n, v in by_name.items()} != dict(expected):
            raise ValueError(f"paired recipient grid incomplete for {t}")
    final = {t: {n: [v[r] for r in sorted(v)] for n, v in by_name.items()}
             for t, by_name in final.items()}
    own = {e["task_id"]: e["own"]["answer"] for e in entries}
    result = analyze(stimuli, own, labels, final, peer)
    result.update({"version": design.VERSION, "integrity_passed": True,
                   "unmatched_tasks": schedule["unmatched_tasks"],
                   "hidden_sha256": sha(run / "hidden/hidden.json"),
                   "recipient_schedule_sha256": sha(data / "recipient-schedule.json"),
                   "per_task": {t: {"own": own[t], "label": labels[t], "final": final[t],
                                    "peer": peer[t]} for t in final}})
    return result


if __name__ == "__main__":
    from transformers import AutoTokenizer
    qwen = AutoTokenizer.from_pretrained("/model", local_files_only=True)
    result = audit(Path("/run"), qwen)
    save(Path("/output/audit.json"), result)
    print(json.dumps({k: v for k, v in result.items() if k != "per_task"}, indent=2))
