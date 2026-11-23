"""Amendment v2 analysis: the v1 contagion analysis applied per arm (seeds 0-3), Holm over 3 arms x 2."""
import json, sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc_analyze as A  # noqa: E402

CONDS = ("clean", "esc1", "esc4", "loop1", "loop4")
NAMES = ("broken_format", "published_correct", "robust_correct", "hit_token_limit")

def arm(run, jobs_file, out_file, max_rep=4):
    labels = {int(k): v for k, v in json.load(open(run / "labels_hidden.json")).items()}
    jobs = {}
    for l in open(run / jobs_file):
        j = json.loads(l)
        if j["replicate"] < max_rep:
            jobs[j["id"]] = j
    outs = [json.loads(l) for l in open(run / out_file)]
    outs = [o for o in outs if o["id"] in jobs]
    assert {o["id"] for o in outs} == set(jobs), "incomplete arm"
    cell = defaultdict(lambda: defaultdict(list))
    for o in outs:
        j = jobs[o["id"]]; y = labels[j["task"]]
        a, b = A.original(o["text"]), A.robust(o["text"])
        cell[j["task"]][j["condition"]].append((a is None, a is not None and round(a, 1) == y,
                                                 b is not None and round(b, 1) == y, o["finish_reason"] == "length"))
    tasks = sorted(cell); k = max_rep
    def score(t, c, i):
        v = cell[t][c]; return Fraction(sum(x[i] for x in v), len(v))
    rates = {c: {n: float(sum(score(t, c, i) for t in tasks) / len(tasks)) for i, n in enumerate(NAMES)} for c in CONDS}
    contrasts = {f"{c}_minus_clean_{n}": A.paired([score(t, c, i) for t in tasks], [score(t, "clean", i) for t in tasks], k)
                 for c in CONDS[1:] for i, n in enumerate(NAMES)}
    return {"tasks": len(tasks), "samples": k, "rates": rates, "contrasts": contrasts}

if __name__ == "__main__":
    run = Path(sys.argv[1])
    res = {"qwen_raw_seeds0-3": arm(run, "contagion_jobs.jsonl", "out-qwen/contagion_out.jsonl")}
    for name in ("qwenchat", "mistral", "gemma"):
        res[name] = arm(run, "contagion_jobs_chat_v2.jsonl", f"out-{name}/contagion_out.jsonl")
    prim = {}
    for name in ("qwenchat", "mistral", "gemma"):
        for key in ("loop4_minus_clean_broken_format", "loop4_minus_clean_robust_correct"):
            prim[f"{name}:{key}"] = res[name]["contrasts"][key]
    for n, p in A.holm({n: r["p"] for n, r in prim.items()}).items():
        prim[n]["holm_p"] = p
    res["primary_v2"] = prim
    json.dump(res, open(run / "analysis_v2.json", "x"), indent=1)
    for name in ("qwen_raw_seeds0-3", "qwenchat", "mistral", "gemma"):
        print("==", name)
        for c, v in res[name]["rates"].items():
            print(f"  {c:6s}", {k: round(x, 3) for k, x in v.items()})
    for n, r in prim.items():
        print(n, round(r["estimate"], 3), [round(x, 3) for x in r["ci95"]], "holm %.2g" % r["holm_p"])
