"""Frozen analysis for format-contagion-v1 (A: twin test; B: blind extraction audit)."""
import json, re, sys
from collections import defaultdict, Counter
from fractions import Fraction
from pathlib import Path
R = Path("/home/lian/swarmworld/research")
sys.path[:0] = [str(R / "reports/peer-theory-20261119")] + [str(R / "collaboration" / d) for d in (
    "peer_warning", "peer_evidence", "kv_communication", "peer_clean", "peer_discrimination",
    "peer_persuasion", "peer_strength")]
from empties import original, robust  # noqa: E402
from pw_audit import sign_flip_exact  # noqa: E402
from pe_audit import bootstrap, holm  # noqa: E402

def paired(a, b, k):
    d = [x - y for x, y in zip(a, b)]
    units = [round(x * k) for x in d]
    return {"estimate": float(sum(d)) / len(d), "p": sign_flip_exact(units),
            "ci95": bootstrap([float(x) for x in d])["ci95"], "tasks": len(d)}

def contagion(run):
    labels = {int(k): v for k, v in json.load(open(run / "labels_hidden.json")).items()}
    jobs = {json.loads(l)["id"]: json.loads(l) for l in open(run / "contagion_jobs.jsonl")}
    outs = [json.loads(l) for l in open(run / "out-qwen/contagion_out.jsonl")]
    assert len(outs) == len(jobs) and {o["id"] for o in outs} == set(jobs)
    cell = defaultdict(lambda: defaultdict(list))   # task -> condition -> list of (broken, orig_ok, rob_ok, length)
    for o in outs:
        j = jobs[o["id"]]; y = labels[j["task"]]
        a, b = original(o["text"]), robust(o["text"])
        cell[j["task"]][j["condition"]].append((a is None, a is not None and round(a, 1) == y,
                                                 b is not None and round(b, 1) == y, o["finish_reason"] == "length"))
    tasks = sorted(cell)
    conds = ("clean", "esc1", "esc4", "loop1", "loop4")
    k = len(cell[tasks[0]]["clean"])
    def score(t, c, i):
        v = cell[t][c]; return Fraction(sum(x[i] for x in v), len(v))
    res = {"tasks": len(tasks), "samples_per_condition": k, "rates": {}}
    names = ("broken_format", "published_correct", "robust_correct", "hit_token_limit")
    for c in conds:
        res["rates"][c] = {n: float(sum(score(t, c, i) for t in tasks) / len(tasks)) for i, n in enumerate(names)}
    prim = {"F1_esc1_minus_clean_broken": paired([score(t, "esc1", 0) for t in tasks], [score(t, "clean", 0) for t in tasks], k),
            "F2_loop4_minus_clean_broken": paired([score(t, "loop4", 0) for t in tasks], [score(t, "clean", 0) for t in tasks], k),
            "F3_loop4_minus_clean_robust_correct": paired([score(t, "loop4", 2) for t in tasks], [score(t, "clean", 2) for t in tasks], k)}
    for n, p in holm({n: r["p"] for n, r in prim.items()}).items():
        prim[n]["holm_p"] = p
    res["primary"] = prim
    sec = {}
    for c in conds[1:]:
        for i, n in enumerate(names):
            sec[f"{c}_minus_clean_{n}"] = paired([score(t, c, i) for t in tasks], [score(t, "clean", i) for t in tasks], k)
    sec["esc4_minus_esc1_broken"] = paired([score(t, "esc4", 0) for t in tasks], [score(t, "esc1", 0) for t in tasks], k)
    sec["loop4_minus_loop1_broken"] = paired([score(t, "loop4", 0) for t in tasks], [score(t, "loop1", 0) for t in tasks], k)
    res["secondary"] = sec
    return res

def num(text):
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return "unparsed"
    try:
        v = json.loads(m.group(0)).get("answer")
    except Exception:
        return "unparsed"
    if v is None:
        return None
    try:
        return round(float(str(v).replace(",", "").replace("$", "")), 1)
    except ValueError:
        return "unparsed"

def audit(run):
    key = {json.loads(l)["id"]: json.loads(l) for l in open(run / "audit_key_hidden.jsonl")}
    outs = [json.loads(l) for l in open(run / "out-gemma/audit_out.jsonl")]
    by = defaultdict(Counter)
    for o in outs:
        k = key[o["id"]]; g = num(o["text"]); c = k["category"]
        by[c]["n"] += 1
        by[c]["gemma_unparsed"] += g == "unparsed"
        by[c]["gemma_eq_robust"] += g == k["robust"]
        by[c]["gemma_eq_original"] += g == k["original"]
        by[c]["gemma_correct"] += g == k["label"]
        by[c]["robust_correct"] += k["robust"] == k["label"]
        by[c]["original_correct"] += k["original"] == k["label"]
    return {c: {**v, **{f"{m}_rate": round(v[m] / v["n"], 3) for m in ("gemma_eq_robust", "gemma_eq_original", "gemma_correct", "robust_correct")}}
            for c, v in by.items()}

if __name__ == "__main__":
    run = Path(sys.argv[1])
    out = {"contagion": contagion(run), "audit": audit(run)}
    json.dump(out, open(run / "analysis.json", "x"), indent=1)
    print(json.dumps(out, indent=1))
