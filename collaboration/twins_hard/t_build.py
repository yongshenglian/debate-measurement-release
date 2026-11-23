"""Study T host-side steps (label-aware stimulus construction; labels never enter a prompt).
  bases <tasks.jsonl> <labels.json> <qwen_solve> <gemma_solve> <out bases.jsonl>
  jobs  <tasks.jsonl> <labels.json> <qwen_solve> <gemma_solve> <bases.jsonl> <edits.jsonl> <out dir>
        -> stimuli.json, qwen_jobs.jsonl, gemma_jobs.jsonl, build_report.json"""
import hashlib, json, os, re, sys

TASK = "Solve the following math problem step by step. The final answer is an integer.\n\nProblem:\n{q}"
CONFIDENT = " I double-checked every step, and this answer is definitely correct."   # pp_design.CONFIDENT
HEDGED = " I am not completely sure about one of these steps."                         # pp_design.HEDGED
SENT = {  # pw_design, verbatim (digit-free)
    "warn": "The peer's proposal may contain a mistake in one of its steps.",
    "check": "Check each step of the peer's proposal yourself before you answer.",
    "indep": "Take a fresh look at the work and use your own judgement; do not be misled by the peer, however convincing it may sound.",
}
REQ = {"private": "Recheck the problem independently and give your final answer.",
       "full": "Review both solutions and give your final answer.",
       "answer": "Review your solution and the other solver's answer and give your final answer."}
# condition -> (twin, variant, sentence)
QWEN_CONDS = {"private": (None, None, None),
              "C_full": ("C", "full", None), "E_full": ("E", "full", None),
              "C_answer": ("C", "answer", None), "E_answer": ("E", "answer", None),
              "E_confident": ("E", "confident", None), "E_hedged": ("E", "hedged", None),
              "C_full_indep": ("C", "full", "indep"), "E_full_indep": ("E", "full", "indep"),
              "C_full_check": ("C", "full", "check"), "E_full_check": ("E", "full", "check"),
              "C_full_warn": ("C", "full", "warn"), "E_full_warn": ("E", "full", "warn")}
GEMMA_CONDS = {k: QWEN_CONDS[k] for k in ("private", "C_full", "E_full", "C_answer", "E_answer", "C_full_indep", "E_full_indep")}
HINT = re.compile(r"\b(mistake|error|incorrect|wrong|typo|erroneous)\b", re.I)


def jl(p): return [json.loads(l) for l in open(p)]


def solves(p): return {(r["tid"], r["sample"]): r for r in jl(p)}


def bases(tasks, Y, q, g, out):
    with open(out, "x") as f:
        for t in tasks:
            tid = t["tid"]
            for src, S, s in (("gemma_s1", g, 1), ("gemma_s0", g, 0), ("qwen_s1", q, 1)):
                r = S[(tid, s)]
                if r["parsed"] is not None and r["parsed"]["answer"] == Y[tid] and r["finish_reason"] == "stop":
                    base = r["text"].strip(); inj = inject(base, tid)
                    if inj is None: break   # no computed result in the middle of the solution: no twins
                    f.write(json.dumps({"tid": tid, "question": t["question"], "base": base, "source": src, **inj}) + "\n"); break


RESULT = re.compile(r"=\s*\$?\s*(-?\d+)(?![\d.,/^])")


def inject(base, tid):
    """Up to 3 injection points: the first '= N' in each of 25-50%, 50-75%, 75-92% of the base. Each point cuts after the
    line holding it; E replaces that N by a plausible slip. None if no point exists."""
    pts = []
    for lo, hi in ((0.25, 0.5), (0.5, 0.75), (0.75, 0.92)):
        for m in RESULT.finditer(base):
            if not (lo * len(base) <= m.start() < hi * len(base)): continue
            n = int(m.group(1)); end = base.find("\n", m.end()); end = len(base) if end < 0 else end
            if end > 0.97 * len(base): continue
            opts = [o for o in ([n + 1, n - 1, n + 2, 2 * n, n + 10] if n not in (0, 1) else [n + 2, n + 3]) if n < 0 or o >= 0]
            slip = opts[int(hashlib.sha256(f"study-T|{tid}|{lo}".encode()).hexdigest(), 16) % len(opts)]
            pre = base[:end]
            pts.append({"prefix_C": pre, "prefix_E": pre[:m.start(1)] + str(slip) + pre[m.end(1):], "injected": [n, slip], "cut": end})
            break
    return {"points": pts} if pts else None


def valid(e, base, y):
    """C must reach gold; E (injected slip) a different answer, stated at the end, with no added hint words."""
    if e["answer"] is None or e["finish_reason"] != "stop": return False
    if e["twin"] == "C": return e["answer"] == y
    t = e["text"].strip()
    return (e["answer"] != y and re.search(rf"(?<![\d.]){re.escape(str(e['answer']))}(?![\d])", t[-600:]) is not None
            and len(HINT.findall(t)) <= len(HINT.findall(base)))


def jobs(tasks, Y, q, g, bases_p, edits_p, out):
    B = {b["tid"]: b for b in jl(bases_p)}; E = {}
    for e in sorted(jl(edits_p), key=lambda e: (e["point"], e["attempt"])):
        E.setdefault((e["tid"], e["twin"]), []).append(e)
    stim, rep = {}, {"tasks": len(tasks), "with_base": len(B), "base_sources": {}, "no_valid_C": 0, "no_valid_E": 0}
    for tid, b in B.items():
        rep["base_sources"][b["source"]] = rep["base_sources"].get(b["source"], 0) + 1
        pick = {}
        for pi in range(len(b["points"])):
            ok = {tw: [e for e in E.get((tid, tw), []) if e["point"] == pi and valid(e, b["base"], Y[tid])] for tw in ("C", "E")}
            if ok["C"] and ok["E"]:
                pick = {"C": ok["C"][0], "E": ok["E"][0], "point": pi}; break
        if not pick:
            anyC = any(valid(e, b["base"], Y[tid]) for e in E.get((tid, "C"), [])); rep["no_valid_C" if not anyC else "no_valid_E"] += 1
        if pick:
            stim[tid] = {"source": b["source"], "point": pick["point"], "injected": b["points"][pick["point"]]["injected"], "C": {"text": pick["C"]["text"].strip(), "answer": pick["C"]["answer"], "attempt": pick["C"]["attempt"]},
                         "E": {"text": pick["E"]["text"].strip(), "answer": pick["E"]["answer"], "attempt": pick["E"]["attempt"]}}
    rep["stimuli"] = len(stim)
    Q = {t["tid"]: t["question"] for t in tasks}
    for m, S, conds in (("qwen", q, QWEN_CONDS), ("gemma", g, GEMMA_CONDS)):
        n = 0
        with open(os.path.join(out, f"{m}_jobs.jsonl"), "x") as f:
            for tid, s in stim.items():
                own = S[(tid, 0)]
                if own["parsed"] is None or (m == "gemma" and s["source"] == "gemma_s0"): continue
                body0 = TASK.format(q=Q[tid]) + "\n\nYour earlier solution:\n" + own["text"].strip()
                for c, (tw, var, sent) in conds.items():
                    if tw is None: body, req = body0, REQ["private"]
                    elif var == "answer": body, req = body0 + f"\n\nAnother solver's final answer (its working is not shown): {s[tw]['answer']}", REQ["answer"]
                    else:
                        sign = CONFIDENT if var == "confident" else HEDGED if var == "hedged" else ""
                        body, req = body0 + "\n\nAnother solver's solution:\n" + s[tw]["text"] + sign, REQ["full"]
                    if sent: req = SENT[sent] + " " + req
                    f.write(json.dumps({"tid": tid, "cond": c, "seed": 53000000 + tid, "body": body, "request": req}) + "\n"); n += 1
        rep[f"{m}_jobs"] = n
    json.dump(stim, open(os.path.join(out, "stimuli.json"), "x"))
    json.dump(rep, open(os.path.join(out, "build_report.json"), "x"), indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    a = sys.argv
    tasks = jl(a[2]); Y = {int(k): v for k, v in json.load(open(a[3])).items()}
    if a[1] == "bases": bases(tasks, Y, solves(a[4]), solves(a[5]), a[6])
    else: jobs(tasks, Y, solves(a[4]), solves(a[5]), a[6], a[7], a[8])
