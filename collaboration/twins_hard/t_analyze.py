"""Pre-registered analysis of study T (reports/twins-hard-20261121-a/PROTOCOL.md).
Usage: t_analyze.py <out dir with {qwen,gemma}_review.jsonl> <stimuli.json> <labels.json> <qwen_solve> <gemma_solve>"""
import json, os, random, sys
sys.path.insert(0, "/home/lian/swarmworld/research/collaboration/when_debate")
from settings_analyze import boot, boot_p, holm

GSM = {"T1 E_full - private": -14.1, "T2b E_full - E_answer": None, "T3 E_confident - E_hedged": -12.0,
       "T4 indep net": +3.0, "T4 check net": +3.2, "T4 warn net": +0.7, "T5 C_full - C_answer": None}


def boot90(d):
    n = len(d); rng = random.Random(20261117)
    s = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(20000))
    return s[int(.05 * 20000)], s[int(.95 * 20000) - 1]


def main():
    d, stim_p, lab_p, qs, gs = sys.argv[1:6]
    stim = {int(k): v for k, v in json.load(open(stim_p)).items()}
    Y = {int(k): v for k, v in json.load(open(lab_p)).items()}
    f = lambda x: f"{100*x[0]:+.1f} [{100*x[1]:+.1f}, {100*x[2]:+.1f}]"
    out = {}
    for m, sp in (("qwen", qs), ("gemma", gs)):
        p = os.path.join(d, f"{m}_review.jsonl")
        if not os.path.exists(p): continue
        own = {}
        for l in open(sp):
            r = json.loads(l)
            if r["sample"] == 0: own[r["tid"]] = None if r["parsed"] is None else r["parsed"]["answer"]
        R = {}
        pad_dev = 0
        for l in open(p):
            r = json.loads(l); R[(r["tid"], r["cond"])] = r["answer"]; pad_dev = max(pad_dev, abs(r["prompt_tokens"] - r["target_tokens"]))
        conds = sorted({c for _, c in R}); T = sorted({t for t, _ in R if all((t, c) in R for c in conds)})
        acc = {c: [float(R[(t, c)] == Y[t]) for t in T] for c in conds}
        right = [t for t in T if own[t] == Y[t]]
        print(f"\n== recipient {m}: {len(T)} tasks; own first answer right {len(right)}; max padding deviation {pad_dev} tokens; "
              f"invalid answers {sum(v is None for v in R.values())}")
        print("   " + "  ".join(f"{c} {100*sum(v)/len(T):.1f}" for c, v in acc.items()))
        rows = {}
        for c in conds:
            if c == "private": continue
            tw = c[0]
            corr = sum(R[(t, c)] != Y[t] for t in right); fol = sum(R[(t, c)] == stim[t]["E"]["answer"] for t in right) if tw == "E" else None
            rows[c] = {"corruption|own right": corr / len(right), "followed E answer|own right": None if fol is None else fol / len(right)}
        print("   corruption | own right: " + "  ".join(f"{c} {100*v['corruption|own right']:.1f}%" for c, v in rows.items()))
        print("   adopts E's answer | own right: " + "  ".join(f"{c} {100*v['followed E answer|own right']:.1f}%" for c, v in rows.items() if v["followed E answer|own right"] is not None))
        dif = lambda a, b: [x - y for x, y in zip(acc[a], acc[b])]
        net = lambda s: [(a + b) / 2 - (c + e) / 2 for a, b, c, e in zip(acc[f"C_full_{s}"], acc[f"E_full_{s}"], acc["C_full"], acc["E_full"])]
        tests = {"T1 E_full - private": dif("E_full", "private"), "T2a E_answer - private": dif("E_answer", "private"),
                 "T2b E_full - E_answer": dif("E_full", "E_answer"), "T5 C_full - C_answer": dif("C_full", "C_answer"),
                 "T1c C_full - private": dif("C_full", "private")}
        for s in ("indep", "check", "warn"):
            if f"E_full_{s}" in acc:
                tests[f"T4 {s} net"] = net(s)
                tests[f"T4 {s}: E"] = dif(f"E_full_{s}", "E_full"); tests[f"T4 {s}: C"] = dif(f"C_full_{s}", "C_full")
        if "E_confident" in acc: tests["T3 E_confident - E_hedged"] = dif("E_confident", "E_hedged")
        prim = [k for k in ("T1 E_full - private", "T2b E_full - E_answer", "T3 E_confident - E_hedged", "T4 indep net", "T5 C_full - C_answer") if k in tests]
        ps = {k: boot_p(tests[k]) for k in prim}; adj = holm(ps)
        res = {"n": len(T), "acc": {c: sum(v) / len(T) for c, v in acc.items()}, "mechanism": rows, "tests": {}}
        for k, dd in tests.items():
            e = boot(dd); res["tests"][k] = {"est": e}
            extra = ""
            if k in ps: res["tests"][k].update(p=ps[k], holm=adj[k]); extra = f"  p={ps[k]:.4f} Holm={adj[k]:.4f}  PRIMARY"
            if k == "T2a E_answer - private":
                lo, hi = boot90(dd); res["tests"][k]["ci90"] = [lo, hi]; extra = f"  90% CI [{100*lo:+.1f}, {100*hi:+.1f}] -> {'equivalent within 2 pp' if -.02 < lo and hi < .02 else 'not shown equivalent'}"
            ref = GSM.get(k); extra += f"  (GSM8K {ref:+.1f})" if ref is not None else ""
            print(f"   {k:28s} {f(e)}{extra}")
        out[m] = res
    json.dump(out, open(os.path.join(d, "..", "analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
