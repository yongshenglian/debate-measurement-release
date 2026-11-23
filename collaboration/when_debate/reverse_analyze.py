"""Pre-registered analysis of the reverse peer test (reports/reverse-peer-20261120-a/PROTOCOL.md).
Usage: reverse_analyze.py <dir with qwen_solve.jsonl gemma_solve.jsonl qwen_review.jsonl gemma_review.jsonl> <labels.json>"""
import json, os, random, sys

B, SEED = 20000, 20261117


def boot(d):
    n = len(d); rng = random.Random(SEED)
    s = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [sum(d) / n, s[int(.025 * B)], s[int(.975 * B) - 1]]


def ans(rec):
    return None if rec is None or rec.get("parsed") is None else rec["parsed"]["answer"]


def main():
    d, lab_path = sys.argv[1:3]
    Y = {int(k): v for k, v in json.load(open(lab_path)).items()}
    sol = {}
    for m in ("qwen", "gemma"):
        for l in open(os.path.join(d, f"{m}_solve.jsonl")):
            r = json.loads(l); sol[(m, r["tid"], r["sample"])] = ans(r)
    rev = {}
    for m in ("qwen", "gemma"):
        for l in open(os.path.join(d, f"{m}_review.jsonl")):
            r = json.loads(l); rev[(m, r["tid"], r["cond"])] = ans(r)
    f = lambda x: f"{100*x[0]:+.1f} [{100*x[1]:+.1f}, {100*x[2]:+.1f}]"
    pct = lambda a, b: f"{100*a/b:.1f}% ({a}/{b})" if b else "n/a"
    out = {}
    for m in ("qwen", "gemma"):
        other = "gemma" if m == "qwen" else "qwen"
        n_all = len(Y)
        inval = {"solve_s0": sum(sol[(m, t, 0)] is None for t in Y), "solve_s1": sum(sol[(m, t, 1)] is None for t in Y)}
        T = [t for t in Y if sol[(m, t, 0)] is not None
             and all((m, t, c) in rev for c in ("self", "peer_qwen", "peer_gemma"))]
        inval["review"] = {c: sum(rev.get((m, t, c)) is None for t in T) for c in ("self", "peer_qwen", "peer_gemma")}
        first = {t: sol[(m, t, 0)] == Y[t] for t in T}
        acc = {c: [float(rev[(m, t, c)] == Y[t]) for t in T] for c in ("self", "peer_qwen", "peer_gemma")}
        print(f"\n== reviewer {m}: {len(T)} of {n_all} tasks analysed; invalid outputs {inval}")
        print(f"   first answer {100*sum(first.values())/len(T):.1f}%  |  " +
              "  ".join(f"{c} {100*sum(v)/len(T):.1f}%" for c, v in acc.items()))
        res = {"n": len(T), "invalid": inval, "first": sum(first.values()) / len(T),
               "acc": {c: sum(v) / len(T) for c, v in acc.items()}}
        for c in ("peer_qwen", "peer_gemma"):
            res[f"{c}-self"] = boot([a - b for a, b in zip(acc[c], acc["self"])])
            print(f"   {c} - self: {f(res[f'{c}-self'])}")
        res["self-first"] = boot([a - float(first[t]) for a, t in zip(acc["self"], T)])
        print(f"   self - first answer: {f(res['self-first'])}")
        # mechanism: corruption / rescue / following, by condition and peer correctness
        mech = {}
        for c, pm in (("self", None), ("peer_qwen", "qwen"), ("peer_gemma", "gemma")):
            ps = (lambda t: sol[(pm, t, 1 if pm == m else 0)]) if pm else (lambda t: None)
            row = {}
            R = [t for t in T if first[t]]; W = [t for t in T if not first[t]]
            row["corruption"] = (sum(rev[(m, t, c)] != Y[t] for t in R), len(R))
            row["rescue"] = (sum(rev[(m, t, c)] == Y[t] for t in W), len(W))
            if pm:
                Rw = [t for t in R if ps(t) is not None and ps(t) != Y[t]]
                Rc = [t for t in R if ps(t) == Y[t]]
                Wc = [t for t in W if ps(t) == Y[t]]
                row["corruption|peer wrong"] = (sum(rev[(m, t, c)] != Y[t] for t in Rw), len(Rw))
                row["followed wrong peer|own right"] = (sum(rev[(m, t, c)] == ps(t) for t in Rw), len(Rw))
                row["corruption|peer right"] = (sum(rev[(m, t, c)] != Y[t] for t in Rc), len(Rc))
                row["rescue|peer right"] = (sum(rev[(m, t, c)] == Y[t] for t in Wc), len(Wc))
            mech[c] = row
            print(f"   [{c}] " + "; ".join(f"{k}: {pct(*v)}" for k, v in row.items()))
        res["mechanism"] = mech
        out[m] = res
    print(f"\nPRIMARY R1 (gemma reads qwen): {f(out['gemma']['peer_qwen-self'])}")
    print(f"MIRROR  R2 (qwen reads gemma): {f(out['qwen']['peer_gemma-self'])}")
    json.dump(out, open(os.path.join(d, "..", "analysis.json") if d.endswith("out") else os.path.join(d, "analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
