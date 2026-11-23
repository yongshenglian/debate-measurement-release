"""Study J analysis (reports/neutral-judge-20261121-a/PROTOCOL.md).
Usage: j_analyze.py <study J dir>   (reads <dir>/sets.json: taskset -> {solve dir, labels, original review dir})"""
import json, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from settings_analyze import boot, boot_p, holm

CONDS = ("owner", "neutral", "neutral_swap", "label_swap")
f = lambda x: f"{100*x[0]:+.1f} [{100*x[1]:+.1f}, {100*x[2]:+.1f}]"


def ans(r):
    return None if r is None or r.get("parsed") is None else r["parsed"]["answer"]


def vote3(g0, g1, q0):
    xs = [x for x in (g0, g1, q0) if x is not None]
    c = Counter(xs)
    if not c: return None
    top = max(c.values())
    for x in (g0, g1, q0):   # ties broken toward Gemma s0, then Gemma s1
        if x is not None and c[x] == top: return x


def main():
    J = sys.argv[1]; sets = json.load(open(os.path.join(J, "sets.json")))
    P1, P2, out = {}, {}, {}
    for ts, cfg in sets.items():
        Y = {int(k): v for k, v in json.load(open(cfg["labels"])).items()}
        S = {}
        for m in ("qwen", "gemma"):
            for l in open(os.path.join(cfg["solve"], f"{m}_solve.jsonl")):
                r = json.loads(l); S[(m, r["tid"], r["sample"])] = ans(r)
        orig = {}
        for m in ("qwen", "gemma"):
            for l in open(os.path.join(cfg["orig"], f"{m}_review.jsonl")):
                r = json.loads(l); orig[(m, r["tid"], r["cond"])] = ans(r)
        V = {}
        for m in ("qwen", "gemma"):
            for l in open(os.path.join(J, "out", f"{ts}_{m}.jsonl")):
                r = json.loads(l); V[(m, r["tid"], r["cond"])] = ans(r)
        print(f"\n######## {ts}")
        for m in ("gemma", "qwen"):
            o = "qwen" if m == "gemma" else "gemma"
            D = sorted({t for (mm, t, c) in V if mm == m})
            inval = {c: sum(V.get((m, t, c)) is None for t in D) for c in CONDS}
            own = {t: S[(m, t, 0)] for t in D}; oth = {t: S[(o, t, 0)] for t in D}
            acc = {c: [float(V.get((m, t, c)) == Y[t]) for t in D] for c in CONDS}
            rep = sum(V.get((m, t, "owner")) == orig.get((m, t, f"peer_{o}")) for t in D) / len(D)
            nr, nw = sum(own[t] == Y[t] for t in D), sum(oth[t] == Y[t] for t in D)
            print(f"== judge {m}: disagreements {len(D)} (own right {nr}, other right {nw}, neither {len(D)-nr-nw}); invalid {inval}")
            print(f"   owner rerun = original review answer on {100*rep:.1f}%")
            print("   accuracy on disagreements: " + "  ".join(f"{c} {100*sum(v)/len(D):.1f}" for c, v in acc.items()))
            j1 = [a - b for a, b in zip(acc["neutral"], acc["owner"])]
            P1[f"{ts}:{m}"] = boot_p(j1)
            out[f"J1 {ts}:{m}"] = boot(j1)
            print(f"   J1 neutral - owner: {f(out[f'J1 {ts}:{m}'])}")
            avg = [(a + b) / 2 - c for a, b, c in zip(acc["neutral"], acc["neutral_swap"], acc["owner"])]
            print(f"   (neutral + neutral_swap)/2 - owner: {f(boot(avg))}")
            print(f"   label_swap - owner: {f(boot([a - b for a, b in zip(acc['label_swap'], acc['owner'])]))}")
            # mechanism: how often the final answer is the judge's own first answer, by framing
            keep = {c: [float(V.get((m, t, c)) == own[t]) for t in D] for c in CONDS}
            print("   final = judge's own first answer: " + "  ".join(f"{c} {100*sum(v)/len(D):.1f}%" for c, v in keep.items()))
            print("   final = other model's answer:     " + "  ".join(
                f"{c} {100*sum(V.get((m, t, c)) == oth[t] for t in D)/len(D):.1f}%" for c in CONDS))
            j3 = [a - b for a, b in zip(keep["label_swap"], keep["owner"])]
            print(f"   J3 retention of own answer, label_swap - owner: {f(boot(j3))}")
            print(f"   position: retention neutral (own = A) - neutral_swap (own = B): "
                  f"{f(boot([a - b for a, b in zip(keep['neutral'], keep['neutral_swap'])]))}")
            # accept / abandon on the two informative cells
            for c in CONDS:
                Rw = [t for t in D if own[t] == Y[t]]; Wr = [t for t in D if oth[t] == Y[t]]
                ab = sum(V.get((m, t, c)) != Y[t] for t in Rw); ac = sum(V.get((m, t, c)) == Y[t] for t in Wr)
                print(f"   [{c}] abandons own right answer {ab}/{len(Rw)} = {100*ab/max(1,len(Rw)):.1f}%  |  "
                      f"accepts other's right answer {ac}/{len(Wr)} = {100*ac/max(1,len(Wr)):.1f}%")
            # system accuracy over all problems: judge's answer on disagreements, otherwise Gemma s0
            T = [t for t in Y if S[("gemma", t, 0)] is not None]
            Dset = set(D)
            g0 = [float(S[("gemma", t, 0)] == Y[t]) for t in T]
            v3 = [float(vote3(S[("gemma", t, 0)], S[("gemma", t, 1)], S[("qwen", t, 0)]) == Y[t]) for t in T]
            for c in ("neutral", "owner"):
                sysacc = [float((V.get((m, t, c)) if t in Dset else S[("gemma", t, 0)]) == Y[t]) for t in T]
                a = [x - y for x, y in zip(sysacc, g0)]; b = [x - y for x, y in zip(sysacc, v3)]
                print(f"   system [{c} judge on disagreements] {100*sum(sysacc)/len(T):.1f} | Gemma s0 {100*sum(g0)/len(T):.1f} "
                      f"| vote3 {100*sum(v3)/len(T):.1f} | - Gemma s0 {f(boot(a))} | - vote3 {f(boot(b))}")
                if c == "neutral" and m == "gemma":
                    out[f"J2a {ts}"], out[f"J2b {ts}"] = boot(a), boot(b)
                    P2[f"J2a {ts}"], P2[f"J2b {ts}"] = boot_p(a), boot_p(b)
    print("\n######## primary")
    for k, p in holm(P1).items():
        print(f"PRIMARY J1 {k}: {f(out['J1 ' + k])}  p={P1[k]:.4f}  Holm={p:.4f}")
    for k, p in holm(P2).items():
        print(f"PRIMARY {k}: {f(out[k])}  p={P2[k]:.4f}  Holm={p:.4f}")
    json.dump(out, open(os.path.join(J, "analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
