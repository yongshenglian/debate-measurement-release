"""Study K analysis (reports/framing-rep-20261122-a/PROTOCOL.md). Usage: k_analyze.py <study K dir> [out subdir, default out]
Unit of analysis: the problem. Per problem and framing, outcomes are averaged over the three decoding replicates.
Primary tests: two-sided sign-flip randomization test on per-problem differences (100,000 draws), Holm over K1-K5.
Intervals: percentile bootstrap over problems (20,000 draws). Sensitivity: replicate-0-only exact McNemar with transition counts."""
import json, os, sys
import numpy as np
from collections import Counter
from math import comb

SEED, B, BP = 20261122, 20000, 100000
FR = ("owner", "neutral", "yours", "another", "yoursol", "nswap", "misattr", "owner_swap")
REPS = (0, 1, 2)


def mean(x): return sum(x) / len(x)


def boot(d):
    rng = np.random.default_rng(SEED); x = np.asarray(d, float); n = len(x)
    s = np.sort(np.concatenate([x[rng.integers(0, n, (1000, n))].mean(1) for _ in range(B // 1000)]))
    return float(x.mean()), float(s[int(.025 * B)]), float(s[int(.975 * B) - 1])


def signflip(d):
    rng = np.random.default_rng(SEED + 1); x = np.asarray(d, float); obs = abs(x.sum()); hit = 0
    for _ in range(BP // 1000):
        sg = rng.integers(0, 2, (1000, len(x))) * 2 - 1
        hit += int((np.abs((sg * x).sum(1)) >= obs - 1e-9).sum())
    return (1 + hit) / (1 + BP)


def mcnemar(a, b):
    """exact two-sided McNemar on paired binary outcomes; returns (b only, a only... ) counts and p"""
    up = sum(x and not y for x, y in zip(a, b)); dn = sum(y and not x for x, y in zip(a, b))
    n, k = up + dn, min(up, dn)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return up, dn, p


def holm(ps):
    order = sorted(ps, key=ps.get); m = len(order); out, run = {}, 0.0
    for k, name in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[name])); out[name] = run
    return out


def vote(xs, tie_first=True):
    xs = [x for x in xs if x is not None]
    if not xs: return None
    c = Counter(xs); top = max(c.values())
    return next(x for x in xs if c[x] == top)   # ties -> earliest listed (Gemma s0 first)


def ans(r): return None if r is None or r.get("parsed") is None else r["parsed"]["answer"]


f3 = lambda x: f"{100*x[0]:+.1f} [{100*x[1]:+.1f}, {100*x[2]:+.1f}]"


def main():
    K = sys.argv[1]; O = os.path.join(K, sys.argv[2] if len(sys.argv) > 2 else "out")
    Y = {int(k): v for k, v in json.load(open(os.path.join(K, "data", "labels.json"))).items()}
    S, TOK = {}, Counter()
    for fn, m in (("gemma_solve_s0", "gemma"), ("gemma_solve_s1", "gemma"), ("gemma_solve_s2", "gemma"), ("qwen_solve", "qwen")):
        for l in open(os.path.join(O, fn + ".jsonl")):
            r = json.loads(l); S[(m, r["tid"], r["sample"])] = ans(r)
            TOK[(m, "solve")] += r["prompt_tokens"] + r["output_tokens"] + r["prompt_tokens_2"] + r["output_tokens_2"]
    T = sorted(Y)
    V, JT = {}, Counter()
    for s in range(4):
        p = os.path.join(O, f"judge_{s}.jsonl")
        if not os.path.exists(p): continue
        for l in open(p):
            r = json.loads(l); V[(r["tid"], r["cond"], r["rep"])] = ans(r)
            JT[r["cond"]] += r["prompt_tokens"] + r["output_tokens"] + r["prompt_tokens_2"] + r["output_tokens_2"]
    Dall = sorted(t for t in T if S.get(("gemma", t, 0)) is not None and S.get(("qwen", t, 0)) is not None
                  and S[("gemma", t, 0)] != S[("qwen", t, 0)])
    D = [t for t in Dall if all((t, f, r) in V for f in FR for r in REPS)]
    print(f"problems {len(T)}; invalid first answers: gemma {sum(S.get(('gemma', t, 0)) is None for t in T)}, "
          f"qwen {sum(S.get(('qwen', t, 0)) is None for t in T)}; disagreement set D {len(Dall)}, complete in judge output {len(D)}")
    inval = Counter(f for (t, f, r), a in V.items() if a is None)
    print(f"invalid judge answers by framing: {dict(inval)}")
    g0 = {t: S[("gemma", t, 0)] for t in D}; q0 = {t: S[("qwen", t, 0)] for t in D}
    cells = Counter("g_right" if g0[t] == Y[t] else "q_right" if q0[t] == Y[t] else "both_wrong" for t in D)
    print(f"D cells: {dict(cells)}")
    acc = {f: {t: mean([V[(t, f, r)] == Y[t] for r in REPS]) for t in D} for f in FR}
    keep = {f: {t: mean([V[(t, f, r)] == g0[t] for r in REPS]) for t in D} for f in FR}
    take = {f: {t: mean([V[(t, f, r)] == q0[t] for r in REPS]) for t in D} for f in FR}
    print("\naccuracy on D:     " + "  ".join(f"{f} {100*mean(list(acc[f].values())):.1f}" for f in FR))
    print("final = own (g0):  " + "  ".join(f"{f} {100*mean(list(keep[f].values())):.1f}" for f in FR))
    print("final = other (q0):" + "  ".join(f"{f} {100*mean(list(take[f].values())):.1f}" for f in FR))

    prim = {"K1 accuracy, neutral - owner": (acc, "neutral", "owner", lambda t, f, r: V[(t, f, r)] == Y[t]),
            "K2 keeps own, yours - neutral": (keep, "yours", "neutral", lambda t, f, r: V[(t, f, r)] == g0[t]),
            "K3 takes other, another - neutral": (take, "another", "neutral", lambda t, f, r: V[(t, f, r)] == q0[t]),
            "K4 keeps own, yoursol - yours": (keep, "yoursol", "yours", lambda t, f, r: V[(t, f, r)] == g0[t]),
            "K5 takes other, misattr - nswap": (take, "misattr", "nswap", lambda t, f, r: V[(t, f, r)] == q0[t])}
    P, est, out = {}, {}, {}
    print("\n######## primary (per-problem means over 3 replicates; sign-flip p; bootstrap CI)")
    for k, (M, a, b, ind) in prim.items():
        d = [M[a][t] - M[b][t] for t in D]
        P[k] = signflip(d); est[k] = boot(d)
        mc = [mcnemar([ind(t, a, r) for t in D], [ind(t, b, r) for t in D]) for r in REPS]
        out[k] = {"est": est[k], "p": P[k], "mcnemar_by_rep": mc}
        print(f"{k}: {f3(est[k])}  sign-flip p={P[k]:.4f}  | McNemar per replicate (a-only, b-only, p): {mc}")
    for k, p in holm(P).items():
        out[k]["holm"] = p
        print(f"PRIMARY {k}: {f3(est[k])}  p={P[k]:.4f}  Holm={p:.4f}")

    print("\n######## secondary")
    for nm, a, b, M in (("position, neutral: keeps own when own first - own second", "neutral", "nswap", keep),
                        ("position, owner labels: own first - own second", "owner", "owner_swap", keep),
                        ("accuracy, (neutral + nswap)/2 - owner", None, None, None)):
        if a:
            d = [M[a][t] - M[b][t] for t in D]
        else:
            d = [(acc["neutral"][t] + acc["nswap"][t]) / 2 - acc["owner"][t] for t in D]
        print(f"{nm}: {f3(boot(d))}  sign-flip p={signflip(d):.4f}")
    Rg = [t for t in D if g0[t] == Y[t]]; Rq = [t for t in D if q0[t] == Y[t]]; Bw = [t for t in D if t not in Rg and t not in Rq]
    for f in FR:
        ab = mean([1 - acc[f][t] for t in Rg]) if Rg else float("nan")
        ac = mean([acc[f][t] for t in Rq]) if Rq else float("nan")
        rc = mean([acc[f][t] for t in Bw]) if Bw else float("nan")
        print(f"[{f}] abandons own right {100*ab:.1f}% (n={len(Rg)}) | accepts other's right {100*ac:.1f}% (n={len(Rq)}) | "
              f"recovers when both wrong {100*rc:.1f}% (n={len(Bw)})")

    # systems over ALL problems (common denominator; an invalid answer counts as wrong). Each uses 3 generation calls.
    Dset = set(D)
    g = lambda t, s: S.get(("gemma", t, s)); q = lambda t: S.get(("qwen", t, 0))
    base = {"gemma_s0": [g(t, 0) == Y[t] for t in T],
            "vote3_mixed (g0,g1,q0)": [vote([g(t, 0), g(t, 1), q(t)]) == Y[t] for t in T],
            "vote3_gemma (g0,g1,g2)": [vote([g(t, 0), g(t, 1), g(t, 2)]) == Y[t] for t in T],
            "oracle g0 or q0": [g(t, 0) == Y[t] or q(t) == Y[t] for t in T]}
    print(f"\nsystems over all {len(T)} problems (incomplete-D problems use Gemma s0):")
    for k, v in base.items(): print(f"   {k}: {100*mean(v):.1f}")
    for f in ("neutral", "owner"):
        sysv = [mean([V[(t, f, r)] == Y[t] for r in REPS]) if t in Dset else float(g(t, 0) == Y[t]) for t in T]
        line = f"   judge [{f}] on D, Gemma s0 elsewhere: {100*mean(sysv):.1f}"
        for k in ("gemma_s0", "vote3_mixed (g0,g1,q0)", "vote3_gemma (g0,g1,g2)"):
            d = [a - float(b) for a, b in zip(sysv, base[k])]
            line += f" | - {k} {f3(boot(d))}"
        print(line)
    print("\ntokens (prompt + output, both turns), total: " + ", ".join(f"{k[0]} {k[1]} {v:,}" for k, v in TOK.items()))
    print("judge tokens by framing: " + ", ".join(f"{k} {v:,}" for k, v in JT.items()))
    json.dump(out, open(os.path.join(O, "analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
