"""Pre-registered analysis of the debate-settings factorial (reports/debate-settings-20261120-a/PROTOCOL.md).
Usage: settings_analyze.py <dir with S*.json>   -> prints the tables and writes <dir>/../analysis.json"""
import glob, json, os, random, sys
from collections import Counter
sys.path.insert(0, "/home/lian/swarmworld/research/reports/peer-theory-20261119")
sys.path.insert(0, "/home/lian/swarmworld/research/collaboration/dmad_rerun")
from empties import robust
_argv = sys.argv; sys.argv = [sys.argv[0], "/nonexistent"]
from sensitivity_unclosed import unclosed
sys.argv = _argv

B, SEED = 20000, 20261117
CONTRASTS = [("C1", "S1_dov", "S2_dmad"), ("C2", "S3_dmad_dovword", "S2_dmad"), ("C3", "S4_dmad_raw", "S2_dmad"),
             ("C4", "S5_dmad_512", "S2_dmad"), ("C5", "S6_dmad_ansonly", "S2_dmad"), ("C6", "S7_dov_chat", "S1_dov")]


def r1(x):
    return None if x is None else round(float(x), 1)


def reader(family):
    return (lambda t: r1(robust(t))) if family == "dov" else (lambda t: r1(unclosed(t)[0]))


def vote(ans, y):
    ne = [a for a in ans if a is not None]
    if not ne: return 0.0
    c = Counter(ne); top = max(c.values()); w = [k for k, v in c.items() if v == top]
    return (y in w) / len(w)


def top(ans):
    ne = [a for a in ans if a is not None]
    return max(Counter(ne).values()) if ne else 0


def boot(d):
    n = len(d); rng = random.Random(SEED)
    s = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return [sum(d) / n, s[int(.025 * B)], s[int(.975 * B) - 1]]


def holm(ps):
    """ps: dict name->two-sided bootstrap p; returns name->adjusted p."""
    order = sorted(ps, key=ps.get); m = len(order); out, run = {}, 0.0
    for k, name in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[name])); out[name] = run
    return out


def boot_p(d):
    """Two-sided bootstrap p for mean != 0 (share of resampled means on the other side of 0, doubled)."""
    n = len(d); rng = random.Random(SEED); m = sum(d) / n
    s = [sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B)]
    return min(1.0, 2 * sum((x <= 0) if m > 0 else (x >= 0) for x in s) / B)


def load(path):
    D = json.load(open(path)); rd = reader(D["cfg"]["family"])
    T = []
    for ex in D["results"]:
        y = r1(float(str(ex["label"]).replace(",", "")))
        T.append((y, [[rd(t) for t in rnd] for rnd in ex["rounds"]]))
    return D["arm"], T


def summarize(T):
    n = len(T); K = len(T[0][1]) - 1
    v = [[vote(r[k], y) for y, r in T] for k in range(K + 1)]
    ind = [sum(a == y for y, r in T for a in r[k]) / (5 * n) for k in range(K + 1)]
    none = [sum(a is None for _, r in T for a in r[k]) / (5 * n) for k in range(K + 1)]
    g = {th: [v[0][j] if top(T[j][1][0]) >= th else v[K][j] for j in range(n)] for th in (4, 5)}
    pool = [vote(r[0] + r[K], y) for y, r in T]
    strata = {}
    for lab, f in (("5/5", lambda k: k == 5), ("4/5", lambda k: k == 4), ("<=3/5", lambda k: k <= 3)):
        idx = [j for j, (y, r) in enumerate(T) if f(top(r[0]))]
        strata[lab] = {"n": len(idx), "rescued": sum(max(0, v[K][j] - v[0][j]) for j in idx),
                       "broken": sum(max(0, v[0][j] - v[K][j]) for j in idx)}
    return {"n": n, "vote_by_round": [sum(x) / n for x in v], "individual_by_round": ind, "no_answer_by_round": none,
            "P1": boot([a - b for a, b in zip(v[K], v[0])]), "P2_G4": boot([a - b for a, b in zip(g[4], v[0])]),
            "G5": boot([a - b for a, b in zip(g[5], v[0])]), "pool": boot([a - b for a, b in zip(pool, v[0])]),
            "strata": strata, "_d1": [a - b for a, b in zip(v[K], v[0])], "_d2": [a - b for a, b in zip(g[4], v[0])]}


def main():
    d = sys.argv[1]
    S = {}
    for p in sorted(glob.glob(os.path.join(d, "S*.json"))):
        arm, T = load(p); S[arm] = summarize(T)
    f = lambda x: f"{100*x[0]:+.1f} [{100*x[1]:+.1f}, {100*x[2]:+.1f}]"
    for arm, s in S.items():
        print(f"== {arm} n={s['n']}")
        print("   vote by round  " + " ".join(f"{100*x:.1f}" for x in s["vote_by_round"]) +
              " | individual " + " ".join(f"{100*x:.1f}" for x in s["individual_by_round"]) +
              " | no answer % " + " ".join(f"{100*x:.1f}" for x in s["no_answer_by_round"]))
        print(f"   P1 final - r0 vote {f(s['P1'])} | P2 G4 gate {f(s['P2_G4'])} | G5 {f(s['G5'])} | pool {f(s['pool'])}")
        print("   rescued/broken by r0 agreement: " + "; ".join(f"{k} (n={v['n']}): +{v['rescued']:.1f} / -{v['broken']:.1f}"
                                                         for k, v in s["strata"].items()))
    res = {}
    for key in ("_d1", "_d2"):
        ps, est = {}, {}
        for name, a, b in CONTRASTS:
            if a in S and b in S:
                dd = [x - y for x, y in zip(S[a][key], S[b][key])]
                est[name] = (a, b, boot(dd)); ps[name] = boot_p(dd)
        adj = holm({k: v for k, v in ps.items() if k != "C1"})
        label = "P1 (final - r0)" if key == "_d1" else "P2 (G4 gate - r0)"
        print(f"\n== Factor contrasts on {label}")
        for name, (a, b, e) in est.items():
            print(f"   {name} {a} - {b}: {f(e)}  p={ps[name]:.4f}  Holm={adj.get(name, float('nan')):.4f}")
        res[label] = {k: {"arms": [a, b], "est": e, "p": ps[k], "holm": adj.get(k)} for k, (a, b, e) in est.items()}
    for s in S.values():
        s.pop("_d1"); s.pop("_d2")
    json.dump({"arms": S, "contrasts": res}, open(os.path.join(d, "..", "analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
