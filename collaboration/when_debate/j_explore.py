"""Study J, EXPLORATORY (not in PROTOCOL.md): split framing into a label effect and a position effect.
Label effect: share choosing the first-block solution when it is labelled "Your earlier solution" minus when labelled "Solution A",
  at fixed content order (owner vs neutral: judge's own first; label_swap vs neutral_swap: other model's first).
Position: share choosing the first block under neutral labels, pooled over both orders.
Usage: j_explore.py <study J dir>"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from settings_analyze import boot
from j_analyze import ans
f = lambda x: f"{100*x[0]:+.1f} [{100*x[1]:+.1f}, {100*x[2]:+.1f}]"
J = sys.argv[1]; sets = json.load(open(os.path.join(J, "sets.json")))
pool = {"gemma": [], "qwen": []}
for ts, cfg in sets.items():
    S = {}
    for m in ("qwen", "gemma"):
        for l in open(os.path.join(cfg["solve"], f"{m}_solve.jsonl")):
            r = json.loads(l)
            if r["sample"] == 0: S[(m, r["tid"])] = ans(r)
    V = {}
    for m in ("qwen", "gemma"):
        for l in open(os.path.join(J, "out", f"{ts}_{m}.jsonl")):
            r = json.loads(l); V[(m, r["tid"], r["cond"])] = ans(r)
    for m in ("gemma", "qwen"):
        o = "qwen" if m == "gemma" else "gemma"
        D = sorted({t for (mm, t, c) in V if mm == m})
        own = {t: S[(m, t)] for t in D}; oth = {t: S[(o, t)] for t in D}
        # first block: own in owner/neutral, other in label_swap/neutral_swap
        first = lambda c, t: float(V[(m, t, c)] == (own[t] if c in ("owner", "neutral") else oth[t]))
        lab_own = [first("owner", t) - first("neutral", t) for t in D]
        lab_oth = [first("label_swap", t) - first("neutral_swap", t) for t in D]
        lab = [(a + b) / 2 for a, b in zip(lab_own, lab_oth)]
        posA = [(first("neutral", t) + first("neutral_swap", t)) / 2 for t in D]
        pos = [x - 0.5 for x in posA]
        pool[m] += lab
        print(f"{ts:6s} {m:6s} n={len(D):3d}  label 'yours' on own: {f(boot(lab_own))}  on other's: {f(boot(lab_oth))}  "
              f"mean: {f(boot(lab))} | neutral picks block A {100*sum(posA)/len(D):.1f}%  (A - 50: {f(boot(pos))})")
for m, v in pool.items():
    print(f"POOLED {m}: label 'yours' effect on choosing that block {f(boot(v))} (n={len(v)})")
