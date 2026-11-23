"""Study D analysis (reports/du-rerun-20261120-a/PROTOCOL.md). Usage: analyze_du.py <du_model.json>"""
import json, sys, random
from collections import Counter
from score_du import du_agent, du_vote, boxed_last, gt_val, E

B, SEED = 20000, 20261117

def vote(vals, y):
    v = [x for x in vals if x is not None]
    if not v: return 0.0
    c = Counter(v); top = max(c.values()); w = [k for k, n in c.items() if n == top]
    return (y in w) / len(w)

def boot(d):
    n = len(d); rng = random.Random(SEED)
    s = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return 100 * sum(d) / n, 100 * s[int(.025 * B)], 100 * s[int(.975 * B) - 1]

def fnum(a):
    try: return float(a)
    except Exception: return None

def run(path):
    D = json.load(open(path))
    qs = list(D.keys())
    out = {}
    for subset, Q in (("all300", qs), ("first100", qs[:100])):
        res = {}
        per = {"du": [], "val": []}
        stats = []
        for r in range(3):
            du_v, val_v, fb, mis, none_corr, ind_du, ind_val, nval = [], [], 0, 0, 0, 0, 0, 0
            for q in Q:
                ctx, gt = D[q]; y = gt_val(gt)
                sols = [c[2 * r + 1]["content"] for c in ctx]
                acc = du_vote(gt, sols)
                du_v.append(float(acc))
                vals = [boxed_last(s) for s in sols]
                val_v.append(vote(vals, y))
                for s, v in zip(sols, vals):
                    a = du_agent(s)
                    if E.parse_answer(s) is None: fb += 1
                    if a is not None and v is not None and fnum(a) != v: mis += 1
                    ind_du += (fnum(a) == y); ind_val += (v == y); nval += (v is None)
                # authors' vote-level "None -> correct"
                pa = [du_agent(s) for s in sols]
                if E.most_frequent(pa) is None: none_corr += 1
            k = 3 * len(Q)
            stats.append({"round": r, "vote_du": sum(du_v) / len(Q), "vote_val": sum(val_v) / len(Q),
                          "ind_du": ind_du / k, "ind_val": ind_val / k, "fallback_pct": 100 * fb / k,
                          "misread_pct": 100 * mis / k, "val_none_pct": 100 * nval / k, "none_scored_correct": none_corr})
            per["du"].append(du_v); per["val"].append(val_v)
        res["by_round"] = stats
        for sc in ("du", "val"):
            res[f"P_{sc}_r2_minus_r0"] = boot([a - b for a, b in zip(per[sc][2], per[sc][0])])
        out[subset] = res
    return out

if __name__ == "__main__":
    o = run(sys.argv[1])
    for sub, res in o.items():
        print(f"== {sys.argv[1].split('/')[-1]} {sub}")
        for s in res["by_round"]:
            print("  r{round}: vote du {vote_du:.3f} val {vote_val:.3f} | ind du {ind_du:.3f} val {ind_val:.3f} | fallback {fallback_pct:.1f}% misread {misread_pct:.1f}% val-none {val_none_pct:.1f}% none->correct {none_scored_correct}".format(**s))
        for sc in ("du", "val"):
            e = res[f"P_{sc}_r2_minus_r0"]; print(f"  {sc:3s} vote r2-r0 {e[0]:+.1f} [{e[1]:+.1f}, {e[2]:+.1f}]")
    json.dump(o, open(sys.argv[1].replace(".json", "_analysis.json"), "w"), indent=1)
