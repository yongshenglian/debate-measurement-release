"""Study C analysis (reports/wynn-rerun-20261120-a/PROTOCOL.md). Run with the wynn env, PYTHONPATH=<talk-aint-cheap copy>.
Usage: analyze_wynn.py <out_dir> [config ...]"""
import json, re, sys, os, random, glob
from collections import Counter
from src.analysis.evaluation import compute_accuracy, parse_gsm8k_answer, solve_math_problems

B, SEED = 20000, 20261117
NUM = r"-?[0-9][0-9,]*(?:\.[0-9]+)?"

def boxed_last(text):
    """Validated: content of the last (optionally backslash-less) boxed{...} with balanced braces, one number."""
    out = None
    for m in re.finditer(r"\\?boxed\s*\{", text):
        i, depth = m.end(), 1
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0); i += 1
        if depth == 0: out = text[m.end():i - 1]
    if out is None: return None
    s = re.sub(r"\\text\{[^}]*\}|\\(?:mathrm|textbf)\{([^}]*)\}", lambda g: g.group(1) or "", out)
    s = s.replace("\\$", "").replace("$", "").replace("\\,", "").replace("\\!", "")
    nums = re.findall(NUM, s)
    return float(nums[0].replace(",", "")) if len(nums) == 1 else None

def vote(vals, y):
    v = [x for x in vals if x is not None]
    if not v: return 0.0
    c = Counter(v); top = max(c.values()); w = [k for k, n in c.items() if n == top]
    return (y in w) / len(w)

def fnum(a):
    try: return float(a)
    except Exception: return None

def load(cfg_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(cfg_dir, "seed*", "gsm8k_result.json"))):
        seed = int(re.search(r"seed(\d+)", p).group(1))
        for d in json.load(open(p)):
            gt = d["correct_answer"]; y = float(solve_math_problems(gt))
            rounds = [list(r["responses"].values()) for r in d["debate_session"]["rounds"]]
            rows.append({"seed": seed, "gt": gt, "y": y, "rounds": rounds})
    return rows

def boot_strat(d, strata):
    groups = {}
    for x, s in zip(d, strata): groups.setdefault(s, []).append(x)
    rng = random.Random(SEED); n = len(d); est = []
    for _ in range(B):
        tot = 0
        for g in groups.values():
            tot += sum(g[rng.randrange(len(g))] for _ in range(len(g)))
        est.append(tot / n)
    est.sort()
    return 100 * sum(d) / n, 100 * est[int(.025 * B)], 100 * est[int(.975 * B) - 1]

def analyze(rows):
    R = len(rows[0]["rounds"]); res = {"n": len(rows), "seeds": sorted({r["seed"] for r in rows})}
    for sc in ("authors", "validated"):
        votes = []
        for r in range(R):
            if sc == "authors":
                votes.append([float(compute_accuracy(t["gt"], t["rounds"][r], "gsm8k") or 0) for t in rows])
            else:
                votes.append([vote([boxed_last(x) for x in t["rounds"][r]], t["y"]) for t in rows])
        res[sc] = {"vote_by_round": [sum(v) / len(rows) for v in votes],
                   "after_minus_before": boot_strat([a - b for a, b in zip(votes[-1], votes[0])], [t["seed"] for t in rows])}
        # agent-level correctness and flips
        def corr(x, y):
            if sc == "authors":
                return compute_accuracy("#### " + (str(int(y)) if y == int(y) else str(y)), x, "gsm8k") == 1.0
            return boxed_last(x) == y
        ind, flips = [], []
        for r in range(R):
            ind.append(sum(corr(x, t["y"]) for t in rows for x in t["rounds"][r]) / sum(len(t["rounds"][r]) for t in rows))
        for r in range(1, R):
            rw = wr = c_prev = w_prev = 0
            for t in rows:
                for a, b in zip(t["rounds"][r - 1], t["rounds"][r]):
                    ca, cb = corr(a, t["y"]), corr(b, t["y"])
                    c_prev += ca; w_prev += (not ca); rw += ca and not cb; wr += (not ca) and cb
            flips.append({"round": r, "right_to_wrong_pct_of_right": 100 * rw / max(c_prev, 1), "wrong_to_right_pct_of_wrong": 100 * wr / max(w_prev, 1)})
        res[sc]["individual_by_round"] = ind; res[sc]["flips"] = flips
    # diagnostics: authors' parse failures, string-equality failures on numerically equal answers
    diag = []
    for r in range(R):
        fail = streq = n = 0
        for t in rows:
            g = solve_math_problems(t["gt"])
            for x in t["rounds"][r]:
                n += 1; a = parse_gsm8k_answer(x)
                if a is None: fail += 1
                elif a != g and fnum(a) is not None and fnum(a) == fnum(g): streq += 1
        diag.append({"round": r, "authors_parse_fail_pct": 100 * fail / n, "string_eq_false_negatives_pct": 100 * streq / n})
    res["diagnostics"] = diag
    return res

if __name__ == "__main__":
    out_dir = sys.argv[1]; cfgs = sys.argv[2:] or sorted(os.listdir(out_dir))
    allres = {}
    for c in cfgs:
        rows = load(os.path.join(out_dir, c))
        if not rows: continue
        r = analyze(rows); allres[c] = r
        print(f"== {c} n={r['n']} seeds={r['seeds']}")
        for sc in ("authors", "validated"):
            a = r[sc]["after_minus_before"]
            print(f"  {sc:9s} vote " + " ".join(f"{100*v:.1f}" for v in r[sc]["vote_by_round"]) + f"  after-before {a[0]:+.1f} [{a[1]:+.1f}, {a[2]:+.1f}]"
                  + "  ind " + " ".join(f"{100*v:.1f}" for v in r[sc]["individual_by_round"])
                  + "  flips R->W/W->R " + " ".join(f"{f['right_to_wrong_pct_of_right']:.1f}/{f['wrong_to_right_pct_of_wrong']:.1f}" for f in r[sc]["flips"]))
        print("  diag " + " ".join(f"r{d['round']}: fail {d['authors_parse_fail_pct']:.1f}% streq {d['string_eq_false_negatives_pct']:.1f}%" for d in r["diagnostics"]))
    json.dump(allres, open(os.path.join(out_dir, "..", "analysis.json"), "w"), indent=1)
