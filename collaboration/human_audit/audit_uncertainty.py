"""Post hoc (after the second external review, 2026-11-23): uncertainty in the audit-informed sensitivity analysis.

score.py's "reference-corrected" reading uses the reference label for each audited output; for an unaudited output where OURS
reads nothing (N) or OURS and G differ (D) it takes the reader that matched the reference more often among audited outputs of the
same cell and stratum; agreeing outputs (A) keep the agreed reading. Its bootstrap treats that reader choice as fixed.

This script reports, for every debate contrast in score.py:
  (1) the point estimate as in score.py;
  (2) a joint bootstrap (2,000 draws): resample the audited items within each cell x stratum (N, D) with replacement, re-make the
      reader choice, then resample problems; percentile 95% interval;
  (3) extreme bounds: every unaudited N/D output in the arms of a contrast read as the gold answer on the side that raises the
      contrast and as one common wrong answer on the other side (upper), and the reverse (lower). Point values, no sampling.
Reference labels: the agreed labels of the two reference LLM readers (PROTOCOL.md amendment 2; primary analysis).
Usage: audit_uncertainty.py <audit dir>"""
import csv, json, os, random, sys
from collections import Counter, defaultdict
sys.path.insert(0, "/home/lian/swarmworld/research/collaboration/when_debate")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = sys.argv; sys.argv = [sys.argv[0], "/nonexistent"]
from score import norm, SRC, R
sys.argv = _argv
from hard_analyze import gemma_val
from settings_analyze import reader, r1, vote

A_DIR = sys.argv[1]; NB = 2000; SEED = 20261123; WRONG = -987654321.0
key = json.load(open(os.path.join(A_DIR, "key.json")))
lab = lambda p: {r["id"]: norm(r["label"]) for r in csv.DictReader(open(p)) if r["label"].strip()}
LA, LB = lab(os.path.join(A_DIR, "labels_claude.csv")), lab(os.path.join(A_DIR, "to-labellers/labels_AI_packet1.csv"))
H = {i: LA[i] for i in key if i in LA and i in LB and LA[i] == LB[i]}
G = {}
for l in open(os.path.join(A_DIR, "stageA", "out.jsonl")):
    d = json.loads(l); G[d["id"]] = gemma_val(d["text"])
audited = defaultdict(list)          # (task, arm, rg, stratum) -> [(ours==h, gemma==h)]
lab_by_src = {}
for i, k in key.items():
    if k["part"] == "A" and i in H:
        audited[(k["task"], k["arm"], k["round"], k["stratum"])].append((k["readings"]["ours"] == H[i], k["readings"]["gemma"] == H[i]))
        lab_by_src[k["src"]] = H[i]


def choose(pairs):
    o = sum(a for a, _ in pairs); g = sum(b for _, b in pairs)
    return "g" if g > o else "o" if o > g else "p"


# per problem: label, and for rounds 0 and K the list of outputs (fixed value, or (cellstratum, ours, g) if unaudited N/D)
P = {}
for sub, (task, arms) in SRC.items():
    study = sub.split("/")[0]
    for arm in arms:
        Dd = json.load(open(f"{R}/{sub}/{arm}.json")); rd = reader(Dd["cfg"]["family"]); rows = []
        for qi, ex in enumerate(Dd["results"]):
            K = len(ex["rounds"]) - 1; rr = {}
            for k in sorted({0, K}):
                rg = "r0" if k == 0 else "final"; outs = []
                for a, t in enumerate(ex["rounds"][k]):
                    iid = f"{study}|{arm}|{k}|{qi}|{a}"; o = rd(t); g = G[iid]
                    s = "N" if o is None else ("A" if o == g else "D")
                    if iid in lab_by_src: outs.append(("fix", lab_by_src[iid]))
                    elif s == "A": outs.append(("fix", o))
                    else: outs.append(("var", (task, arm, rg, s), o, g))
                rr["r0" if k == 0 else "final"] = outs
            rows.append((r1(float(ex["label"])), rr))
        P[(task, arm)] = rows


def val(out, ch, y=None, mode=None):
    if out[0] == "fix": return out[1]
    if mode == "hi": return y
    if mode == "lo": return WRONG
    c = ch[out[1]]; o, g = out[2], out[3]
    return g if c == "g" else o if c == "o" else (o if o is not None else g)


def votes(task, arm, rg, ch, mode=None):
    return [vote([val(x, ch, y, mode) for x in rr[rg]], y) for y, rr in P[(task, arm)]]


def contrasts(ch, mode=None):
    """mode None: per-problem difference lists. mode 'hi'/'lo': extreme point values per contrast."""
    out = {}
    for task in ("math", "mmlu"):
        arms = [a for (t, a) in P if t == task and a != "H_vote20"]
        for arm in arms:
            if mode is None:
                fin, r0, v20 = votes(task, arm, "final", ch), votes(task, arm, "r0", ch), votes(task, "H_vote20", "r0", ch)
            else:
                up, dn = ("hi", "lo") if mode == "hi" else ("lo", "hi")
                fin, r0, v20 = votes(task, arm, "final", ch, up), votes(task, arm, "r0", ch, dn), votes(task, "H_vote20", "r0", ch, dn)
            out[f"{task} {arm} final - vote20"] = [a - b for a, b in zip(fin, v20)]
            out[f"{task} {arm} final - r0 vote"] = [a - b for a, b in zip(fin, r0)]
        for a, b in (("H_ans", "H_full"), ("H_ans_indep", "H_full_indep")):
            if (task, a) in P and (task, b) in P:
                if mode is None:
                    da = [x - y for x, y in zip(votes(task, a, "final", ch), votes(task, a, "r0", ch))]
                    db = [x - y for x, y in zip(votes(task, b, "final", ch), votes(task, b, "r0", ch))]
                else:
                    up, dn = ("hi", "lo") if mode == "hi" else ("lo", "hi")
                    da = [x - y for x, y in zip(votes(task, a, "final", ch, up), votes(task, a, "r0", ch, dn))]
                    db = [x - y for x, y in zip(votes(task, b, "final", ch, dn), votes(task, b, "r0", ch, up))]
                out[f"{task} contrast {a} - {b} on (final - r0)"] = [x - y for x, y in zip(da, db)]
    return out


cells = sorted({x[1] for rows in P.values() for _, rr in rows for rg in rr for x in rr[rg] if x[0] == "var"} | set(audited))
ch0 = {c: choose(audited.get(c, [])) for c in cells}
point = {k: sum(v) / len(v) for k, v in contrasts(ch0).items()}
hi = {k: sum(v) / len(v) for k, v in contrasts(ch0, "hi").items()}
lo = {k: sum(v) / len(v) for k, v in contrasts(ch0, "lo").items()}
n_var = Counter(x[1][:2] for rows in P.values() for _, rr in rows for rg in rr for x in rr[rg] if x[0] == "var")

rng = random.Random(SEED); draws = defaultdict(list)
cache = {}
for b in range(NB):
    ch = {c: choose([rng.choice(audited[c]) for _ in audited[c]]) if audited.get(c) else ch0[c] for c in cells}
    sig = tuple(sorted(ch.items()))
    if sig not in cache: cache[sig] = contrasts(ch)
    per = cache[sig]
    for k, v in per.items():
        n = len(v); draws[k].append(sum(v[rng.randrange(n)] for _ in range(n)) / n)

print(f"reference labels (agreed): {len(H)}; unaudited N/D outputs by (task, arm): " + ", ".join(f"{t} {a} {n}" for (t, a), n in sorted(n_var.items())))
print(f"distinct reader-choice configurations across {NB} draws: {len(cache)}")
print(f"\n{'contrast':58s} {'point':>7s}  {'joint 95% interval':>20s}  {'extreme bounds':>18s}")
for k in point:
    s = sorted(draws[k]); lo95, hi95 = s[int(.025 * NB)], s[int(.975 * NB) - 1]
    print(f"{k:58s} {100*point[k]:+7.1f}  [{100*lo95:+6.1f}, {100*hi95:+6.1f}]      [{100*lo[k]:+6.1f}, {100*hi[k]:+6.1f}]")
