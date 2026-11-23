"""Study M analysis (reports/mmlupro-20261121-a/PROTOCOL.md): the H/I debate analyses with a multiple-choice reader.
  fill-prepare <out dir> <jobs.jsonl>         blind Gemma (MCQ system prompt) for outputs the MCQ reader scores as no-answer
  audit-prepare <out dir> <jobs.jsonl>        100 random outputs per arm (50 round 0, 50 final), seed 20261122
  audit-score <out dir> <audit jobs> <gemma out>
  debate <out dir> <fill out>                 H_full, H_ans, H_full_indep, H_ans_indep vs H_vote20
"""
import glob, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hard_analyze as ha
from settings_analyze import vote, top, boot, boot_p, holm

MCQ_SYSTEM = ("You read a response to a multiple-choice question whose options are numbered. Report the number of the "
              "option the response gives as its final answer. If it gives no single final option, report null. "
              'Reply with JSON only: {"answer": <option number or null>}.')


def mcq(t):
    """Option number from the last <answer> segment (closed or not); None if absent or ambiguous."""
    i = t.rfind("<answer>")
    if i < 0: return None
    seg = t[i + 8:].split("</answer>")[0].strip()
    m = re.match(r"^(?:\*\*)?(?:option\s*)?\(?\s*(10|[1-9])\s*\)?(?![\d.,]\d)", seg, re.I)
    if not m:
        ms = set(re.findall(r"\boption\s*\(?(10|[1-9])\b", seg, re.I))
        m = None if len(ms) != 1 else type("M", (), {"group": lambda self, k, v=ms.pop(): v})()
    return None if m is None else float(int(m.group(1)))


ha.reader = lambda family: mcq   # hard_analyze.load reads answers through this name


def gval(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        a = json.loads(m.group(0))["answer"]
        return None if a is None else float(int(a))
    except Exception:
        return None


def jobs_line(i, t):
    return json.dumps({"id": i, "seed": 0, "messages": [{"role": "system", "content": MCQ_SYSTEM}, {"role": "user", "content": "Response:\n" + t}]}) + "\n"


def audit_items(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "H_*.json"))):
        D = json.load(open(p))
        if D["arm"] == "H_vote20": continue
        K = len(D["results"][0]["rounds"]) - 1; rng = random.Random(f"20261122|{D['arm']}")
        for rnd in (0, K):
            cells = [(qi, a) for qi in range(len(D["results"])) for a in range(5)]
            for qi, a in rng.sample(cells, 50):
                out.append((f"{D['arm']}|{rnd}|{qi}|{a}", D["results"][qi]["rounds"][rnd][a]))
    return out


def debate(d, fill):
    G = {json.loads(l)["id"]: gval(json.loads(l)["text"]) for l in open(fill)}
    A = ha.load(d, G); _, V = A["H_vote20"]; n = len(V)
    v20 = [vote(r[0], y) for y, r in V]; v5 = [vote(r[0][:5], y) for y, r in V]
    f = lambda x: f"{100*x[0]:+.1f} [{100*x[1]:+.1f}, {100*x[2]:+.1f}]"
    print(f"H_vote20: vote of 20 {100*sum(v20)/n:.1f} | first 5 {100*sum(v5)/n:.1f} | 20 - 5 {f(boot([a-b for a,b in zip(v20,v5)]))}")
    res, P1 = {"vote20": sum(v20) / n, "vote5": sum(v5) / n}, {}
    for arm in ("H_full", "H_ans", "H_full_indep", "H_ans_indep"):
        if arm not in A: continue
        _, T = A[arm]; K = len(T[0][1]) - 1
        v = [[vote(r[k], y) for y, r in T] for k in range(K + 1)]
        ind = [sum(a == y for y, r in T for a in r[k]) / (5 * n) for k in range(K + 1)]
        none = [sum(a is None for _, r in T for a in r[k]) / (5 * n) for k in range(K + 1)]
        g4 = [v[0][j] if top(T[j][1][0]) >= 4 else v[K][j] for j in range(n)]
        pool = [vote(sum(r, []), y) for y, r in T]
        P1[arm] = [a - b for a, b in zip(v[K], v[0])]
        r = {"vote_by_round": [sum(x) / n for x in v], "individual_by_round": ind, "no_answer_by_round": none, "P1": boot(P1[arm]),
             "P2_G4": boot([a - b for a, b in zip(g4, v[0])]), "final-vote20": boot([a - b for a, b in zip(v[K], v20)]),
             "pool20-vote20": boot([a - b for a, b in zip(pool, v20)]), "final_unanimous": sum(top(r[K]) == 5 for _, r in T) / n}
        res[arm] = r
        print(f"== {arm}: vote " + " ".join(f"{100*x:.1f}" for x in r["vote_by_round"]) + " | individual " + " ".join(f"{100*x:.1f}" for x in ind)
              + " | no answer % " + " ".join(f"{100*x:.1f}" for x in none) + f" | final unanimous {100*r['final_unanimous']:.1f}%")
        print(f"   P1 {f(r['P1'])} | G4 {f(r['P2_G4'])} | final - vote20 {f(r['final-vote20'])} | pool of 20 - vote20 {f(r['pool20-vote20'])}")
    ps, est = {}, {}
    for a, b in (("H_ans", "H_full"), ("H_full_indep", "H_full"), ("H_ans_indep", "H_ans")):
        if a in P1 and b in P1:
            dd = [x - y for x, y in zip(P1[a], P1[b])]; est[f"{a} - {b}"] = boot(dd); ps[f"{a} - {b}"] = boot_p(dd)
    adj = holm(ps)
    for k in est: print(f"   P1 contrast {k}: {f(est[k])}  p={ps[k]:.4f}  Holm={adj[k]:.4f}")
    res["contrasts"] = {k: {"est": est[k], "p": ps[k], "holm": adj[k]} for k in est}
    json.dump(res, open(os.path.join(d, "..", "analysis_debate.json"), "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "fill-prepare":
        with open(a[3], "x") as fo:
            for arm, (D, T) in ha.load(a[2]).items():
                for qi, ex in enumerate(D["results"]):
                    for k, rnd in enumerate(ex["rounds"]):
                        for ag, t in enumerate(rnd):
                            if T[qi][1][k][ag] is None: fo.write(jobs_line(f"{arm}|{k}|{qi}|{ag}", t))
    elif a[1] == "audit-prepare":
        with open(a[3], "x") as fo:
            for i, t in audit_items(a[2]): fo.write(jobs_line(i, t))
    elif a[1] == "audit-score":
        its = dict(audit_items(a[2])); G = {json.loads(l)["id"]: gval(json.loads(l)["text"]) for l in open(a[4])}
        for arm in sorted({i.split("|")[0] for i in its}):
            ids = [i for i in its if i.startswith(arm + "|")]; rec = [i for i in ids if mcq(its[i]) is not None]
            ag = sum(mcq(its[i]) == G[i] for i in rec)
            print(f"{arm:14s} agreement on recovered {ag}/{len(rec)} = {100*ag/max(1,len(rec)):.1f}%  {'PASS' if rec and ag/len(rec) >= .95 else 'FAIL'}"
                  f" | ours none, Gemma reads one: {sum(mcq(its[i]) is None and G[i] is not None for i in ids)}/{len(ids)}")
    else:
        debate(a[2], a[3])
