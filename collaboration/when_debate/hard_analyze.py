"""Pre-registered analysis of study H, settings part (reports/debate-hard-20261120-a/PROTOCOL.md).
  fill-prepare <out dir> <jobs.jsonl>       : blind Gemma jobs for every output our reader scores as no-answer
  analyze <out dir> [gemma_fill_out.jsonl]  : tables; with the Gemma file -> primary (Gemma-filled) reading, else ours only
  hr3 <peer out dir> <labels.json>          : HR3 predicted net for Gemma reading Qwen (study R pooled GSM8K rates)
Writes <out dir>/../analysis[_filled].json"""
import glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from settings_analyze import reader, r1, vote, top, boot, boot_p, holm
sys.path.insert(0, "/home/lian/swarmworld/research/reports/format-contagion-20261119-a")
from fc_audit_prepare import SYSTEM

DEBATE = ["H_full", "H_ans", "H_dovword"]


def gemma_val(text):   # as settings_audit.gemma_val
    m = re.search(r"\{.*\}", text, re.S)
    try:
        a = json.loads(m.group(0))["answer"]
        if isinstance(a, str): a = float(a.replace(",", "").replace("$", ""))
        return None if a is None else r1(float(a))
    except Exception:
        return None


def load(d, G=None):
    A = {}
    for p in sorted(glob.glob(os.path.join(d, "H_*.json"))):
        D = json.load(open(p)); rd = reader(D["cfg"]["family"]); arm = D["arm"]; T = []
        for qi, ex in enumerate(D["results"]):
            def val(k, a, t):
                v = rd(t)
                return v if v is not None or G is None else G.get(f"{arm}|{k}|{qi}|{a}")
            T.append((r1(float(ex["label"])), [[val(k, a, t) for a, t in enumerate(rnd)] for k, rnd in enumerate(ex["rounds"])]))
        A[arm] = (D, T)
    return A


def main():
    if sys.argv[1] == "fill-prepare":
        with open(sys.argv[3], "x") as f:
            for arm, (D, T) in load(sys.argv[2]).items():
                for qi, ex in enumerate(D["results"]):
                    for k, rnd in enumerate(ex["rounds"]):
                        for a, t in enumerate(rnd):
                            if T[qi][1][k][a] is None:
                                f.write(json.dumps({"id": f"{arm}|{k}|{qi}|{a}", "seed": 0, "messages": [
                                    {"role": "system", "content": SYSTEM}, {"role": "user", "content": "Response:\n" + t}]}) + "\n")
        return
    if sys.argv[1] == "hr3":
        Y = {int(k): v for k, v in json.load(open(sys.argv[3])).items()}
        sol = {}
        for m in ("qwen", "gemma"):
            for l in open(os.path.join(sys.argv[2], f"{m}_solve.jsonl")):
                r = json.loads(l); sol[(m, r["tid"], r["sample"])] = None if r["parsed"] is None else r["parsed"]["answer"]
        rw = sum(sol[("gemma", t, 0)] == Y[t] and sol[("qwen", t, 0)] is not None and sol[("qwen", t, 0)] != Y[t] for t in Y)
        wr = sum(sol[("gemma", t, 0)] != Y[t] and sol[("qwen", t, 0)] == Y[t] for t in Y)
        print(f"HR3: N={len(Y)}, gemma right & qwen wrong {rw}, gemma wrong & qwen right {wr}; "
              f"predicted net {100*(-0.288*rw + 0.882*wr)/len(Y):+.2f} pp")
        return
    d = sys.argv[2]
    G = {json.loads(l)["id"]: gemma_val(json.loads(l)["text"]) for l in open(sys.argv[3])} if len(sys.argv) > 3 else None
    A = load(d, G)
    f = lambda x: f"{100*x[0]:+.1f} [{100*x[1]:+.1f}, {100*x[2]:+.1f}]"
    print("reading:", "ours + blind Gemma for no-answer outputs (PRIMARY)" if G else "ours only")
    res = {}
    _, V = A["H_vote20"]; n = len(V)
    v20 = [vote(r[0], y) for y, r in V]; v5 = [vote(r[0][:5], y) for y, r in V]
    ind20 = sum(a == y for y, r in V for a in r[0]) / (20 * n)
    print(f"== H_vote20: vote of 20 {100*sum(v20)/n:.1f} | vote of first 5 {100*sum(v5)/n:.1f} | individual {100*ind20:.1f}"
          f" | 20 - 5: {f(boot([a-b for a,b in zip(v20,v5)]))}")
    res["H_vote20"] = {"vote20": sum(v20) / n, "vote5": sum(v5) / n, "individual": ind20, "20-5": boot([a - b for a, b in zip(v20, v5)])}
    P1 = {}
    for arm in DEBATE:
        if arm not in A: continue
        _, T = A[arm]; K = len(T[0][1]) - 1
        v = [[vote(r[k], y) for y, r in T] for k in range(K + 1)]
        ind = [sum(a == y for y, r in T for a in r[k]) / (5 * n) for k in range(K + 1)]
        none = [sum(a is None for _, r in T for a in r[k]) / (5 * n) for k in range(K + 1)]
        g4 = [v[0][j] if top(T[j][1][0]) >= 4 else v[K][j] for j in range(n)]
        P1[arm] = [a - b for a, b in zip(v[K], v[0])]
        st = {}
        for lab, fn in (("5/5", lambda k: k == 5), ("4/5", lambda k: k == 4), ("<=3/5", lambda k: k <= 3)):
            idx = [j for j, (y, r) in enumerate(T) if fn(top(r[0]))]
            st[lab] = {"n": len(idx), "rescued": sum(max(0, v[K][j] - v[0][j]) for j in idx), "broken": sum(max(0, v[0][j] - v[K][j]) for j in idx)}
        r = {"vote_by_round": [sum(x) / n for x in v], "individual_by_round": ind, "no_answer_by_round": none,
             "P1": boot(P1[arm]), "P2_G4": boot([a - b for a, b in zip(g4, v[0])]),
             "final-vote20": boot([a - b for a, b in zip(v[K], v20)]), "G4-vote20": boot([a - b for a, b in zip(g4, v20)]), "strata": st}
        res[arm] = r
        print(f"== {arm}: vote by round " + " ".join(f"{100*x:.1f}" for x in r["vote_by_round"]) +
              " | individual " + " ".join(f"{100*x:.1f}" for x in ind) + " | no answer % " + " ".join(f"{100*x:.1f}" for x in none))
        print(f"   P1 final - r0 vote {f(r['P1'])} | P2 G4 gate - r0 {f(r['P2_G4'])} | final - vote20 {f(r['final-vote20'])} | G4 - vote20 {f(r['G4-vote20'])}")
        print("   rescued/broken by r0 agreement: " + "; ".join(f"{k} (n={s['n']}): +{s['rescued']:.1f} / -{s['broken']:.1f}" for k, s in st.items()))
    ps, est = {}, {}
    for a in ("H_ans", "H_dovword"):
        if a in P1 and "H_full" in P1:
            dd = [x - y for x, y in zip(P1[a], P1["H_full"])]; est[a] = boot(dd); ps[a] = boot_p(dd)
    adj = holm(ps) if ps else {}
    for a in est:
        print(f"   contrast {a} - H_full on P1: {f(est[a])}  p={ps[a]:.4f}  Holm={adj[a]:.4f}")
    res["contrasts"] = {a: {"est": est[a], "p": ps[a], "holm": adj[a]} for a in est}
    json.dump(res, open(os.path.join(d, "..", "analysis_filled.json" if G else "analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
