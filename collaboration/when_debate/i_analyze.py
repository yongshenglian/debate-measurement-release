"""Pre-registered analysis of study I (reports/debate-indep-20261121-a/PROTOCOL.md).
  fill-prepare <I out> <jobs.jsonl>                  : blind Gemma jobs for no-answer outputs of the I debate arms
  debate <I out> <H out> <I fill out> <H fill out>   : independence-sentence debate arms vs study H arms and H_vote20
  peer <I out> <H out> <labels.json>                 : review with sentences (I) vs neutral review (H), paired by task
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hard_analyze import load, gemma_val, SYSTEM
from settings_analyze import vote, top, boot, boot_p, holm

f = lambda x: f"{100*x[0]:+.1f} [{100*x[1]:+.1f}, {100*x[2]:+.1f}]"


def G(path):
    return {json.loads(l)["id"]: gemma_val(json.loads(l)["text"]) for l in open(path)}


def debate(i_out, h_out, i_fill, h_fill):
    A = load(i_out, G(i_fill)); B = load(h_out, G(h_fill))
    _, V = B["H_vote20"]; n = len(V); v20 = [vote(r[0], y) for y, r in V]
    res, P1, ps = {}, {}, {}
    for arm in ("H_full", "H_ans", "H_full_indep", "H_ans_indep"):
        _, T = (A if arm.endswith("indep") else B)[arm]; K = len(T[0][1]) - 1
        v = [[vote(r[k], y) for y, r in T] for k in range(K + 1)]
        ind = [sum(a == y for y, r in T for a in r[k]) / (5 * n) for k in range(K + 1)]
        g4 = [v[0][j] if top(T[j][1][0]) >= 4 else v[K][j] for j in range(n)]
        pool = [vote(sum(r, []), y) for y, r in T]
        unan = sum(top(r[K]) == 5 for _, r in T) / n
        P1[arm] = [a - b for a, b in zip(v[K], v[0])]
        st = {}
        for lab, fn in (("<=3/5", lambda k: k <= 3),):
            idx = [j for j, (y, r) in enumerate(T) if fn(top(r[0]))]
            st[lab] = {"n": len(idx), "rescued": sum(max(0, v[K][j] - v[0][j]) for j in idx), "broken": sum(max(0, v[0][j] - v[K][j]) for j in idx)}
        r = {"vote_by_round": [sum(x) / n for x in v], "individual_by_round": ind, "P1": boot(P1[arm]),
             "P2_G4": boot([a - b for a, b in zip(g4, v[0])]), "final-vote20": boot([a - b for a, b in zip(v[K], v20)]),
             "pool20-vote20": boot([a - b for a, b in zip(pool, v20)]), "final_unanimous": unan, "contested": st}
        res[arm] = r
        print(f"== {arm}: vote " + " ".join(f"{100*x:.1f}" for x in r["vote_by_round"]) + " | individual " +
              " ".join(f"{100*x:.1f}" for x in ind) + f" | final unanimous {100*unan:.1f}%")
        print(f"   P1 {f(r['P1'])} | G4 {f(r['P2_G4'])} | final - vote20 {f(r['final-vote20'])} | pool of 20 - vote20 {f(r['pool20-vote20'])}"
              f" | contested n={st['<=3/5']['n']} +{st['<=3/5']['rescued']:.1f}/-{st['<=3/5']['broken']:.1f}")
    print(f"\nvote20 {100*sum(v20)/n:.1f}")
    print(f"PRIMARY IS1  H_ans_indep final - vote20: {f(res['H_ans_indep']['final-vote20'])}")
    est = {}
    for a in ("H_full", "H_ans"):
        dd = [x - y for x, y in zip(P1[a + "_indep"], P1[a])]; est[a] = boot(dd); ps[a] = boot_p(dd)
    adj = holm(ps)
    for a in est:
        print(f"IS2  P1({a}_indep) - P1({a}): {f(est[a])}  p={ps[a]:.4f}  Holm={adj[a]:.4f}")
    res["IS2"] = {a: {"est": est[a], "p": ps[a], "holm": adj[a]} for a in est}
    json.dump(res, open(os.path.join(i_out, "..", "analysis_debate.json"), "w"), indent=1)


def peer(i_out, h_out, lab):
    Y = {int(k): v for k, v in json.load(open(lab)).items()}
    ans = lambda r: None if r.get("parsed") is None else r["parsed"]["answer"]
    rv = {}
    for tag, d in (("I", i_out), ("H", h_out)):
        for m in ("qwen", "gemma"):
            for l in open(os.path.join(d, f"{m}_review.jsonl")):
                r = json.loads(l); rv[(tag, m, r["tid"], r["cond"])] = ans(r)
    res, ps = {}, {}
    for m in ("gemma", "qwen"):
        for c in ("self", "peer_qwen", "peer_gemma"):
            T = [t for t in Y if ("I", m, t, c) in rv and ("H", m, t, c) in rv]
            a = [float(rv[("I", m, t, c)] == Y[t]) for t in T]; b = [float(rv[("H", m, t, c)] == Y[t]) for t in T]
            d = [x - y for x, y in zip(a, b)]; e = boot(d); key = f"{m}:{c}"
            res[key] = {"n": len(T), "indep": sum(a) / len(T), "neutral": sum(b) / len(T), "indep-neutral": e}
            if key in ("gemma:peer_qwen", "qwen:peer_qwen", "qwen:peer_gemma"): ps[key] = boot_p(d)
            print(f"{m} {c:10s} n={len(T)}: neutral {100*sum(b)/len(T):.1f}  indep {100*sum(a)/len(T):.1f}  indep - neutral {f(e)}")
    adj = holm(ps)
    for k in ps:
        print(f"PRIMARY IR  {k}: {f(res[k]['indep-neutral'])}  p={ps[k]:.4f}  Holm={adj[k]:.4f}"); res[k]["p"] = ps[k]; res[k]["holm"] = adj[k]
    json.dump(res, open(os.path.join(i_out, "..", "analysis_peer.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "fill-prepare":
        with open(sys.argv[3], "x") as fo:
            for arm, (D, T) in load(sys.argv[2]).items():
                for qi, ex in enumerate(D["results"]):
                    for k, rnd in enumerate(ex["rounds"]):
                        for a, t in enumerate(rnd):
                            if T[qi][1][k][a] is None:
                                fo.write(json.dumps({"id": f"{arm}|{k}|{qi}|{a}", "seed": 0, "messages": [
                                    {"role": "system", "content": SYSTEM}, {"role": "user", "content": "Response:\n" + t}]}) + "\n")
    elif sys.argv[1] == "debate":
        debate(*sys.argv[2:6])
    else:
        peer(*sys.argv[2:5])
