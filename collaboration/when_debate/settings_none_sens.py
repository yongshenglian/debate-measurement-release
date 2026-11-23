"""POST-HOC sensitivity for study S (not pre-registered): every output our reader scores as no-answer is read blind by Gemma;
P1/P2 are recomputed with Gemma's reading filling those gaps.
  prepare <out dir> <jobs.jsonl> | score <out dir> <gemma_out.jsonl>"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from settings_analyze import reader, r1, vote, top, boot
sys.path.insert(0, "/home/lian/swarmworld/research/reports/format-contagion-20261119-a")
from fc_audit_prepare import SYSTEM
import re
def gemma_val(text):   # copy of settings_audit.gemma_val (that module runs on import)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        a = json.loads(m.group(0))["answer"]
        if isinstance(a, str): a = float(a.replace(",", "").replace("$", ""))
        return None if a is None else r1(float(a))
    except Exception:
        return None

def cells(d):
    for p in sorted(glob.glob(os.path.join(d, "S*.json"))):
        D = json.load(open(p)); rd = reader(D["cfg"]["family"])
        yield D, rd

if sys.argv[1] == "prepare":
    with open(sys.argv[3], "x") as f:
        for D, rd in cells(sys.argv[2]):
            for qi, ex in enumerate(D["results"]):
                for k, rnd in enumerate(ex["rounds"]):
                    for a, t in enumerate(rnd):
                        if rd(t) is None:
                            f.write(json.dumps({"id": f"{D['arm']}|{k}|{qi}|{a}", "seed": 0, "messages": [
                                {"role": "system", "content": SYSTEM}, {"role": "user", "content": "Response:\n" + t}]}) + "\n")
else:
    G = {json.loads(l)["id"]: gemma_val(json.loads(l)["text"]) for l in open(sys.argv[3])}
    f = lambda x: f"{100*x[0]:+.1f} [{100*x[1]:+.1f}, {100*x[2]:+.1f}]"
    P = {}
    for D, rd in cells(sys.argv[2]):
        T = []
        for qi, ex in enumerate(D["results"]):
            y = r1(float(str(ex["label"]).replace(",", "")))
            T.append((y, [[rd(t) if rd(t) is not None else G.get(f"{D['arm']}|{k}|{qi}|{a}") for a, t in enumerate(rnd)]
                          for k, rnd in enumerate(ex["rounds"])]))
        K = len(T[0][1]) - 1
        v = [[vote(r[k], y) for y, r in T] for k in range(K + 1)]
        g4 = [v[0][j] if top(T[j][1][0]) >= 4 else v[K][j] for j in range(len(T))]
        P[D["arm"]] = [a - b for a, b in zip(v[K], v[0])]
        none = sum(a is None for _, r in T for a in r[K]) / (5 * len(T))
        print(f"{D['arm']:18s} vote r0 {100*sum(v[0])/len(T):.1f} final {100*sum(v[K])/len(T):.1f} | P1 {f(boot([a-b for a,b in zip(v[K],v[0])]))}"
              f" | P2 {f(boot([a-b for a,b in zip(g4,v[0])]))} | final no-answer {100*none:.1f}%")
    from settings_analyze import CONTRASTS, boot_p
    print("contrasts on P1 (Gemma fills no-answers):")
    for name, a, b in CONTRASTS:
        dd = [x - y for x, y in zip(P[a], P[b])]
        print(f"   {name} {a} - {b}: {f(boot(dd))}  p={boot_p(dd):.4f}")
