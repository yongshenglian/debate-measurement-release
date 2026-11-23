"""POST-HOC (not pre-registered) summary of study R: rescue/follow rates pooled over peers, Wilson 95% CIs,
and an accounting check: net change ~= (#own wrong & peer right) * rescue - (#own right & peer wrong) * follow."""
import json, math, os, sys
d, lab = sys.argv[1:3]
Y = {int(k): v for k, v in json.load(open(lab)).items()}
ans = lambda r: None if r.get("parsed") is None else r["parsed"]["answer"]
sol, rev = {}, {}
for m in ("qwen", "gemma"):
    for l in open(os.path.join(d, f"{m}_solve.jsonl")):
        r = json.loads(l); sol[(m, r["tid"], r["sample"])] = ans(r)
    for l in open(os.path.join(d, f"{m}_review.jsonl")):
        r = json.loads(l); rev[(m, r["tid"], r["cond"])] = ans(r)
def wilson(k, n, z=1.96):
    if n == 0: return "n/a"
    p = k / n; c = (p + z*z/(2*n)) / (1 + z*z/n); h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return f"{100*p:.1f}% [{100*(c-h):.1f}, {100*(c+h):.1f}] ({k}/{n})"
for m in ("gemma", "qwen"):
    pool = {"accept correct peer (own wrong, peer right)": [0, 0], "follow wrong peer (own right, peer wrong)": [0, 0],
            "abandon right answer (own right, peer wrong)": [0, 0]}
    for pm in ("qwen", "gemma"):
        c = f"peer_{pm}"; s = 1 if pm == m else 0
        n_wr = n_rw = res = fol = ab = 0
        for t in Y:
            if (m, t, c) not in rev: continue   # review not generated (study H: 1-4 tasks per reviewer)
            own, peer, fin = sol[(m, t, 0)], sol[(pm, t, s)], rev[(m, t, c)]
            if own != Y[t] and peer == Y[t]: n_wr += 1; res += fin == Y[t]
            if own == Y[t] and peer is not None and peer != Y[t]: n_rw += 1; fol += fin == peer; ab += fin != Y[t]
        net = sum((rev[(m, t, c)] == Y[t]) - (sol[(m, t, 0)] == Y[t]) for t in Y if (m, t, c) in rev)
        print(f"{m} reads {pm}: own-wrong&peer-right {n_wr}, rescued {res} | own-right&peer-wrong {n_rw}, followed {fol}, abandoned {ab}"
              f" | predicted net {res - ab:+d} vs observed net vs first answer {net:+d}")
        for key, (k, n) in zip(pool, ((res, n_wr), (fol, n_rw), (ab, n_rw))): pool[key][0] += k; pool[key][1] += n
    for key, (k, n) in pool.items(): print(f"   {m} pooled {key}: {wilson(k, n)}")
