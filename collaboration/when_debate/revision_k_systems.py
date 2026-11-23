"""Post-hoc (after the second external review, 2026-11-23): study K systems under all eight framings.
System = Gemma judge on D (accuracy averaged over the 3 seeds), Gemma s0 elsewhere, over all 1,500 problems.
Differences vs Gemma s0 and vs vote(g0, g1, g2); two-sided sign-flip (k_analyze settings), Holm over 8 within each family.
Usage: revision_k_systems.py > reports/revision-reanalysis-20261122/k_systems.out"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k_analyze as ka

K = "/home/lian/swarmworld/research/reports/framing-rep-20261122-a"; O = K + "/out"
Y = {int(k): v for k, v in json.load(open(K + "/data/labels.json")).items()}
S, V, TOK, JT = {}, {}, {}, {}
for fn, m in (("gemma_solve_s0", "g"), ("gemma_solve_s1", "g"), ("gemma_solve_s2", "g"), ("qwen_solve", "q")):
    for l in open(f"{O}/{fn}.jsonl"):
        r = json.loads(l); S[(m, r["tid"], r["sample"])] = ka.ans(r)
        TOK[fn] = TOK.get(fn, 0) + r["prompt_tokens"] + r["output_tokens"] + r["prompt_tokens_2"] + r["output_tokens_2"]
for s in range(4):
    for l in open(f"{O}/judge_{s}.jsonl"):
        r = json.loads(l); V[(r["tid"], r["cond"], r["rep"])] = ka.ans(r)
        if r["rep"] == 0: JT[r["cond"]] = JT.get(r["cond"], 0) + r["prompt_tokens"] + r["output_tokens"] + r["prompt_tokens_2"] + r["output_tokens_2"]
T = sorted(Y)
D = {t for t in T if S[("g", t, 0)] is not None and S[("q", t, 0)] is not None and S[("g", t, 0)] != S[("q", t, 0)]}
base = {"gemma_s0": [float(S[("g", t, 0)] == Y[t]) for t in T],
        "vote3_gemma": [float(ka.vote([S[("g", t, 0)], S[("g", t, 1)], S[("g", t, 2)]]) == Y[t]) for t in T]}
SY = {f: [ka.mean([V[(t, f, r)] == Y[t] for r in ka.REPS]) if t in D else float(S[("g", t, 0)] == Y[t]) for t in T] for f in ka.FR}
print(f"problems {len(T)}, |D| {len(D)}; gemma_s0 {100*ka.mean(base['gemma_s0']):.2f}, vote3_gemma {100*ka.mean(base['vote3_gemma']):.2f}")
print(f"logical calls: judge systems 2 x {len(T)} + {len(D)} = {2*len(T)+len(D)} ({(2*len(T)+len(D))/len(T):.2f} per problem, cap 3); "
      f"vote3 {3*len(T)}; each call = 2 inference requests")
print(f"tokens (prompt+output, both requests): gemma s0 {TOK['gemma_solve_s0']:,}; vote3 {TOK['gemma_solve_s0']+TOK['gemma_solve_s1']+TOK['gemma_solve_s2']:,}; "
      f"judge system = gemma s0 + qwen s0 {TOK['qwen_solve']:,} + one seed of judge calls (by framing: "
      + ", ".join(f"{f} {JT[f]:,}" for f in ka.FR) + ")")
for ref in ("gemma_s0", "vote3_gemma"):
    dd = {f: [a - b for a, b in zip(SY[f], base[ref])] for f in ka.FR}
    ps = {f: ka.signflip(dd[f]) for f in ka.FR}; hp = ka.holm(ps)
    print(f"\n== system - {ref} (pp, 95% bootstrap CI; sign-flip p; Holm over 8)")
    for f in ka.FR:
        print(f"   {f:10s} {100*ka.mean(SY[f]):.2f}  {ka.f3(ka.boot(dd[f]))}  p={ps[f]:.4f}  Holm={hp[f]:.4f}")
