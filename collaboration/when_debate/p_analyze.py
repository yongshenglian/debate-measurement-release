"""Study P analysis (reports/private-revision-20261123-a/PROTOCOL.md). Usage: p_analyze.py <judge: gemma|qwen38> <study P out subdir>
Peer review (8 framings, from study K or L) against private revision (this study), same judge, same disagreement set D, same seeds.
Unit: the problem; outcomes averaged over the 3 seeds. Primary (Holm over 3 within judge): accuracy on D, f - self for
f in {neutral, nswap, misattr}; two-sided sign-flip (100,000 draws), percentile bootstrap (20,000). Secondary: all eight framings;
correctness-cell decomposition (contributions to accuracy over all 1,500 problems); systems; truncation."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k_analyze as ka

J, SUB = sys.argv[1], sys.argv[2]
R = "/home/lian/swarmworld/research/reports"; KD = f"{R}/framing-rep-20261122-a"; PD = f"{R}/private-revision-20261123-a"
if J == "gemma":
    SOL, JDIR = [f"{KD}/out/gemma_solve_s{s}.jsonl" for s in range(3)], f"{KD}/out"
else:
    SOL, JDIR = [f"{R}/framing-gen-20261122-a/out-qwen38/judge_solve_s{s}.jsonl" for s in range(3)], f"{R}/framing-gen-20261122-a/out-qwen38"
Y = {int(k): v for k, v in json.load(open(f"{KD}/data/labels.json")).items()}
S = {}
for p in SOL + [f"{KD}/out/qwen_solve.jsonl"]:
    m = "p" if "qwen_solve" in p else "j"
    for l in open(p):
        r = json.loads(l); S[(m, r["tid"], r["sample"])] = ka.ans(r)
V, LEN, N = {}, {}, {}
for d, pat in ((JDIR, "judge_{}.jsonl"), (f"{PD}/{SUB}", "self_{}.jsonl")):
    for s in range(4):
        p = os.path.join(d, pat.format(s))
        if not os.path.exists(p): continue
        for l in open(p):
            r = json.loads(l); V[(r["tid"], r["cond"], r["rep"])] = ka.ans(r)
            LEN[r["cond"]] = LEN.get(r["cond"], 0) + (r["finish_reason"] == "length"); N[r["cond"]] = N.get(r["cond"], 0) + 1
T = sorted(Y); FR = ka.FR + ("self",)
Dall = [t for t in T if S[("j", t, 0)] is not None and S[("p", t, 0)] is not None and S[("j", t, 0)] != S[("p", t, 0)]]
D = [t for t in Dall if all((t, f, r) in V for f in FR for r in ka.REPS)]
print(f"judge {J}; |D| {len(Dall)}, complete for all framings and self {len(D)}; invalid self answers "
      f"{sum(V[(t, 'self', r)] is None for t in D for r in ka.REPS)}")
print("first turn ends at the token limit (%): " + ", ".join(f"{f} {100 * LEN[f] / N[f]:.1f}" for f in FR if N.get(f)))
acc = {f: {t: ka.mean([V[(t, f, r)] == Y[t] for r in ka.REPS]) for t in D} for f in FR}
print("accuracy on D: " + "  ".join(f"{f} {100 * ka.mean(list(acc[f].values())):.1f}" for f in FR))

print("\n######## primary: accuracy on D, framing - private revision (Holm over 3)")
P, E = {}, {}
for f in ("neutral", "nswap", "misattr"):
    d = [acc[f][t] - acc["self"][t] for t in D]; E[f] = ka.boot(d); P[f] = ka.signflip(d)
H = ka.holm(P)
for f in P:
    print(f"   {f:8s} - self: {ka.f3(E[f])}  p={P[f]:.4f}  Holm={H[f]:.4f}")

print("\n######## secondary: all framings - self, accuracy on D (unadjusted)")
for f in ka.FR:
    d = [acc[f][t] - acc["self"][t] for t in D]
    print(f"   {f:10s} {ka.f3(ka.boot(d))}  p={ka.signflip(d):.4f}")

print("\n######## correctness-cell decomposition: contributions to accuracy over all 1,500 problems (pp)")
cell = {t: ("peer_right" if S[("p", t, 0)] == Y[t] else "own_right" if S[("j", t, 0)] == Y[t] else "both_wrong") for t in D}
n = len(T)
for f in FR:
    rec = sum(acc[f][t] for t in D if cell[t] == "peer_right") / n
    cor = -sum(1 - acc[f][t] for t in D if cell[t] == "own_right") / n
    bw = sum(acc[f][t] for t in D if cell[t] == "both_wrong") / n
    print(f"   {f:10s} gains peer's right answer {100*rec:+.2f} | loses own right answer {100*cor:+.2f} | new right answer, both wrong {100*bw:+.2f} | net {100*(rec+cor+bw):+.2f}")
print("   cell sizes:", {c: sum(v == c for v in cell.values()) for c in ("peer_right", "own_right", "both_wrong")})
print("   framing - self within each cell (pp of 1,500; bootstrap over the cell's problems):")
for f in ka.FR:
    parts = []
    for c in ("peer_right", "own_right", "both_wrong"):
        ts = [t for t in D if cell[t] == c]
        d = [(acc[f][t] - acc["self"][t]) * len(ts) / n for t in ts]
        parts.append(f"{c} {ka.f3(ka.boot(d))}")
    print(f"   {f:10s} " + " | ".join(parts))

print("\n######## systems over all 1,500 problems (judge on D, judge s0 elsewhere)")
Ds = set(D)
g0 = [float(S[("j", t, 0)] == Y[t]) for t in T]
v3 = [float(ka.vote([S[("j", t, 0)], S[("j", t, 1)], S[("j", t, 2)]]) == Y[t]) for t in T]
sysv = {f: [acc[f][t] if t in Ds else float(S[("j", t, 0)] == Y[t]) for t in T] for f in FR}
print(f"   judge s0 {100*ka.mean(g0):.2f}; vote(j0, j1, j2) {100*ka.mean(v3):.2f}; private revision {100*ka.mean(sysv['self']):.2f}")
d = [a - b for a, b in zip(sysv["self"], g0)]; print(f"   private revision - judge s0: {ka.f3(ka.boot(d))}  p={ka.signflip(d):.4f}")
d = [a - b for a, b in zip(sysv["self"], v3)]; print(f"   private revision - vote3: {ka.f3(ka.boot(d))}  p={ka.signflip(d):.4f}")
print(f"   note: revising only on D uses the peer's answer to choose D, so this is a matched comparison, not a stand-alone policy;"
      f" logical calls as run: {3000 + len(D)} for both private revision and peer review on D; vote3 4,500")
