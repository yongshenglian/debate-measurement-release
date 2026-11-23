import json, os, sys, collections
R="/home/lian/swarmworld/research/reports"
SETS={"GSM8K":(f"{R}/reverse-peer-20261120-a/out",None,f"{R}/reverse-peer-20261120-a/labels.json"),
      "MATH":(f"{R}/debate-hard-20261120-a/out",f"{R}/debate-indep-20261121-a/out",f"{R}/debate-hard-20261120-a/data/labels.json"),
      "MMLU":(f"{R}/mmlupro-20261121-a/out-neutral",f"{R}/mmlupro-20261121-a/out-indep",f"{R}/mmlupro-20261121-a/data/labels.json")}
def ans(r): return None if r.get("parsed") is None else r["parsed"]["answer"]
def load(d):
    S,V={},{}
    for m in ("qwen","gemma"):
        for l in open(f"{d}/{m}_solve.jsonl"): r=json.loads(l); S[(m,r["tid"],r["sample"])]=ans(r)
        for l in open(f"{d}/{m}_review.jsonl"): r=json.loads(l); V[(m,r["tid"],r["cond"])]=ans(r)
    return S,V
for name,(dn,di,lab) in SETS.items():
    Y={int(k):v for k,v in json.load(open(lab)).items()}
    S,V=load(dn); VI=load(di)[1] if di else {}
    T=[t for t in Y if all((m,t,c) in V for m in("qwen","gemma") for c in("self","peer_qwen","peer_gemma"))]
    n=len(T); acc=lambda f: 100*sum(f(t)==Y[t] for t in T)/n
    g0=lambda t:S[("gemma",t,0)]; g1=lambda t:S[("gemma",t,1)]; q0=lambda t:S[("qwen",t,0)]; q1=lambda t:S[("qwen",t,1)]
    def vote(xs):
        c=collections.Counter(x for x in xs if x is not None)
        if not c: return None
        top=max(c.values()); 
        for x in xs:
            if x is not None and c[x]==top: return x
    P={
     "gemma s0":g0, "gemma self-review":lambda t:V[("gemma",t,"self")],
     "gemma reads qwen":lambda t:V[("gemma",t,"peer_qwen")],
     "gemma reads gemma s1":lambda t:V[("gemma",t,"peer_gemma")],
     "qwen reads gemma":lambda t:V[("qwen",t,"peer_gemma")],
     "vote g0,g1,q0":lambda t:vote([g0(t),g1(t),q0(t)]),
     "vote g0,q0,q1":lambda t:vote([g0(t),q0(t),q1(t)]),
     "vote g0,g1,q0,q1":lambda t:vote([g0(t),g1(t),q0(t),q1(t)]),
     "gate: g reads q iff q0==q1!=g0":lambda t:V[("gemma",t,"peer_qwen")] if (q0(t)==q1(t) and q0(t)!=g0(t)) else g0(t),
     "gate: g reads q iff g0!=g1":lambda t:V[("gemma",t,"peer_qwen")] if g0(t)!=g1(t) else g0(t),
     "gate: q reads g; ans=g0 if agree":lambda t:V[("qwen",t,"peer_gemma")],
    }
    if VI:
        P["gemma reads qwen +indep"]=lambda t:VI[("gemma",t,"peer_qwen")]
        P["gate indep: iff q0==q1!=g0"]=lambda t:VI[("gemma",t,"peer_qwen")] if (q0(t)==q1(t) and q0(t)!=g0(t)) else g0(t)
    print(f"== {name} n={n}  oracle(g0 or q0) {100*sum(Y[t] in (g0(t),q0(t)) for t in T)/n:.1f}")
    for k,f in P.items(): print(f"   {k:38s} {acc(f):.1f}")
    # cascade: Qwen answers; consult Gemma only when q0!=q1
    G=[t for t in T if q0(t)!=q1(t)]; fr=len(G)/n
    c_read=acc(lambda t:V[("qwen",t,"peer_gemma")] if q0(t)!=q1(t) else q0(t))
    c_take=acc(lambda t:g0(t) if q0(t)!=q1(t) else q0(t))
    print(f"   cascade (gemma on {100*fr:.0f}% where q0!=q1): qwen reads gemma {c_read:.1f} | just take g0 {c_take:.1f} | qwen s0 {acc(q0):.1f}")
