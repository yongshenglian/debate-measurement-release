"""Scoring of packet 3 (reports/human-audit-20261122/human2/PROTOCOL.md), written before any label existed.
Usage: human2_score.py <audit dir> --human A.csv B.csv [--adjudicated adj.csv]"""
import csv, json, os, random, sys
from collections import Counter
sys.path.insert(0, "/home/lian/swarmworld/research/collaboration/when_debate")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = sys.argv; sys.argv = [sys.argv[0], "/nonexistent"]
from score import norm, kappa, SRC, R, f3
sys.argv = _argv
from hard_analyze import gemma_val
from settings_analyze import reader, r1, vote, boot

A_DIR = sys.argv[1]; a = sys.argv[2:]
take = lambda flag, n: a[a.index(flag) + 1:a.index(flag) + 1 + n] if flag in a else []
lab = lambda p: {r["id"]: norm(r["label"]) for r in csv.DictReader(open(p)) if r["label"].strip()}
hA, hB = [lab(p) for p in take("--human", 2)]
adj = lab(take("--adjudicated", 1)[0]) if "--adjudicated" in a else {}
key = json.load(open(os.path.join(A_DIR, "human2", "key.json")))

both = [i for i in key if i in hA and i in hB]
po, kp = kappa([str(hA[i]) for i in both], [str(hB[i]) for i in both])
print(f"items {len(key)}; labelled by A {len(hA)}, B {len(hB)}; agreement {100*po:.1f}%, Cohen's kappa {kp:.3f} (n={len(both)})")
H, how = {}, Counter()
for i in key:
    if i in hA and i in hB and hA[i] == hB[i]: H[i] = hA[i]; how["agree"] += 1
    elif i in adj: H[i] = adj[i]; how["adjudicated"] += 1
    else: how["excluded"] += 1
print("final labels:", dict(how))

print("\n######## reader validity on pivotal outputs (share matching the human label; 95% bootstrap over items)")
for task in ("math", "mmlu"):
    ids = [i for i in H if key[i]["task"] == task]
    if not ids: continue
    for rd in ("ours", "gemma", "primary"):
        get = (lambda k: k["readings"]["ours"] if k["readings"]["ours"] is not None else k["readings"]["gemma"]) if rd == "primary" \
            else (lambda k, rd=rd: k["readings"][rd])
        x = [float(get(key[i]) == H[i]) for i in ids]
        m, lo, hi = boot(x)
        print(f"   {task} {rd:8s} {100*m:.1f}% [{100*lo:.1f}, {100*hi:.1f}]  (n={len(ids)})")

print("\n######## vote-level: paper reading vs paper reading with human labels on labelled pivotal outputs (pp, 95% CI)")
G = {}
for l in open(os.path.join(A_DIR, "stageA", "out.jsonl")):
    d = json.loads(l); G[d["id"]] = gemma_val(d["text"])
over = {key[i]["src"]: H[i] for i in H}
for mode in ("paper", "human"):
    V = {}
    for sub, (task, arms) in SRC.items():
        study = sub.split("/")[0]
        for arm in arms:
            D = json.load(open(f"{R}/{sub}/{arm}.json")); rd = reader(D["cfg"]["family"]); T = []
            for qi, ex in enumerate(D["results"]):
                K = len(ex["rounds"]) - 1; rr = {}
                for k in sorted({0, K}):
                    vals = []
                    for ag, t in enumerate(ex["rounds"][k]):
                        iid = f"{study}|{arm}|{k}|{qi}|{ag}"; o = rd(t); v = o if o is not None else G[iid]
                        if mode == "human" and iid in over: v = over[iid]
                        vals.append(v)
                    rr[k] = vals
                T.append((r1(float(ex["label"])), rr[0], rr[K]))
            V[(task, arm)] = T
    print(f"== reading: {mode}")
    for task in ("math", "mmlu"):
        v20 = [vote(r0, y) for y, r0, _ in V[(task, "H_vote20")]]; P1 = {}
        for (t2, arm), T in V.items():
            if t2 != task or arm == "H_vote20": continue
            fin = [vote(rk, y) for y, _, rk in T]; r0v = [vote(r0, y) for y, r0, _ in T]
            P1[arm] = [x - y for x, y in zip(fin, r0v)]
            print(f"   {task} {arm:13s} final - vote20 {f3(boot([x - y for x, y in zip(fin, v20)]))} | final - r0 vote {f3(boot(P1[arm]))}")
        for x, y in (("H_ans", "H_full"), ("H_ans_indep", "H_full_indep")):
            if x in P1 and y in P1:
                print(f"   {task} contrast {x} - {y} on (final - r0): {f3(boot([p - q for p, q in zip(P1[x], P1[y])]))}")
