"""Pre-registered scoring of the human audit (reports/human-audit-20261122/PROTOCOL.md).
Usage: score.py <audit dir> labels_A.csv [labels_B.csv] [--adjudicated adjudicated.csv]
  Two files: agreed labels are final; disagreements need --adjudicated (id,label,prose), else excluded and reported.
Outputs: inter-rater agreement and kappa; reweighted reader validity per cell; vote-level sensitivity of every debate contrast."""
import csv, json, math, os, re, sys
from collections import Counter, defaultdict
sys.path.insert(0, "/home/lian/swarmworld/research/collaboration/when_debate")
from hard_analyze import gemma_val
from settings_analyze import reader, r1, vote, boot
sys.path.insert(0, "/home/lian/swarmworld/research/collaboration/dmad_rerun")
_argv = sys.argv; sys.argv = [sys.argv[0], "/nonexistent"]
from sensitivity_unclosed import unclosed
from analyze import validated
from analyze import vote as dvote
from utils.eval_utils import clean_answer
sys.argv = _argv

R = "/home/lian/swarmworld/research/reports"
SRC = {"debate-hard-20261120-a/out": ("math", ["H_full", "H_ans", "H_dovword", "H_vote20"]),
       "debate-indep-20261121-a/out": ("math", ["H_full_indep", "H_ans_indep"]),
       "mmlupro-20261121-a/out-debate": ("mmlu", ["H_full", "H_ans", "H_full_indep", "H_ans_indep", "H_vote20"])}
f3 = lambda x: f"{100*x[0]:+.1f} [{100*x[1]:+.1f}, {100*x[2]:+.1f}]"


def norm(v):
    """NONE/MULTI -> None; a number or a single expression (1/2, sqrt(5), 4+2sqrt(3), 2^48) -> its value rounded as r1;
    several values ("2, -1") or anything not evaluable -> a string token that matches no machine reading (amendment, before any human label)."""
    v = (v or "").strip()
    if v == "": return "MISSING"
    if v.upper() in ("NONE", "MULTI"): return None
    if "," in v: return "LIST:" + re.sub(r"\s+", "", v)
    e = re.sub(r"(\d|\))\s*(sqrt|pi|\()", r"\1*\2", v.replace(" ", "")).replace("^", "**")
    try:
        if not re.fullmatch(r"[-+*/().0-9a-z]*", e) or set(re.findall(r"[a-z]+", e)) - {"sqrt", "pi"}: raise ValueError
        return r1(eval(e, {"__builtins__": {}}, {"sqrt": math.sqrt, "pi": math.pi}))
    except Exception:
        return "EXPR:" + v.replace(" ", "")


def load(path):
    return {r["id"]: (r["label"], r.get("prose", "")) for r in csv.DictReader(open(path)) if r["label"].strip()}


def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b); pe = sum(ca[k] * cb[k] for k in ca) / n ** 2
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    """score.py <audit dir> --human A.csv B.csv --claude C.csv [--adjudicated adj.csv]
    Final label: the two humans when they agree; otherwise the majority of the three readers (humans + blind Claude reader);
    items where all three differ go to adjudication by the two humans together (adj.csv), else are excluded and reported.
    Sensitivity: humans-only (items where the two humans agree)."""
    A_DIR = sys.argv[1]; a = sys.argv[2:]
    take = lambda flag, n: a[a.index(flag) + 1:a.index(flag) + 1 + n] if flag in a else []
    hA, hB = [load(x) for x in take("--human", 2)]
    cC = load(take("--claude", 1)[0]) if "--claude" in a else {}
    adj = load(take("--adjudicated", 1)[0]) if "--adjudicated" in a else {}
    key = json.load(open(os.path.join(A_DIR, "key.json"))); strata = json.load(open(os.path.join(A_DIR, "strata.json")))
    print(f"items {len(key)}; labelled by human A {len(hA)}, human B {len(hB)}, Claude {len(cC)}")
    for n1, L1, n2, L2 in (("A", hA, "B", hB), ("A", hA, "Claude", cC), ("B", hB, "Claude", cC)):
        both = [i for i in key if i in L1 and i in L2]
        if both:
            po, kp = kappa([str(norm(L1[i][0])) for i in both], [str(norm(L2[i][0])) for i in both])
            print(f"agreement {n1}-{n2}: {100*po:.1f}%, Cohen's kappa {kp:.3f} (n={len(both)})")
    final, human_only, how = {}, {}, Counter()
    for i in key:
        if i not in hA or i not in hB: how["missing human label"] += 1; continue
        la, lb = norm(hA[i][0]), norm(hB[i][0])
        if la == lb:
            pa, pb = hA[i][1].strip(), hB[i][1].strip()
            final[i] = (hA[i][0], pa if pa == pb else (pa or pb)); human_only[i] = final[i]; how["humans agree"] += 1
        elif i in cC and norm(cC[i][0]) in (la, lb):
            src = hA if norm(cC[i][0]) == la else hB; final[i] = src[i]; how["majority with Claude"] += 1
        elif i in adj:
            final[i] = adj[i]; how["adjudicated by humans"] += 1
        else:
            how["unresolved (excluded)"] += 1
    print("final labels:", dict(how))
    for tag, F in (("PRIMARY (humans, Claude as tie-break)", final), ("SENSITIVITY (humans agree only)", human_only)):
        print(f"\n################ {tag}")
        run(A_DIR, key, strata, F)


def run(A_DIR, key, strata, final):
    H = {i: norm(v[0]) for i, v in final.items()}
    HP = {i: (norm(v[1]) if (v[1] or "").strip() else norm(v[0])) for i, v in final.items()}   # prose-precedence sensitivity
    print(f"items with a prose answer different from the formatted answer: {sum(1 for v in final.values() if (v[1] or '').strip())}")
    # ---- part A: reader validity per cell, reweighted to the cell population
    print("\n######## Part A: agreement with the human label (formatted-answer rule), reweighted within each cell")
    cellrows = defaultdict(list)
    for i, k in key.items():
        if k["part"] == "A" and i in H: cellrows[(k["task"], k["arm"], k["round"])].append((k, H[i]))
    for c, rows in sorted(cellrows.items()):
        line = f"{c[0]:4s} {c[1]:13s} {c[2]:5s}"
        for rd in ("ours", "gemma", "primary"):
            num = den = 0.0
            for s in ("N", "D", "A"):
                pop = strata.get(f"{c[0]}|{c[1]}|{c[2]}|{s}", 0); xs = [k["readings"][rd] == h for k, h in rows if k["stratum"] == s]
                if pop and xs: num += pop * sum(xs) / len(xs); den += pop
                elif pop and s == "A": num += pop; den += pop   # unsampled agree stratum: assumed correct (checked where sampled)
            line += f" | {rd} {100*num/den:.1f}%" if den else ""
        print(line)

    # ---- human-corrected readings for the vote-level sensitivity
    side = defaultdict(Counter)   # (task, arm, round, stratum) -> which machine reading the humans sided with
    for i, k in key.items():
        if k["part"] != "A" or i not in H: continue
        h = H[i]; c = (k["task"], k["arm"], k["round"], k["stratum"])
        side[c]["ours"] += k["readings"]["ours"] == h; side[c]["gemma"] += k["readings"]["gemma"] == h
    lab_by_src = {key[i]["src"]: i for i in H if key[i]["part"] == "A"}
    G = {}
    for l in open(os.path.join(A_DIR, "stageA", "out.jsonl")):
        d = json.loads(l); G[d["id"]] = gemma_val(d["text"])

    def reading(mode, iid, task, arm, rg, ours):
        g = G[iid]; prim = ours if ours is not None else g
        if mode == "primary": return prim
        if mode == "gemma": return g
        s = "N" if ours is None else ("A" if ours == g else "D")
        if iid in lab_by_src:
            j = lab_by_src[iid]; return H[j] if mode == "human" else HP[j]
        if s == "A": return ours
        sc = side[(task, arm, rg, s)]
        return g if sc["gemma"] > sc["ours"] else ours if sc["ours"] > sc["gemma"] else prim

    print("\n######## Part A: vote-level sensitivity of every debate contrast (pp, 95% CI)")
    for mode in ("primary", "gemma", "human", "human_prose"):
        V = {}
        for sub, (task, arms) in SRC.items():
            study = sub.split("/")[0]
            for arm in arms:
                D = json.load(open(f"{R}/{sub}/{arm}.json")); rd = reader(D["cfg"]["family"]); T = []
                for qi, ex in enumerate(D["results"]):
                    K = len(ex["rounds"]) - 1
                    rr = {k: [reading(mode, f"{study}|{arm}|{k}|{qi}|{a}", task, arm, "r0" if k == 0 else "final", rd(t))
                              for a, t in enumerate(ex["rounds"][k])] for k in sorted({0, K})}
                    T.append((r1(float(ex["label"])), rr[0], rr[K]))
                V[(task, arm)] = T
        print(f"== reading: {mode}")
        for task in ("math", "mmlu"):
            v20 = [vote(r0, y) for y, r0, _ in V[(task, "H_vote20")]]
            P1 = {}
            for (t2, arm), T in V.items():
                if t2 != task or arm == "H_vote20": continue
                fin = [vote(rk, y) for y, _, rk in T]; r0v = [vote(r0, y) for y, r0, _ in T]
                P1[arm] = [a - b for a, b in zip(fin, r0v)]
                print(f"   {task} {arm:13s} final - vote20 {f3(boot([a - b for a, b in zip(fin, v20)]))} | final - r0 vote {f3(boot(P1[arm]))}")
            for a, b in (("H_ans", "H_full"), ("H_ans_indep", "H_full_indep")):
                if a in P1 and b in P1:
                    print(f"   {task} contrast {a} - {b} on (final - r0): {f3(boot([x - y for x, y in zip(P1[a], P1[b])]))}")

    # ---- part B: DMAD Llama, official template, fixed arm
    print("\n######## Part B: DMAD Llama (official template, fixed): last - round-0 vote")
    sideB = Counter(); labB = {}
    for i, k in key.items():
        if k["part"] == "B" and i in H:
            labB[k["src"]] = H[i]
            if k["stratum"] == "rule": sideB["unclosed"] += k["readings"]["unclosed"] == H[i]; sideB["none"] += H[i] is None
    Rb = json.load(open(f"{R}/dmad-rerun-20261120-a/out/llama-t_fixed.json"))["results"]
    for mode in ("validated", "unclosed", "human"):
        d = []
        for qi, ex in enumerate(Rb):
            y = float(clean_answer(str(ex["label"]))); rr = ex["round_results"]; vs = []
            for k in (0, len(rr) - 1):
                vals = []
                for a, ag in enumerate(rr[k]["agents_responses"]):
                    c = ag["completion"]; v = validated(c); u, used = unclosed(c)
                    if mode == "validated": vals.append(v)
                    elif mode == "unclosed": vals.append(u)
                    else:
                        src = f"llama-t_fixed|{k}|{qi}|{a}"
                        vals.append(labB[src] if src in labB else (u if (not used or sideB["unclosed"] >= sideB["none"]) else v))
                vs.append(dvote(vals, y))
            d.append(vs[1] - vs[0])
        print(f"   {mode:9s} last - r0: {f3(boot(d))}")
    print(f"   humans on rule-recovered items: agree with the rule {sideB['unclosed']}, say no answer {sideB['none']}")


if __name__ == "__main__":
    main()
