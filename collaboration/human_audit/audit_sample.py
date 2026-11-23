"""Human audit sample (reports/human-audit-20261122/PROTOCOL.md). Run after stage A (blind Gemma reading of all part-A outputs).
Usage: audit_sample.py <audit dir>   (reads <audit dir>/stageA/out.jsonl; writes items.json, key.json, strata.json, label.html)"""
import json, os, random, re, sys
sys.path.insert(0, "/home/lian/swarmworld/research/collaboration/when_debate")
from hard_analyze import gemma_val
from settings_analyze import reader, r1
sys.path.insert(0, "/home/lian/swarmworld/research/collaboration/dmad_rerun")
_argv = sys.argv; sys.argv = [sys.argv[0], "/nonexistent"]
from sensitivity_unclosed import unclosed
from analyze import validated
sys.argv = _argv

R = "/home/lian/swarmworld/research/reports"
A_DIR = sys.argv[1]
SRC = {"debate-hard-20261120-a/out": ("math", ["H_full", "H_ans", "H_dovword", "H_vote20"]),
       "debate-indep-20261121-a/out": ("math", ["H_full_indep", "H_ans_indep"]),
       "mmlupro-20261121-a/out-debate": ("mmlu", ["H_full", "H_ans", "H_full_indep", "H_ans_indep", "H_vote20"])}
CAP_ND, N_A = 12, 4
B_RULE, B_VAL = 40, 10

G = {}
for l in open(os.path.join(A_DIR, "stageA", "out.jsonl")):
    d = json.loads(l); G[d["id"]] = gemma_val(d["text"])

cells, pool = {}, []
for sub, (task, arms) in SRC.items():
    study = sub.split("/")[0]
    for arm in arms:
        D = json.load(open(f"{R}/{sub}/{arm}.json")); rd = reader(D["cfg"]["family"])
        for qi, ex in enumerate(D["results"]):
            K = len(ex["rounds"]) - 1
            for k in sorted({0, K}):
                rg = "r0" if k == 0 else "final"
                for a, t in enumerate(ex["rounds"][k]):
                    iid = f"{study}|{arm}|{k}|{qi}|{a}"
                    ours = rd(t); g = G[iid]
                    s = "N" if ours is None else ("A" if ours == g else "D")
                    cells.setdefault((task, arm, rg), {}).setdefault(s, []).append(
                        {"part": "A", "src": iid, "task": task, "arm": arm, "round": rg, "stratum": s,
                         "readings": {"ours": ours, "gemma": g, "primary": ours if ours is not None else g}, "text": t, "question": ex["question"]})

rng = random.Random(20261125)
sample = []
for c, st in sorted(cells.items()):
    nd = st.get("N", []) + st.get("D", [])
    sample += nd if len(nd) <= CAP_ND else rng.sample(nd, CAP_ND)
    sample += rng.sample(st.get("A", []), min(N_A, len(st.get("A", []))))

# part B: DMAD Llama, official template, fixed arm (the paper's -11.8 row)
P = f"{R}/dmad-rerun-20261120-a/out/llama-t_fixed.json"
Rb = json.load(open(P))["results"]; bcells = {"rule": [], "validated": []}
for qi, ex in enumerate(Rb):
    rr = ex["round_results"]
    for k in sorted({0, len(rr) - 1}):
        for a, ag in enumerate(rr[k]["agents_responses"]):
            c = ag["completion"]; v = validated(c); u, used = unclosed(c)
            key = "rule" if used else ("validated" if v is not None else None)
            if key:
                bcells[key].append({"part": "B", "src": f"llama-t_fixed|{k}|{qi}|{a}", "task": "gsm8k", "arm": "dmad_llama_fixed",
                                    "round": "r0" if k == 0 else "final", "stratum": key,
                                    "readings": {"validated": None if v is None else r1(v), "unclosed": None if u is None else r1(u)}, "text": c, "question": ex["question"]})
sample += rng.sample(bcells["rule"], min(B_RULE, len(bcells["rule"]))) + rng.sample(bcells["validated"], min(B_VAL, len(bcells["validated"])))

rng.shuffle(sample)
for i, it in enumerate(sample):
    it["id"] = f"H{i + 1:03d}"
os.chdir(A_DIR)
json.dump([{"id": it["id"], "question": it["question"], "text": it["text"]} for it in sample], open("items.json", "x"), ensure_ascii=False, indent=0)
json.dump({it["id"]: {k: it[k] for k in ("part", "src", "task", "arm", "round", "stratum", "readings")} for it in sample},
          open("key.json", "x"), indent=1)
strata = {f"{t}|{a}|{r}|{s}": len(v) for (t, a, r), st in cells.items() for s, v in st.items()}
strata.update({f"B|{k}": len(v) for k, v in bcells.items()})
json.dump(strata, open("strata.json", "x"), indent=1)
page = open("/home/lian/swarmworld/research/collaboration/human_audit/label_template.html").read()
data = json.dumps([{"id": it["id"], "question": it["question"], "text": it["text"]} for it in sample], ensure_ascii=False).replace("</", "<\\/")
open("label.html", "x").write(page.replace("/*ITEMS*/[]", data))
from collections import Counter
print("items", len(sample), dict(Counter(it["part"] + ":" + it["stratum"] for it in sample)))
