"""Targeted human audit, packet 3 (reports/human-audit-20261122/human2/PROTOCOL.md).
Pivotal outputs: unaudited outputs where OURS and G disagree (N or D) and switching that output's reading between OURS and G,
with the other agents in its round read as in the paper (PRIMARY), changes whether the problem's vote is correct.
Draws N_PER pivotal outputs per task (seed fixed), shuffled. Writes items.json, key.json, pool.json and label.html (own storage key).
Usage: human2_sample.py <audit dir>"""
import json, os, random, sys
from collections import Counter
sys.path.insert(0, "/home/lian/swarmworld/research/collaboration/when_debate")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = sys.argv; sys.argv = [sys.argv[0], "/nonexistent"]
from score import SRC, R
sys.argv = _argv
from hard_analyze import gemma_val
from settings_analyze import reader, r1, vote

A_DIR = sys.argv[1]; OUT = os.path.join(A_DIR, "human2"); N_PER = {"math": 50, "mmlu": 50}; SEED = 20261124
key1 = json.load(open(os.path.join(A_DIR, "key.json"))); done = {k["src"] for k in key1.values()}
G = {}
for l in open(os.path.join(A_DIR, "stageA", "out.jsonl")):
    d = json.loads(l); G[d["id"]] = gemma_val(d["text"])
pool = []
for sub, (task, arms) in SRC.items():
    study = sub.split("/")[0]
    for arm in arms:
        D = json.load(open(f"{R}/{sub}/{arm}.json")); rd = reader(D["cfg"]["family"])
        for qi, ex in enumerate(D["results"]):
            y = r1(float(ex["label"])); K = len(ex["rounds"]) - 1
            for k in sorted({0, K}):
                texts = ex["rounds"][k]; ids = [f"{study}|{arm}|{k}|{qi}|{a}" for a in range(len(texts))]
                ours = [rd(t) for t in texts]; prim = [o if o is not None else G[i] for o, i in zip(ours, ids)]
                for a, i in enumerate(ids):
                    if ours[a] == G[i] or i in done: continue
                    v_o = vote(prim[:a] + [ours[a]] + prim[a + 1:], y); v_g = vote(prim[:a] + [G[i]] + prim[a + 1:], y)
                    if v_o != v_g:
                        pool.append({"src": i, "task": task, "arm": arm, "round": "r0" if k == 0 else "final",
                                     "stratum": "N" if ours[a] is None else "D", "readings": {"ours": ours[a], "gemma": G[i]},
                                     "text": texts[a], "question": ex["question"]})
rng = random.Random(SEED); sample = []
for task, n in N_PER.items():
    p = [x for x in pool if x["task"] == task]; sample += rng.sample(p, min(n, len(p)))
rng.shuffle(sample)
for j, it in enumerate(sample): it["id"] = f"P{j + 1:03d}"
os.makedirs(OUT, exist_ok=True); os.chdir(OUT)
json.dump([{"id": it["id"], "question": it["question"], "text": it["text"]} for it in sample], open("items.json", "x"), ensure_ascii=False, indent=0)
json.dump({it["id"]: {k: it[k] for k in ("src", "task", "arm", "round", "stratum", "readings")} for it in sample}, open("key.json", "x"), indent=1)
json.dump(dict(Counter(f"{x['task']}|{x['stratum']}" for x in pool)), open("pool.json", "x"), indent=1)
page = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_template.html")).read()
assert page.count('"debate-audit-labels-v1"') == 1
page = page.replace('"debate-audit-labels-v1"', '"debate-audit-human2-v1"')
data = json.dumps([{"id": it["id"], "question": it["question"], "text": it["text"]} for it in sample], ensure_ascii=False).replace("</", "<\\/")
open("label.html", "x").write(page.replace("/*ITEMS*/[]", data))
print("pool", dict(Counter(f"{x['task']}|{x['stratum']}" for x in pool)), "| sample", dict(Counter(f"{x['task']}|{x['stratum']}" for x in sample)))
