"""Study H problem sets: MATH (hendrycks) test, Levels 4-5, integer gold answer (from the solution's last \\boxed{}).
Seed 20261123. settings: 300 problems (csv question,label). peer: 600 problems = the same 300 + 300 more (tasks.jsonl + labels.json)."""
import csv, json, random, re, sys, datasets
SUBS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]

def boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0: return None
    j = k = i + 7; d = 1
    while k < len(s) and d:
        d += (s[k] == "{") - (s[k] == "}"); k += 1
    return s[j:k - 1]

out = sys.argv[1]
pool = []
for sub in SUBS:
    for r in datasets.load_dataset("EleutherAI/hendrycks_math", sub, split="test"):
        a = boxed(r["solution"])
        a = None if a is None else a.replace(",", "").replace("\\$", "").replace("$", "").replace("\\!", "").strip()
        if r["level"] in ("Level 4", "Level 5") and a is not None and re.fullmatch(r"-?\d+", a):
            pool.append({"question": r["problem"], "label": int(a), "subject": sub, "level": r["level"]})
rng = random.Random(20261123); rng.shuffle(pool)
sel = pool[:600]
with open(f"{out}/math_hard300.csv", "x", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["question", "label"]); w.writeheader()
    for r in sel[:300]: w.writerow({"question": r["question"], "label": r["label"]})
with open(f"{out}/tasks.jsonl", "x") as f:
    for i, r in enumerate(sel): f.write(json.dumps({"tid": i, "question": r["question"]}) + "\n")
json.dump({str(i): r["label"] for i, r in enumerate(sel)}, open(f"{out}/labels.json", "x"))
json.dump([{"tid": i, "subject": r["subject"], "level": r["level"]} for i, r in enumerate(sel)], open(f"{out}/meta.json", "x"))
print(len(pool), "eligible;", len(sel), "selected")
