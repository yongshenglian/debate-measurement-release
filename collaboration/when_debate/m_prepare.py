"""Study M problem sets: MMLU-Pro test (TIGER-Lab/MMLU-Pro, cached), all categories except math (studied in H/I/T).
600 questions, seed 20261124, shuffled within the pool; options numbered 1..k; gold = answer_index + 1.
debate: first 300 (csv question,label); peer: all 600 (tasks.jsonl, labels.json, meta.json)."""
import csv, json, random, sys, datasets
out = sys.argv[1]
d = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="test")
pool = [r for r in d if r["category"] != "math"]
rng = random.Random(20261124); rng.shuffle(pool); sel = pool[:600]
def text(r):
    return r["question"].strip() + "\n\nOptions:\n" + "\n".join(f"{i+1}. {o}" for i, o in enumerate(r["options"])) + \
        "\n\nThe final answer is the number of the correct option."
with open(f"{out}/mmlupro300.csv", "x", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["question", "label"]); w.writeheader()
    for r in sel[:300]: w.writerow({"question": text(r), "label": r["answer_index"] + 1})
with open(f"{out}/tasks.jsonl", "x") as f:
    for i, r in enumerate(sel): f.write(json.dumps({"tid": i, "question": text(r)}) + "\n")
json.dump({str(i): r["answer_index"] + 1 for i, r in enumerate(sel)}, open(f"{out}/labels.json", "x"))
json.dump([{"tid": i, "category": r["category"], "question_id": r["question_id"], "n_options": len(r["options"])} for i, r in enumerate(sel)], open(f"{out}/meta.json", "x"))
print(len(pool), "eligible;", len(sel), "selected")
