"""Study K problem set (reports/framing-rep-20261122-a/PROTOCOL.md): untouched MMLU-Pro non-math questions.
Same pool, seed and shuffle as study M (m_prepare.py); study M used positions 0..599, study K takes positions 600..2099.
Asserts that positions 0..599 reproduce study M's question ids, so the two sets cannot overlap.
Writes tasks.jsonl (tid 0..1499), labels.json (gold option number), meta.json. Usage: k_prepare.py <out dir> <study M meta.json>"""
import json, random, sys
import datasets

out, m_meta = sys.argv[1], sys.argv[2]
d = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="test")
pool = [r for r in d if r["category"] != "math"]
rng = random.Random(20261124); rng.shuffle(pool)
used = [m["question_id"] for m in json.load(open(m_meta))]
assert [r["question_id"] for r in pool[:600]] == used, "shuffle does not reproduce study M"
sel = pool[600:2100]
assert not set(r["question_id"] for r in sel) & set(used)


def text(r):   # identical to m_prepare.py
    return r["question"].strip() + "\n\nOptions:\n" + "\n".join(f"{i+1}. {o}" for i, o in enumerate(r["options"])) + \
        "\n\nThe final answer is the number of the correct option."


with open(f"{out}/tasks.jsonl", "x") as f:
    for i, r in enumerate(sel): f.write(json.dumps({"tid": i, "question": text(r)}) + "\n")
json.dump({str(i): r["answer_index"] + 1 for i, r in enumerate(sel)}, open(f"{out}/labels.json", "x"))
json.dump([{"tid": i, "category": r["category"], "question_id": r["question_id"], "n_options": len(r["options"])}
           for i, r in enumerate(sel)], open(f"{out}/meta.json", "x"))
print(len(pool), "eligible;", len(sel), "selected; no overlap with study M")
