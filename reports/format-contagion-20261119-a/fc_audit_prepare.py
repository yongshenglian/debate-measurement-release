"""Blind extraction audit sample: debate responses with no label, for an independent LLM extractor."""
import json, random, sys
from pathlib import Path
sys.path.insert(0, "/home/lian/swarmworld/research/reports/peer-theory-20261119")
from empties import original, robust  # noqa: E402
from rescore import rows, r1  # noqa: E402

BASE = "/home/lian/swarmworld/research/reports/"
RUNS = {"gsm_dense": "debate-vote-gsm8k-20261117-a/dense/out/history/gsm8k_300__qwen2.5-7b_N=5_R=5.jsonl",
        "gsm_sparse": "debate-vote-gsm8k-20261117-a/sparse/out/history/gsm8k_300__qwen2.5-7b_N=5_R=5_SPARSE.jsonl",
        "ar_dense": "debate-vote-arithmetic-20261116-a/dense/out/history/arithmetics_100__qwen2.5-7b_N=5_R=5.jsonl",
        "ar_sparse": "debate-vote-arithmetic-20261116-a/sparse/out/history/arithmetics_100__qwen2.5-7b_N=5_R=5_SPARSE.jsonl"}
QUOTA = {"recovered": 240, "agree": 60, "disagree": 60, "both_none": 40}
SYSTEM = ("You read a model's response to a math problem and report the single final numeric answer the "
          "model committed to. Ignore numbers it only mentions in passing or attributes to others. If it "
          "never commits to a final numeric answer, report null. Reply with JSON only: {\"answer\": <number or null>}.")

def main(out):
    out = Path(out)
    pool = {k: [] for k in QUOTA}
    for run, p in RUNS.items():
        for ti, t in enumerate(rows(BASE + p)):
            for r, rd in enumerate(t["rounds"]):
                for i, x in enumerate(rd):
                    o, b = r1(original(x)), r1(robust(x))
                    cat = ("both_none" if o is None and b is None else "recovered" if o is None
                           else "agree" if o == b else "disagree")
                    pool[cat].append({"id": f"{run}|{ti}|{r}|{i}", "category": cat, "original": o,
                                      "robust": b, "label": t["label"], "text": x})
    rng = random.Random(20261119)
    sample = []
    for cat, n in QUOTA.items():
        sample += rng.sample(pool[cat], min(n, len(pool[cat])))
    rng.shuffle(sample)
    with open(out / "audit_jobs.jsonl", "w") as f:   # what the extractor sees: no label, no category
        for s in sample:
            f.write(json.dumps({"id": s["id"], "messages": [{"role": "system", "content": SYSTEM},
                                {"role": "user", "content": "Response:\n" + s["text"]}],
                                "seed": 0}) + "\n")
    with open(out / "audit_key_hidden.jsonl", "w") as f:
        for s in sample:
            f.write(json.dumps({k: s[k] for k in ("id", "category", "original", "robust", "label")}) + "\n")
    print({k: len(v) for k, v in pool.items()}, "sampled", len(sample))

if __name__ == "__main__":
    main(sys.argv[1])
