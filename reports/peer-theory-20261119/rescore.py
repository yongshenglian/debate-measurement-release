"""Re-score Debate-or-Vote replication histories with the original and a robust final-answer extractor."""
import json, sys, random
from collections import Counter
from empties import original, robust

def r1(x):
    return None if x is None else round(x, 1)

def rows(path):
    out = []
    for line in open(path):
        d = json.loads(line)
        out.append({"label": round(float(d["0"]["answer"]), 1),
                    "rounds": [list(d[r]["responses"].values()) for r in sorted(d, key=int)]})
    return out

def vote(ans, y):
    ne = [a for a in ans if a is not None]
    if not ne:
        return 0.0
    c = Counter(ne); top = max(c.values()); win = [k for k, v in c.items() if v == top]
    return (y in win) / len(win)

def summarize(path, label):
    T = rows(path)
    R = len(T[0]["rounds"])
    print(f"== {label} (n={len(T)})")
    v0 = {}
    for name, ext in (("original", original), ("robust", robust)):
        for r in range(R):
            per = []
            ind = emp = 0
            for t in T:
                a = [r1(ext(x)) for x in t["rounds"][r]]
                ind += sum(x == t["label"] for x in a); emp += sum(x is None for x in a)
                per.append(vote(a, t["label"]))
            if r == 0:
                v0[name] = per
            d = [p - q for p, q in zip(per, v0[name])]
            rng = random.Random(20261117); n = len(d)
            bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(4000))
            na = len(T[0]["rounds"][r])
            print(f"  {name:8s} round {r}: individual {ind/(n*na):.3f}  empty {emp:4d}  vote {sum(per)/n:.3f}"
                  + ("" if r == 0 else f"  vote − round0 {sum(d)/n:+.3f} [{bs[100]:+.3f}, {bs[3900]:+.3f}]"))

if __name__ == "__main__":
    base = "/home/lian/swarmworld/research/reports"
    for lab, p in [
        ("GSM8K dense", "debate-vote-gsm8k-20261117-a/dense/out/history/gsm8k_300__qwen2.5-7b_N=5_R=5.jsonl"),
        ("GSM8K sparse", "debate-vote-gsm8k-20261117-a/sparse/out/history/gsm8k_300__qwen2.5-7b_N=5_R=5_SPARSE.jsonl"),
        ("Arithmetic dense", "debate-vote-arithmetic-20261116-a/dense/out/history/arithmetics_100__qwen2.5-7b_N=5_R=5.jsonl"),
        ("Arithmetic sparse", "debate-vote-arithmetic-20261116-a/sparse/out/history/arithmetics_100__qwen2.5-7b_N=5_R=5_SPARSE.jsonl")]:
        summarize(f"{base}/{p}", lab)
