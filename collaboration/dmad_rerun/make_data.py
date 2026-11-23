"""GSM8K test CSV for DMAD, via DMAD's own loader (load_gsm8k(split='test', data_size=300, seed=0), as in its README)."""
import sys, csv
sys.path.insert(0, "/dmad")
from data.datasets import load_gsm8k
q, l = load_gsm8k(split="test", data_size=300, cache_dir="/hfcache")
with open(sys.argv[1], "w", newline="") as f:
    w = csv.writer(f); w.writerow(["fold", "dataset", "question", "label"])
    for a, b in zip(q, l): w.writerow(["test", "gsm8k", a, b])
print(len(q))
