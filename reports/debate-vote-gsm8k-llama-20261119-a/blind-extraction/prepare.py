import json, random, sys
sys.path.insert(0, "/home/lian/swarmworld/research/reports/peer-theory-20261119")
sys.path.insert(0, "/home/lian/swarmworld/research/reports/format-contagion-20261119-a")
from rescore import rows
from fc_audit_prepare import SYSTEM
P = "/home/lian/swarmworld/research/reports/debate-vote-gsm8k-llama-20261119-a/dense/out/history/gsm8k_300__llama3.1-8b_N=5_R=5.jsonl"
T = rows(P)
jobs = [{"id": f"{ti}|{r}|{i}", "messages": [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": "Response:\n" + x}], "seed": 0}
        for ti, t in enumerate(T) for r, rd in enumerate(t["rounds"]) for i, x in enumerate(rd)]
random.Random(20261120).shuffle(jobs)
with open("jobs.jsonl", "x") as f:
    for j in jobs: f.write(json.dumps(j) + "\n")
print(len(jobs))
