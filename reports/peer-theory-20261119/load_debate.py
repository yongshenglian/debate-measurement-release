"""Load the dense/sparse Qwen2.5-7B debate histories into per-task answer lists by round."""
import json, math
BASE = "/home/lian/swarmworld/research/reports/debate-vote-gsm8k-20261117-a"

def conv(a):
    if a is None or a == "":
        return None
    try:
        x = float(a)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def load(arm):
    path = f"{BASE}/{arm}/out/history/gsm8k_300__qwen2.5-7b_N=5_R=5{"_SPARSE" if arm == "sparse" else ""}.jsonl"
    tasks = []
    for line in open(path):
        d = json.loads(line)
        label = None
        rounds = []
        for r in sorted(d, key=int):
            v = d[r]
            label = v["answer"]
            ans = [conv(a) for a in v["final_answers"]]
            rounds.append(ans)
        tasks.append({"label": float(label), "rounds": rounds})
    return tasks
