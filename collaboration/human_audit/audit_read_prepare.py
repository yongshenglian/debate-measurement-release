"""Human audit, stage A (reports/human-audit-20261122/PROTOCOL.md): blind Gemma-3-27B reading of EVERY debate output that
enters a vote in the paper's debate results (round 0 and the final round; all 20 samples of the 20-vote).
Same blind-reader prompt as the earlier audits (format-contagion fc_audit_prepare.SYSTEM): the reader sees only the response.
Writes jobs.jsonl for blind-extraction/generate.py chat, ids "<study>|<arm>|<round>|<problem>|<agent>".
Usage: audit_read_prepare.py <out jobs.jsonl>"""
import json, sys
sys.path.insert(0, "/home/lian/swarmworld/research/reports/format-contagion-20261119-a")
from fc_audit_prepare import SYSTEM

R = "/home/lian/swarmworld/research/reports"
SRC = {"debate-hard-20261120-a/out": ["H_full", "H_ans", "H_dovword", "H_vote20"],
       "debate-indep-20261121-a/out": ["H_full_indep", "H_ans_indep"],
       "mmlupro-20261121-a/out-debate": ["H_full", "H_ans", "H_full_indep", "H_ans_indep", "H_vote20"]}

n = 0
with open(sys.argv[1], "x") as f:
    for sub, arms in SRC.items():
        study = sub.split("/")[0]
        for arm in arms:
            D = json.load(open(f"{R}/{sub}/{arm}.json"))
            for qi, ex in enumerate(D["results"]):
                K = len(ex["rounds"]) - 1
                for k in sorted({0, K}):
                    for a, t in enumerate(ex["rounds"][k]):
                        f.write(json.dumps({"id": f"{study}|{arm}|{k}|{qi}|{a}", "seed": 0, "messages": [
                            {"role": "system", "content": SYSTEM}, {"role": "user", "content": "Response:\n" + t}]}) + "\n")
                        n += 1
print(n, "jobs")
