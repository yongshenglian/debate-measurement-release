"""POST HOC sensitivity (not pre-registered; written 2026-11-20 after seeing the official-template Llama arms).
Llama often ends its turn right after '<answer>\\nN' without '</answer>'. Both DMAD's parser and our pre-registered
`validated` need the closing tag. Here: if `validated` gives None and the text after the LAST '<answer>' has no
closing tag and contains exactly one number, take that number. Everything else as analyze.py."""
import re, sys, glob, os
sys.argv += [] ; sys.path.insert(0, os.path.dirname(__file__))
from analyze import validated, vote, boot, NUM
import json
from utils.eval_utils import clean_answer

def unclosed(c):
    v = validated(c)
    if v is not None: return v, False
    if "<answer>" not in c: return None, False
    tail = c.split("<answer>")[-1]
    if "</answer>" in tail: return None, False
    s = re.sub(r"\\text\{[^}]*\}", "", tail).replace("$", "").replace("\\", "")
    nums = re.findall(NUM, s)
    return (float(nums[0].replace(",", "")), True) if len(nums) == 1 else (None, False)

for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    R = json.load(open(p))["results"]; name = os.path.basename(p)[:-5]
    votes, inds, rec = [], [], []
    for ex in R:
        y = float(clean_answer(str(ex["label"])))
        vr, ir, rr = [], [], []
        for r in ex["round_results"]:
            got = [unclosed(a["completion"]) for a in r["agents_responses"]]
            vals = [g[0] for g in got]
            vr.append(vote(vals, y)); ir.append(sum(v == y for v in vals) / len(vals)); rr.append(sum(g[1] for g in got) / len(got))
        votes.append(vr); inds.append(ir); rec.append(rr)
    n = len(R); K = len(votes[0])
    m, lo, hi = boot([v[-1] - v[0] for v in votes])
    print(f"== {name}: recovered-by-unclosed-rule % by round " + " ".join(f"{100*sum(x[k] for x in rec)/n:.1f}" for k in range(K)))
    print("   vote " + " ".join(f"{sum(v[k] for v in votes)/n:.3f}" for k in range(K)) + "  ind " +
          " ".join(f"{sum(v[k] for v in inds)/n:.3f}" for k in range(K)) + f"  last-r0 {100*m:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]")
