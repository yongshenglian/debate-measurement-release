"""Pre-registered reader-validity audit for study S (PROTOCOL.md, "Answer reading").
  prepare <out dir> <jobs.jsonl> : 100 random outputs per arm (50 round 0, 50 final round), seed 20261122, response text only
  score   <out dir> <jobs.jsonl> <gemma_out.jsonl> : agreement of our reader with blind Gemma, per arm
Gemma is run with blind-extraction/generate.py in chat mode (greedy, 64 tokens)."""
import json, random, re, sys, glob, os
sys.path.insert(0, "/home/lian/swarmworld/research/reports/format-contagion-20261119-a")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fc_audit_prepare import SYSTEM
from settings_analyze import reader, r1


def items(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "S*.json"))):
        D = json.load(open(p)); K = len(D["results"][0]["rounds"]) - 1
        rng = random.Random(f"20261122|{D['arm']}")
        for rnd in (0, K):
            cells = [(qi, a) for qi in range(len(D["results"])) for a in range(5)]
            for qi, a in rng.sample(cells, 50):
                out.append({"id": f"{D['arm']}|{rnd}|{qi}|{a}", "family": D["cfg"]["family"],
                            "text": D["results"][qi]["rounds"][rnd][a]})
    return out


def gemma_val(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        a = json.loads(m.group(0))["answer"]
        if isinstance(a, str): a = float(a.replace(",", "").replace("$", ""))
        return None if a is None else r1(float(a))
    except Exception:
        return None


if sys.argv[1] == "prepare":
    with open(sys.argv[3], "x") as f:
        for it in items(sys.argv[2]):
            f.write(json.dumps({"id": it["id"], "seed": 0, "messages": [{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": "Response:\n" + it["text"]}]}) + "\n")
else:
    its = {it["id"]: it for it in items(sys.argv[2])}
    G = {json.loads(l)["id"]: gemma_val(json.loads(l)["text"]) for l in open(sys.argv[4])}
    arms = sorted({i.split("|")[0] for i in its})
    for arm in arms:
        ids = [i for i in its if i.startswith(arm + "|")]
        ours = {i: reader(its[i]["family"])(its[i]["text"]) for i in ids}
        rec = [i for i in ids if ours[i] is not None]
        agree = sum(ours[i] == G[i] for i in rec)
        none_mis = sum(ours[i] is None and G[i] is not None for i in ids)
        verdict = "PASS" if rec and agree / len(rec) >= 0.95 else "FAIL"
        print(f"{arm:18s} agreement on recovered {agree}/{len(rec)} = {100*agree/max(1,len(rec)):.1f}%  {verdict}"
              f"  | our reader none but Gemma reads a number: {none_mis}/{len(ids)}")
