"""Pre-registered analysis of the blind Gemma extraction of Llama Debate-or-Vote outputs (PROTOCOL.md).
Step 'spot': print 30 random Gemma-vs-published disagreements WITHOUT labels, for a blind hand check.
Step 'main': vote and individual accuracy by round under Gemma readings, plus agreement with each regex extractor."""
import json, re, sys, random
sys.path.insert(0, "/home/lian/swarmworld/research/reports/peer-theory-20261119")
from rescore import rows, r1, vote
from empties import original, robust
ECHO = re.compile(r"""["'“]\s*\{\s*final answer:\s*123\s*\}\s*["'”]""", re.I)
EXT = {"published": original, "published+echo": lambda x: original(ECHO.sub(" ", x)),
       "robust": robust, "robust+echo": lambda x: robust(ECHO.sub(" ", x))}
P = "/home/lian/swarmworld/research/reports/debate-vote-gsm8k-llama-20261119-a/dense/out/history/gsm8k_300__llama3.1-8b_N=5_R=5.jsonl"
T = rows(P)

def gemma_val(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        a = json.loads(m.group(0))["answer"]
        if isinstance(a, str): a = float(a.replace(",", "").replace("$", ""))
        return None if a is None else r1(float(a))
    except Exception:
        return None

G = {}
for l in open("out/out.jsonl"):
    d = json.loads(l); G[d["id"]] = gemma_val(d["text"])

def gem(ti, r, i): return G[f"{ti}|{r}|{i}"]

if sys.argv[1] == "spot":
    dis = [(ti, r, i) for ti, t in enumerate(T) for r in range(6) for i in range(5)
           if gem(ti, r, i) != r1(original(t["rounds"][r][i]))]
    print("disagreements:", len(dis), "of", len(G))
    for k, (ti, r, i) in enumerate(random.Random(20261120).sample(dis, 30)):
        x = T[ti]["rounds"][r][i]
        print(f"\n##### item {k}  gemma={gem(ti,r,i)}  published={r1(original(x))}  len={len(x)}")
        print(x[-900:])
else:
    B = 20000; n = len(T)
    res = {}
    for r in range(6):
        res[r] = {"vote": [vote([gem(ti, r, i) for i in range(5)], t["label"]) for ti, t in enumerate(T)],
                  "ind": sum(gem(ti, r, i) == t["label"] for ti, t in enumerate(T) for i in range(5)) / (5 * n),
                  "none": sum(gem(ti, r, i) is None for ti in range(n) for i in range(5))}
        print(f"gemma r{r}: vote {sum(res[r]['vote'])/n:.3f} individual {res[r]['ind']:.3f} no-answer {res[r]['none']}")
    d = [a - b for a, b in zip(res[5]["vote"], res[0]["vote"])]
    rng = random.Random(20261117)
    s = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    print(f"PRIMARY vote r5 - r0 (Gemma readings): {100*sum(d)/n:+.1f} pp [{100*s[500]:+.1f}, {100*s[19499]:+.1f}]")
    for name, ext in EXT.items():
        ag = [sum(r1(ext(T[ti]["rounds"][r][i])) == gem(ti, r, i) for ti in range(n) for i in range(5)) / (5 * n) for r in range(6)]
        print(f"agreement with {name:15s} by round: " + " ".join(f"{100*a:.1f}" for a in ag))
