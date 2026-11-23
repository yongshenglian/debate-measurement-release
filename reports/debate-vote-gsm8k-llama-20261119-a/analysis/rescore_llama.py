"""Re-score the Llama-3.1-8B Debate-or-Vote replication (197 completed problems) under four extractors.

Exploratory: the 'echo' variants were defined after seeing that Llama quotes the prompt's
format example '{final answer: 123}' back, which the Qwen-validated robust extractor reads as an answer.
"""
import re, random, sys
sys.path.insert(0, "/home/lian/swarmworld/research/reports/peer-theory-20261119")
sys.path.insert(0, "/home/lian/swarmworld/research/collaboration/parser_audit")
from rescore import rows, r1, vote
from empties import original, robust
import natural

P = "/home/lian/swarmworld/research/reports/debate-vote-gsm8k-llama-20261119-a/dense/out/history/gsm8k_300__llama3.1-8b_N=5_R=5.jsonl"
ECHO = re.compile(r"""["'“]\s*\{\s*final answer:\s*123\s*\}\s*["'”]""", re.I)
EXT = {"published": original, "published+echo": lambda x: original(ECHO.sub(" ", x)),
       "robust(Qwen)": robust, "robust+echo": lambda x: robust(ECHO.sub(" ", x))}

T = rows(P); n = len(T)
print(f"n problems = {n}; 5 agents x 6 rounds; bootstrap 4000 draws, seed 20261117, 95% CI")
for name, ext in EXT.items():
    v0 = None
    for r in range(6):
        per = []; ind = emp = 0
        for t in T:
            a = [r1(ext(x)) for x in t["rounds"][r]]
            ind += sum(x == t["label"] for x in a); emp += sum(x is None for x in a); per.append(vote(a, t["label"]))
        if r == 0: v0 = per
        d = [p - q for p, q in zip(per, v0)]; rng = random.Random(20261117)
        bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(4000))
        print(f"{name:15s} r{r} individual {ind/(5*n):.3f} empty {emp:3d} vote {sum(per)/n:.3f}"
              + ("" if r == 0 else f"  vote-r0 {sum(d)/n:+.3f} [{bs[100]:+.3f}, {bs[3900]:+.3f}]"))
print("\npattern % of agent outputs by round")
keys = ["escaped_delims", "loop_5plus", "two_distinct_answers", "no_answer_marker", "braces_after_answer"]
for r in range(6):
    xs = [x for t in T for x in t["rounds"][r]]
    c = {k: 100 * sum(natural.classify(x)[k] for x in xs) / len(xs) for k in keys}
    echo = 100 * sum(bool(ECHO.search(x)) for x in xs) / len(xs)
    ph = 100 * sum(bool(re.search(r"\{\s*final answer:\s*123\s*\}", ECHO.sub(" ", x), re.I)) for x in xs) / len(xs)
    print(f"r{r} " + " ".join(f"{k} {v:.1f}" for k, v in c.items()) + f" quoted_example {echo:.1f} placeholder_as_answer {ph:.1f}")
