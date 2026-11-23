"""POST HOC sensitivity (written after seeing analysis.out): validated extraction + the authors' vote rule
(most_frequent: first answer reaching the max count wins; agents ordered llm1 first, parse failures dropped).
Separates extraction effects from the tie-breaking rule."""
import sys
sys.path.insert(0, "/c")
from analyze_wynn import load, boxed_last, boot_strat
from src.analysis.evaluation import most_frequent

def vote_authors_rule(vals, y):
    v = [x for x in vals if x is not None]
    return float(bool(v) and most_frequent(v) == y)

for cfg in sys.argv[2:]:
    rows = load(f"{sys.argv[1]}/{cfg}")
    R = len(rows[0]["rounds"])
    votes = [[vote_authors_rule([boxed_last(x) for x in t["rounds"][r]], t["y"]) for t in rows] for r in range(R)]
    ties = [sum(len(set(v for v in (boxed_last(x) for x in t["rounds"][r]) if v is not None)) == 3 for t in rows) / len(rows) for r in range(R)]
    e = boot_strat([a - b for a, b in zip(votes[-1], votes[0])], [t["seed"] for t in rows])
    print(f"{cfg}: validated extraction + authors' tie rule: vote " + " ".join(f"{100*sum(v)/len(rows):.1f}" for v in votes)
          + f"  after-before {e[0]:+.1f} [{e[1]:+.1f}, {e[2]:+.1f}]   3-way-distinct share by round " + " ".join(f"{100*x:.1f}%" for x in ties))
