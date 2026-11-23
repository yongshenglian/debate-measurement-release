"""How often each failure-inducing pattern occurs in real debate outputs, by round."""
import json, re, sys
from collections import Counter
sys.path.insert(0, "/home/lian/swarmworld/research/reports/peer-theory-20261119")
from rescore import rows  # noqa: E402

PATTERNS = {
    "escaped_delims": re.compile(r"\\\{\s*final answer", re.I),
    "dollar_in_answer": re.compile(r"\{\s*final answer:\s*\\?\$", re.I),
    "commas_in_answer": re.compile(r"\{\s*final answer:\s*\$?\d{1,3}(,\d{3})+", re.I),
    "latex_in_answer": re.compile(r"\{\s*final answer:[^}]*\\(text|frac|boxed)", re.I),
    "newline_inside": re.compile(r"\{\s*\n\s*final answer", re.I),
    "braces_after_answer": re.compile(r"\{\s*final answer:[^}]*\}[^{]*\{[^}]*[a-zA-Z\\][^}]*\}\s*\S*\s*$", re.I | re.S),
    "loop_5plus": re.compile(r"(final answer[^\n]{0,40}){5,}", re.I | re.S),
    "two_distinct_answers": None,
    "no_answer_marker": None,
}
FA = re.compile(r"final answer\s*:?\s*\\?\$?\s*(-?[\d,]+(?:\.\d+)?)", re.I)

def classify(t):
    out = {k: bool(p.search(t)) for k, p in PATTERNS.items() if p is not None}
    vals = {v.replace(",", "") for v in FA.findall(t)}
    out["two_distinct_answers"] = len(vals) >= 2
    out["no_answer_marker"] = not re.search(r"final answer", t, re.I)
    return out

def main():
    base = "/home/lian/swarmworld/research/reports/"
    runs = {"gsm_dense": "debate-vote-gsm8k-20261117-a/dense/out/history/gsm8k_300__qwen2.5-7b_N=5_R=5.jsonl",
            "gsm_sparse": "debate-vote-gsm8k-20261117-a/sparse/out/history/gsm8k_300__qwen2.5-7b_N=5_R=5_SPARSE.jsonl"}
    res = {}
    for run, p in runs.items():
        T = rows(base + p)
        res[run] = {}
        for r in range(len(T[0]["rounds"])):
            c = Counter(); n = 0
            for t in T:
                for x in t["rounds"][r]:
                    n += 1
                    for k, v in classify(x).items():
                        c[k] += v
            res[run][r] = {k: round(100 * v / n, 1) for k, v in c.items()}
        print("==", run, "(% of agent outputs)")
        keys = list(res[run][0])
        print(f"{'pattern':22s}" + "".join(f"  r{r:<5d}" for r in res[run]))
        for k in keys:
            print(f"{k:22s}" + "".join(f"  {res[run][r][k]:<6.1f}" for r in res[run]))
    json.dump(res, open(sys.argv[1], "w"), indent=1)

if __name__ == "__main__":
    main()
