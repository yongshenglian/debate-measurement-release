"""Classify empty extractions in the debate run and re-score with a robust extractor."""
import json, re, math
from collections import Counter
BASE = "/home/lian/swarmworld/research/reports/debate-vote-gsm8k-20261117-a"
FILES = {"dense": "dense/out/history/gsm8k_300__qwen2.5-7b_N=5_R=5.jsonl",
         "sparse": "sparse/out/history/gsm8k_300__qwen2.5-7b_N=5_R=5_SPARSE.jsonl"}
NUM = r"-?\$?\s*-?[0-9][0-9,]*(?:\.[0-9]+)?"

def original(resp):
    """The published extractor: last {...} on one line, minus 'final answer:', as a float."""
    try:
        pred = re.findall(r"\{(.*?)\}", resp)[-1]
        return float(pred.replace("final answer:", "").strip())
    except Exception:
        return None

def to_float(s):
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None

def robust(resp):
    """Last 'final answer: <number>' anywhere (across newlines / escaped braces); else last \\boxed{number}."""
    # Prefer the last answer closed by a brace (complete); a generation cut at the token limit
    # can end mid-number ("final answer: 12" for 120).
    closed = re.findall(r"final answer\s*[:：]?\s*\\?\$?\s*(" + NUM + r")\s*\\?\$?\s*\\?\}", resp, flags=re.I | re.S)
    if closed:
        return to_float(closed[-1])
    m = re.findall(r"final answer\s*[:：]?\s*\\?\$?\s*(" + NUM + ")", resp, flags=re.I | re.S)
    if m:
        return to_float(m[-1])
    m = re.findall(r"\\boxed\{\s*(" + NUM + r")\s*\}", resp)
    if m:
        return to_float(m[-1])
    return original(resp)   # fall back to the published extractor (e.g. a bare "{19}")

def classify(resp):
    if re.search(r"final answer", resp, re.I):
        return "has 'final answer' text" if robust(resp) is not None else "final-answer text, no number"
    if re.search(r"\\boxed\{", resp):
        return "boxed only"
    return "no answer marker"

def load(arm):
    rows = []
    for line in open(f"{BASE}/{FILES[arm]}"):
        d = json.loads(line)
        rounds = []
        for r in sorted(d, key=int):
            v = d[r]
            resps = list(v["responses"].values())
            rounds.append({"resp": resps, "logged": v["final_answers"]})
        rows.append({"label": float(d["0"]["answer"]), "rounds": rounds})
    return rows

if __name__ == "__main__":
    for arm in FILES:
        T = load(arm)
        # The original extractor must reproduce the logged answers.
        mism = sum((original(rs) is None) != (lg in ("", None) or (isinstance(lg, float) and math.isnan(lg)))
                   for t in T for rd in t["rounds"] for rs, lg in zip(rd["resp"], rd["logged"]))
        print(f"== {arm}: original-extractor mismatches vs logged answers: {mism}")
        for r in range(6):
            kinds = Counter()
            rec_c = rec_w = 0
            o_c = r_c = r_none = 0
            vote_o = vote_r = 0
            for t in T:
                y = t["label"]
                resps = t["rounds"][r]["resp"]
                o = [original(x) for x in resps]
                rb = [robust(x) for x in resps]
                for x, a, b in zip(resps, o, rb):
                    if a is None:
                        kinds[classify(x)] += 1
                        if b is not None:
                            rec_c += b == y
                            rec_w += b != y
                o_c += sum(a == y for a in o)
                r_c += sum(b == y for b in rb)
                r_none += sum(b is None for b in rb)
                for ans, key in ((o, "o"), (rb, "r")):
                    ne = [a for a in ans if a is not None]
                    if ne:
                        cnt = Counter(ne); top = max(cnt.values())
                        win = [k for k, v in cnt.items() if v == top]
                        ok = (y in win) / len(win)   # expected accuracy under random tie-break
                        if key == "o": vote_o += ok
                        else: vote_r += ok
            print(f"round {r}: original correct {o_c}/1500 vote {vote_o/300:.3f} | robust correct {r_c}/1500, "
                  f"still empty {r_none}, vote {vote_r/300:.3f} | empties recovered: {rec_c} correct, {rec_w} wrong | {dict(kinds)}")
