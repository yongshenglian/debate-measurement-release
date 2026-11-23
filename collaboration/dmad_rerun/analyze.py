"""Study B analysis (pre-registered in reports/dmad-rerun-20261120-a/PROTOCOL.md)."""
import json, re, sys, random, glob, os
from collections import Counter
sys.path.insert(0, "/home/lian/swarmworld/research/source/related-work/DMAD")
from utils.eval_utils import check_answer_correct, get_majority_answer, clean_answer
sys.path.insert(0, "/home/lian/swarmworld/research/collaboration/du_rerun")
from score_du import boxed_last

NUM = r"-?[0-9][0-9,]*(?:\.[0-9]+)?"
B, SEED = 20000, 20261117


def validated(completion):
    """The agent's own committed answer: the last <answer>...</answer> containing exactly one number
    (cleaned of $ , units); else the last \\boxed{N}; else 'final answer is N'; else None."""
    for a in reversed(re.findall(r"<answer>(.*?)</answer>", completion, flags=re.S)):
        s = re.sub(r"\\text\{[^}]*\}", "", a).replace("$", "").replace("\\", "")
        nums = re.findall(NUM, s)
        if len(nums) == 1:
            return float(nums[0].replace(",", ""))
        if len(nums) > 1:
            return None   # ambiguous: left to the blind audit
    b = boxed_last(completion)
    if b is not None:
        return b
    m = re.findall(r"final answer(?: is)?\s*[:：]?\s*\$?(" + NUM + ")", completion, flags=re.I)
    return float(m[-1].replace(",", "")) if m else None


def vote(vals, y):
    v = [x for x in vals if x is not None]
    if not v: return 0.0
    c = Counter(v); top = max(c.values()); w = [k for k, n in c.items() if n == top]
    return (y in w) / len(w)


def boot(d):
    n = len(d); rng = random.Random(SEED)
    s = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(d) / n, s[int(.025 * B)], s[int(.975 * B) - 1]


def load(path):
    R = json.load(open(path))["results"]
    out = []
    for ex in R:
        y = float(clean_answer(str(ex["label"])))
        rounds = []
        for rr in ex["round_results"]:
            ag = rr["agents_responses"]
            rounds.append({
                "dmad_vote": float(check_answer_correct(rr["majority_answer"], ex["label"], ex["dataset"])),
                "dmad_ind": [check_answer_correct(a["final_answer"], ex["label"], ex["dataset"]) for a in ag],
                "parse_err": [a["final_answer"] == "PARSE_ERROR" for a in ag],
                "val": [validated(a["completion"]) for a in ag],
                "texts": [a["completion"] for a in ag],
                "dmad_ans": [clean_answer(a["final_answer"]) for a in ag]})
        for r in rounds:
            r["val_vote"] = vote(r["val"], y); r["val_ind"] = [v == y for v in r["val"]]
        out.append({"y": y, "rounds": rounds})
    return out


def summarize(T, name):
    R = len(T[0]["rounds"]); n = len(T); res = {"n": n}
    for sc in ("dmad", "val"):
        rows = []
        for r in range(R):
            v = [t["rounds"][r][f"{sc}_vote"] for t in T]
            ind = sum(sum(t["rounds"][r][f"{sc}_ind"]) for t in T) / (5 * n)
            rows.append({"round": r, "vote": sum(v) / n, "individual": ind})
        d = [t["rounds"][-1][f"{sc}_vote"] - t["rounds"][0][f"{sc}_vote"] for t in T]
        res[sc] = {"by_round": rows, "P1_last_minus_r0": boot(d)}
    res["parse_error_rate"] = [sum(sum(t["rounds"][r]["parse_err"]) for t in T) / (5 * n) for r in range(R)]
    r0 = [t["rounds"][0] for t in T]
    res["r0_identical_texts_pct"] = 100 * sum(len(set(x["texts"])) == 1 for x in r0) / n
    res["r0_mean_distinct_texts"] = sum(len(set(x["texts"])) for x in r0) / n
    res["r0_mean_unique_answers_dmad"] = sum(len(set(a for a in x["dmad_ans"] if a != "[invalid]")) for x in r0) / n
    res["r0_mean_unique_answers_val"] = sum(len(set(a for a in x["val"] if a is not None)) for x in r0) / n
    return res


if __name__ == "__main__":
    d = sys.argv[1]
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        name = os.path.basename(p)[:-5]
        out[name] = summarize(load(p), name)
        o = out[name]
        print(f"== {name} n={o['n']}  r0 identical texts {o['r0_identical_texts_pct']:.1f}%  distinct texts {o['r0_mean_distinct_texts']:.2f}"
              f"  unique answers dmad {o['r0_mean_unique_answers_dmad']:.2f} val {o['r0_mean_unique_answers_val']:.2f}")
        print("   parse errors by round", [round(100 * x, 1) for x in o["parse_error_rate"]])
        for sc in ("dmad", "val"):
            br = o[sc]["by_round"]; p1 = o[sc]["P1_last_minus_r0"]
            print(f"   {sc:4s} vote " + " ".join(f"{x['vote']:.3f}" for x in br) + "  ind " + " ".join(f"{x['individual']:.3f}" for x in br)
                  + f"  last-r0 {100*p1[0]:+.1f} [{100*p1[1]:+.1f}, {100*p1[2]:+.1f}]")
    json.dump(out, open(os.path.join(d, "..", f"analysis_{os.path.basename(d.rstrip('/'))}.json"), "w"), indent=1)
