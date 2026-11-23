"""Study J (reports/neutral-judge-20261121-a/PROTOCOL.md): does the "your earlier solution" framing cost the reader?
On problems where Gemma's and Qwen's first solutions (sample 0) disagree, the judge sees the same two solutions under four framings:
  owner        "Your earlier solution:" (judge's own) then "Another solver's solution:" (other model's)  = the study R/H/M PEER prompt
  neutral      "Solution A:" (judge's own) then "Solution B:" (other's), same order, labels only
  neutral_swap "Solution A:" (other's) then "Solution B:" (judge's own)
  label_swap   "Your earlier solution:" holding the OTHER model's solution, "Another solver's solution:" holding the judge's own
Harness, decoding, seeds (47000000 + tid, shared by conditions) and token limits as the original review runs:
  gsm8k -> reverse_gen.py (1,024), math -> hard_reverse.py (2,048), mmlu -> m_reverse.py (2,048, multiple-choice task line).
Usage: j_gen.py <gsm8k|math|mmlu> <judge qwen|gemma> <tasks.jsonl> <qwen_solve.jsonl> <gemma_solve.jsonl> <out.jsonl>"""
import json, os, sys, time

taskset, judge = sys.argv[1], sys.argv[2]
assert os.environ.get("INDEP") != "1"
if taskset == "math":
    import hard_reverse  # noqa: F401
elif taskset == "mmlu":
    import m_reverse  # noqa: F401
else:
    assert taskset == "gsm8k"
import reverse_gen as rg

REQ = "\n\nReview both solutions and give your final answer."
assert rg.PEER == rg.TASK + "\n\nYour earlier solution:\n{own}\n\nAnother solver's solution:\n{peer}" + REQ
FRAMES = {
    "owner": rg.PEER,
    "neutral": rg.TASK + "\n\nSolution A:\n{own}\n\nSolution B:\n{peer}" + REQ,
    "neutral_swap": rg.TASK + "\n\nSolution A:\n{peer}\n\nSolution B:\n{own}" + REQ,
    "label_swap": rg.TASK + "\n\nYour earlier solution:\n{peer}\n\nAnother solver's solution:\n{own}" + REQ,
}


def main():
    tasks = [json.loads(l) for l in open(sys.argv[3])]
    sol = {}
    for m, p in (("qwen", sys.argv[4]), ("gemma", sys.argv[5])):
        for l in open(p):
            d = json.loads(l)
            if d["sample"] == 0:
                sol[(m, d["tid"])] = d if d["parsed"] is not None else None
    other = "gemma" if judge == "qwen" else "qwen"
    jobs = []
    for t in tasks:
        own, peer = sol.get((judge, t["tid"])), sol.get((other, t["tid"]))
        if own is None or peer is None or own["parsed"]["answer"] == peer["parsed"]["answer"]:
            continue
        for c, tpl in FRAMES.items():
            jobs.append(dict(tid=t["tid"], cond=c, seed=rg.SEEDS["review"] + t["tid"],
                             prompt=tpl.format(q=t["question"], own=rg.show(own), peer=rg.show(peer))))
    if os.environ.get("LIMIT"):   # smoke runs only
        jobs = jobs[:4 * int(os.environ["LIMIT"])]
    llm = rg.LLM(model="/model", dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.88, seed=0,
                 structured_outputs_config={"backend": "guidance", "disable_any_whitespace": True})
    t0 = time.time()
    with open(sys.argv[6], "x") as f:
        for j in rg.run(llm, jobs):
            f.write(json.dumps(j) + "\n")
    print(json.dumps({"stage": "judge", "taskset": taskset, "model": judge, "jobs": len(jobs), "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
