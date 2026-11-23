"""Study K (reports/framing-rep-20261122-a/PROTOCOL.md): pre-registered replication of the study J framing effect on untouched
MMLU-Pro problems, with one label cue changed at a time and three decoding seeds per problem.

Harness as study M (m_reverse.py: multiple-choice task line, 2,048-token free text, two-turn JSON answer, T 0.7, top-p 0.95,
top-k 20). The run function below is reverse_gen.run plus prompt-token counts (for equal-call vs token accounting).

Stages:
  solve <model> <samples, e.g. 0,1,2> <tasks.jsonl> <out.jsonl>      seed 61000000 + 10*tid + sample
  judge <shard 0-3> <tasks.jsonl> <qwen_solve.jsonl> <gemma_solve.jsonl> <out.jsonl>
        judge = Gemma. Disagreement set D: Gemma s0 and Qwen s0 both valid and different. Problems with tid % 4 == shard.
        Each problem in D: 8 framings x 3 replicates, seed 67000000 + 10*tid + rep, shared by the 8 framings.
LIMIT=n (smoke only): first n tasks (solve) or first n problems of D in the shard (judge)."""
import json, os, sys, time
import m_reverse  # noqa: F401  (patches reverse_gen: MCQ task line, 2,048 tokens)
import reverse_gen as rg

REQ = "\n\nReview both solutions and give your final answer."
assert rg.PEER == rg.TASK + "\n\nYour earlier solution:\n{own}\n\nAnother solver's solution:\n{peer}" + REQ
assert rg.TASK.startswith("Answer the following multiple-choice question step by step.")
YE, AN, YS = "Your earlier solution:", "Another solver's solution:", "Your solution:"
A, B = "Solution A:", "Solution B:"
# (first label, first content, second label, second content); own = judge's sample 0, oth = Qwen's sample 0
FRAMES = {
    "owner":       (YE, "own", AN, "oth"),   # the study R/H/M/J prompt
    "neutral":     (A,  "own", B,  "oth"),   # K1 contrast: all labels
    "yours":       (YE, "own", B,  "oth"),   # K2: own-block label only
    "another":     (A,  "own", AN, "oth"),   # K3: other-block label only
    "yoursol":     (YS, "own", B,  "oth"),   # K4: "earlier" removed from the own-block label
    "nswap":       (A,  "oth", B,  "own"),   # neutral, other model's solution first
    "misattr":     (YE, "oth", B,  "own"),   # K5: "Your earlier solution" on the other model's solution
    "owner_swap":  (AN, "oth", YE, "own"),   # owner labels, own solution second
}
REPS = (0, 1, 2)
SOLVE_SEED, JUDGE_SEED = 61000000, 67000000


def prompt(q, frame, own, oth):
    l1, c1, l2, c2 = FRAMES[frame]; c = {"own": own, "oth": oth}
    return rg.TASK.format(q=q) + f"\n\n{l1}\n{c[c1]}\n\n{l2}\n{c[c2]}" + REQ


def run(llm, jobs):
    free = [rg.SamplingParams(temperature=0.7, top_p=0.95, top_k=20, max_tokens=1024, seed=j["seed"]) for j in jobs]  # 2,048 via hard_reverse
    convs = [[{"role": "user", "content": j["prompt"]}] for j in jobs]
    outs = llm.chat(convs, free)
    convs2 = [c + [{"role": "assistant", "content": o.outputs[0].text}, {"role": "user", "content": rg.ASK}] for c, o in zip(convs, outs)]
    ansp = [rg.SamplingParams(temperature=0.7, top_p=0.95, top_k=20, max_tokens=16, seed=j["seed"] + 1,
                              structured_outputs=rg.StructuredOutputsParams(json=rg.SCHEMA)) for j in jobs]
    outs2 = llm.chat(convs2, ansp)
    for j, o, o2 in zip(jobs, outs, outs2):
        c = o.outputs[0]
        j.update(text=c.text, finish_reason=c.finish_reason, output_tokens=len(c.token_ids), prompt_tokens=len(o.prompt_token_ids),
                 prompt_tokens_2=len(o2.prompt_token_ids), output_tokens_2=len(o2.outputs[0].token_ids),
                 answer_text=o2.outputs[0].text, parsed=rg.parse(o2.outputs[0].text))
        del j["prompt"]
    return jobs


def main():
    stage = sys.argv[1]
    lim = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
    if stage == "solve":
        model, samples, tpath, out = sys.argv[2], [int(s) for s in sys.argv[3].split(",")], sys.argv[4], sys.argv[5]
        tasks = [json.loads(l) for l in open(tpath)][:lim]
        jobs = [{"tid": t["tid"], "sample": s, "model": model, "seed": SOLVE_SEED + 10 * t["tid"] + s, "prompt": rg.TASK.format(q=t["question"])}
                for t in tasks for s in samples]
    else:
        assert stage == "judge"
        shard, tpath, qpath, gpath, out = int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
        tasks = [json.loads(l) for l in open(tpath)]
        sol = {}
        for m, p in (("qwen", qpath), ("gemma", gpath)):
            for l in open(p):
                d = json.loads(l)
                if d["sample"] == 0:
                    sol[(m, d["tid"])] = d if d["parsed"] is not None else None
        D = [t for t in tasks if t["tid"] % 4 == shard and sol.get(("gemma", t["tid"])) and sol.get(("qwen", t["tid"]))
             and sol[("gemma", t["tid"])]["parsed"]["answer"] != sol[("qwen", t["tid"])]["parsed"]["answer"]][:lim]
        jobs = [dict(tid=t["tid"], cond=f, rep=r, seed=JUDGE_SEED + 10 * t["tid"] + r,
                     prompt=prompt(t["question"], f, rg.show(sol[("gemma", t["tid"])]), rg.show(sol[("qwen", t["tid"])])))
                for t in D for r in REPS for f in FRAMES]
    llm = rg.LLM(model="/model", dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.88, seed=0,
                 structured_outputs_config={"backend": "guidance", "disable_any_whitespace": True})
    t0 = time.time()
    CH = 960   # write in chunks so a timeout keeps finished work (whole problems: 24 judge jobs per problem)
    with open(out, "x") as f:
        for i in range(0, len(jobs), CH):
            for j in run(llm, jobs[i:i + CH]):
                f.write(json.dumps(j) + "\n")
            f.flush()
    print(json.dumps({"stage": stage, "args": sys.argv[2:4], "jobs": len(jobs), "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
