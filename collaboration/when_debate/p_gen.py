"""Study P (reports/private-revision-20261123-a/PROTOCOL.md): private revision on the disagreement sets of studies K and L.
The judge sees only its own frozen first solution (sample 0) under the harness's self-review prompt (reverse_gen.SELF, MCQ task line
via m_reverse), with the same seeds (67000000 + 10*tid + rep, rep 0-2), token limits and two-turn JSON answer as the peer-review calls.
Usage: p_gen.py <shard 0-3> <tasks.jsonl> <peer_solve.jsonl> <judge_solve.jsonl> <out.jsonl>
Env: CTK, MODEL, LLM_KW as l_gen.py; LIMIT=n (smoke only)."""
import json, os, sys, time
import k_gen as kg
from k_gen import rg
from l_gen import run

assert rg.SELF.startswith("Answer the following multiple-choice question step by step.")
assert rg.SELF.endswith("\n\nYour earlier solution:\n{own}\n\nReview your earlier solution and give your final answer.")


def main():
    shard, tpath, ppath, jpath, out = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    lim = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
    tasks = [json.loads(l) for l in open(tpath)]
    sol = {}
    for m, p in (("peer", ppath), ("judge", jpath)):
        for l in open(p):
            d = json.loads(l)
            if d["sample"] == 0:
                sol[(m, d["tid"])] = d if d["parsed"] is not None else None
    D = [t for t in tasks if t["tid"] % 4 == shard and sol.get(("judge", t["tid"])) and sol.get(("peer", t["tid"]))
         and sol[("judge", t["tid"])]["parsed"]["answer"] != sol[("peer", t["tid"])]["parsed"]["answer"]][:lim]
    jobs = [dict(tid=t["tid"], cond="self", rep=r, seed=kg.JUDGE_SEED + 10 * t["tid"] + r,
                 prompt=rg.SELF.format(q=t["question"], own=rg.show(sol[("judge", t["tid"])])))
            for t in D for r in kg.REPS]
    llm = rg.LLM(model=os.environ.get("MODEL", "/model"), dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.88, seed=0,
                 structured_outputs_config={"backend": "guidance", "disable_any_whitespace": True},
                 **json.loads(os.environ.get("LLM_KW", "{}")))
    t0 = time.time()
    CH = 960
    with open(out, "x") as f:
        for i in range(0, len(jobs), CH):
            for j in run(llm, jobs[i:i + CH]):
                f.write(json.dumps(j) + "\n")
            f.flush()
    print(json.dumps({"stage": "self", "shard": shard, "jobs": len(jobs), "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
