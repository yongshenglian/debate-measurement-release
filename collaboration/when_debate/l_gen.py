"""Study L (reports/framing-gen-20261122-a/PROTOCOL.md): study K with a different judge model. Same 1,500 MMLU-Pro problems,
same harness, framings, seeds and replicates as k_gen.py; the peer is Qwen2.5-7B sample 0 from study K (reused, not regenerated).
The only change is the judge, served with its chat template's thinking turned off (CTK below), so it is tested like Gemma.

Stages:
  solve <samples, e.g. 0,1,2> <tasks.jsonl> <out.jsonl>                       judge model's own solutions, seed 61000000 + 10*tid + sample
  judge <shard 0-3> <tasks.jsonl> <peer_solve.jsonl> <judge_solve.jsonl> <out.jsonl>
        D: judge s0 and peer s0 both valid and different; problems with tid % 4 == shard; 8 framings x 3 replicates.
Env: CTK = JSON chat_template_kwargs (e.g. {"enable_thinking": false}); MODEL = model path in the container (default /model);
     LLM_KW = extra JSON kwargs for vllm.LLM; LIMIT=n (smoke only)."""
import json, os, sys, time
import k_gen as kg            # FRAMES, REPS, seeds, prompt(); imports the study M harness patches
from k_gen import rg

CTK = json.loads(os.environ.get("CTK", "{}"))


def run(llm, jobs):
    """k_gen.run with chat_template_kwargs passed to both turns."""
    free = [rg.SamplingParams(temperature=0.7, top_p=0.95, top_k=20, max_tokens=1024, seed=j["seed"]) for j in jobs]  # 2,048 via hard_reverse
    convs = [[{"role": "user", "content": j["prompt"]}] for j in jobs]
    outs = llm.chat(convs, free, chat_template_kwargs=CTK)
    convs2 = [c + [{"role": "assistant", "content": o.outputs[0].text}, {"role": "user", "content": rg.ASK}] for c, o in zip(convs, outs)]
    ansp = [rg.SamplingParams(temperature=0.7, top_p=0.95, top_k=20, max_tokens=16, seed=j["seed"] + 1,
                              structured_outputs=rg.StructuredOutputsParams(json=rg.SCHEMA)) for j in jobs]
    outs2 = llm.chat(convs2, ansp, chat_template_kwargs=CTK)
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
        samples, tpath, out = [int(s) for s in sys.argv[2].split(",")], sys.argv[3], sys.argv[4]
        tasks = [json.loads(l) for l in open(tpath)][:lim]
        jobs = [{"tid": t["tid"], "sample": s, "model": "judge", "seed": kg.SOLVE_SEED + 10 * t["tid"] + s, "prompt": rg.TASK.format(q=t["question"])}
                for t in tasks for s in samples]
    else:
        assert stage == "judge"
        shard, tpath, ppath, jpath, out = int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
        tasks = [json.loads(l) for l in open(tpath)]
        sol = {}
        for m, p in (("peer", ppath), ("judge", jpath)):
            for l in open(p):
                d = json.loads(l)
                if d["sample"] == 0:
                    sol[(m, d["tid"])] = d if d["parsed"] is not None else None
        D = [t for t in tasks if t["tid"] % 4 == shard and sol.get(("judge", t["tid"])) and sol.get(("peer", t["tid"]))
             and sol[("judge", t["tid"])]["parsed"]["answer"] != sol[("peer", t["tid"])]["parsed"]["answer"]][:lim]
        jobs = [dict(tid=t["tid"], cond=f, rep=r, seed=kg.JUDGE_SEED + 10 * t["tid"] + r,
                     prompt=kg.prompt(t["question"], f, rg.show(sol[("judge", t["tid"])]), rg.show(sol[("peer", t["tid"])])))
                for t in D for r in kg.REPS for f in kg.FRAMES]
    llm = rg.LLM(model=os.environ.get("MODEL", "/model"), dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.88, seed=0,
                 structured_outputs_config={"backend": "guidance", "disable_any_whitespace": True},
                 **json.loads(os.environ.get("LLM_KW", "{}")))
    t0 = time.time()
    CH = 960   # whole problems (24 judge jobs each); a timeout keeps finished chunks
    with open(out, "x") as f:
        for i in range(0, len(jobs), CH):
            for j in run(llm, jobs[i:i + CH]):
                f.write(json.dumps(j) + "\n")
            f.flush()
    print(json.dumps({"stage": stage, "args": sys.argv[2:4], "ctk": CTK, "jobs": len(jobs), "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
