"""Reverse peer test (reports/reverse-peer-20261120-a/PROTOCOL.md): does a strong model get worse reading a weaker model's work?

Models: qwen = Qwen2.5-7B-Instruct, gemma = Gemma-3-27B-it W4A16. T 0.7, top-p 0.95, top-k 20.
Each call is two turns: (1) a free-text solution (<= 1,024 tokens); (2) in the same conversation, the model states only its
final answer as JSON {"answer": integer} (structured output, guidance backend, no free whitespace, <= 16 tokens).
So peers show natural working, and every final answer is read exactly (no regex).

Stages (inside the pinned vLLM image, /model = the model being run):
  solve  <model> <tasks.jsonl> <out.jsonl>          two independent solutions per task (sample 0 = own, sample 1 = same-model peer)
  review <model> <tasks.jsonl> <qwen_solve.jsonl> <gemma_solve.jsonl> <out.jsonl>
         conditions per task: self (own solution only), peer_qwen, peer_gemma; the recipient's own solution is its sample 0;
         a same-model peer is its sample 1, the other model's peer is that model's sample 0. One seed per task, shared by conditions.
"""
import json, os, sys, time
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

SCHEMA = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"], "additionalProperties": False}
ASK = "State only your final answer as JSON."
TASK = "Solve the following math problem step by step. The final answer is an integer.\n\nProblem:\n{q}"
SELF = TASK + "\n\nYour earlier solution:\n{own}\n\nReview your earlier solution and give your final answer."
PEER = TASK + "\n\nYour earlier solution:\n{own}\n\nAnother solver's solution:\n{peer}\n\nReview both solutions and give your final answer."
SEEDS = {"solve": 31000000, "review": 47000000}


def show(sol):
    return sol["text"].strip()


def parse(text):
    try:
        v = json.loads(text)
        if isinstance(v, dict) and type(v.get("answer")) is int:
            return {"answer": v["answer"]}
    except Exception:
        pass
    return None


def run(llm, jobs):
    free = [SamplingParams(temperature=0.7, top_p=0.95, top_k=20, max_tokens=1024, seed=j["seed"]) for j in jobs]
    convs = [[{"role": "user", "content": j["prompt"]}] for j in jobs]
    outs = llm.chat(convs, free)
    convs2 = [c + [{"role": "assistant", "content": o.outputs[0].text}, {"role": "user", "content": ASK}] for c, o in zip(convs, outs)]
    ansp = [SamplingParams(temperature=0.7, top_p=0.95, top_k=20, max_tokens=16, seed=j["seed"] + 1,
                           structured_outputs=StructuredOutputsParams(json=SCHEMA)) for j in jobs]
    outs2 = llm.chat(convs2, ansp)
    for j, o, o2 in zip(jobs, outs, outs2):
        c = o.outputs[0]
        j.update(text=c.text, finish_reason=c.finish_reason, output_tokens=len(c.token_ids),
                 answer_text=o2.outputs[0].text, parsed=parse(o2.outputs[0].text))
        del j["prompt"]
    return jobs


def main():
    stage, model = sys.argv[1], sys.argv[2]
    tasks = [json.loads(l) for l in open(sys.argv[3])]
    if os.environ.get("LIMIT"):   # smoke runs only
        tasks = tasks[:int(os.environ["LIMIT"])]
    llm = LLM(model="/model", dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.88, seed=0,
              structured_outputs_config={"backend": "guidance", "disable_any_whitespace": True})  # as the peer studies
    t0 = time.time()
    if stage == "solve":
        out = sys.argv[4]
        jobs = [{"tid": t["tid"], "sample": s, "seed": SEEDS["solve"] + 10 * t["tid"] + s, "prompt": TASK.format(q=t["question"])}
                for t in tasks for s in (0, 1)]
    else:
        sol = {}
        for m, p in (("qwen", sys.argv[4]), ("gemma", sys.argv[5])):
            for l in open(p):
                d = json.loads(l); sol[(m, d["tid"], d["sample"])] = d if d["parsed"] is not None else None
        out = sys.argv[6]
        jobs = []
        for t in tasks:
            own = sol[(model, t["tid"], 0)]
            if own is None:   # recipient produced no valid first solution: excluded, counted in the analysis
                continue
            base = dict(tid=t["tid"], seed=SEEDS["review"] + t["tid"])
            jobs.append(dict(base, cond="self", prompt=SELF.format(q=t["question"], own=show(own))))
            for pm in ("qwen", "gemma"):
                peer = sol[(pm, t["tid"], 1 if pm == model else 0)]
                if peer is not None:
                    jobs.append(dict(base, cond=f"peer_{pm}", prompt=PEER.format(q=t["question"], own=show(own), peer=show(peer))))
    with open(out, "x") as f:
        for j in run(llm, jobs):
            f.write(json.dumps(j) + "\n")
    print(json.dumps({"stage": stage, "model": model, "jobs": len(jobs), "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
