"""Offline vLLM generation inside the pinned image. Usage: fc_generate.py <mode: raw|chat> <jobs.jsonl> <out.jsonl>."""
import json, sys, time
from vllm import LLM, SamplingParams



def main():
    mode, jobs_path, out_path = sys.argv[1:4]
    jobs = [json.loads(l) for l in open(jobs_path)]
    if mode in ("raw", "chatgen"):   # Debate-or-Vote decoding: T 1.0, top-p 0.9, 512 new tokens
        params = [SamplingParams(temperature=1.0, top_p=0.9, max_tokens=512, seed=j["seed"]) for j in jobs]
    else:               # blind extractor: greedy, short JSON answer
        params = [SamplingParams(temperature=0.0, max_tokens=64, seed=j["seed"]) for j in jobs]
    llm = LLM(model="/model", max_model_len=8192, gpu_memory_utilization=0.88, seed=0, language_model_only=True)
    t0 = time.time()
    if mode == "raw":
        outs = llm.generate([j["prompt"] for j in jobs], params)
    else:
        outs = llm.chat([j["messages"] for j in jobs], params)   # chat (audit) and chatgen (debate prompt via the chat template)
    with open(out_path, "x") as f:
        for j, o in zip(jobs, outs):
            c = o.outputs[0]
            f.write(json.dumps({"id": j["id"], "text": c.text, "finish_reason": c.finish_reason,
                                "prompt_tokens": len(o.prompt_token_ids), "output_tokens": len(c.token_ids)}) + "\n")
    print(json.dumps({"jobs": len(jobs), "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
