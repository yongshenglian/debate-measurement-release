"""Study T (reports/twins-hard-20261121-a/PROTOCOL.md): GSM8K twin study ported to hard problems.
Runs inside the pinned vLLM image (/model = the model being run). Two-turn calls as in study R/H:
free text (T 0.7, top-p 0.95, top-k 20), then {"answer": integer} by structured output.
  edit   <bases.jsonl> <out.jsonl>                  continuer (/model = Qwen2.5-7B): C and E twins per injection point
  review <jobs.jsonl> <out.jsonl>                   recipient; prompts of one task are padded to equal token length
"""
import json, sys, time
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

SCHEMA = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"], "additionalProperties": False}
ASK = "State only your final answer as JSON."
ASK_EDIT = "State only the final answer that the completed solution arrives at, as JSON."
# Twin construction (changed from GSM8K's whole-solution edit after two smoke runs: on MATH, Gemma neither carried an
# edited mistake through nor reliably made one when asked). t_build.py cuts the correct base after the line holding its
# first computed result "= N" (25-75% of its length); the E prefix has N replaced by a slip N'. Gemma then continues
# both prefixes with the same neutral instruction, so the twins differ by one injected number and its consequences.
EDIT_SYSTEM = ("You continue worked solutions to math problems. Reply with the continuation only, starting exactly where "
               "the solution so far stops, with no preamble and no comments.")
EDIT_TASK = "Continue the solution from where it stops to the final answer, in the same style."
PAD_WORD = " pad"   # digit-free filler


def parse(text):
    try:
        v = json.loads(text)
        if isinstance(v, dict) and type(v.get("answer")) is int:
            return v["answer"]
    except Exception:
        pass
    return None


def run(llm, convs, seeds, max_tokens, ask, temperature=0.7):
    free = [SamplingParams(temperature=temperature, top_p=0.95, top_k=20, max_tokens=max_tokens, seed=s) for s in seeds]
    outs = llm.chat(convs, free)
    convs2 = [c + [{"role": "assistant", "content": o.outputs[0].text}, {"role": "user", "content": ask}] for c, o in zip(convs, outs)]
    ansp = [SamplingParams(temperature=0.7, top_p=0.95, top_k=20, max_tokens=16, seed=s + 1,
                           structured_outputs=StructuredOutputsParams(json=SCHEMA)) for s in seeds]
    outs2 = llm.chat(convs2, ansp)
    return [(o.outputs[0].text, o.outputs[0].finish_reason, len(o.outputs[0].token_ids), o2.outputs[0].text, parse(o2.outputs[0].text))
            for o, o2 in zip(outs, outs2)]


def main():
    stage = sys.argv[1]
    llm = LLM(model="/model", dtype="bfloat16", max_model_len=12288, gpu_memory_utilization=0.88, seed=0,
              structured_outputs_config={"backend": "guidance", "disable_any_whitespace": True})
    tok = llm.get_tokenizer(); t0 = time.time()
    if stage == "edit":
        bases = [json.loads(l) for l in open(sys.argv[2])]
        # per injection point: 2 C attempts, 5 E attempts (the continuer is Qwen2.5-7B: in the smoke run Gemma silently
        # corrected most injected slips, 15% yield, against 55% for Qwen)
        jobs = [(b, pi, tw, k) for b in bases for pi in range(len(b["points"])) for tw, na in (("C", 2), ("E", 5)) for k in range(na)]
        convs = [[{"role": "system", "content": EDIT_SYSTEM},
                  {"role": "user", "content": f"Problem:\n{b['question']}\n\nSolution so far:\n{b['points'][pi]['prefix_' + tw]}\n\nTask: {EDIT_TASK}"}]
                 for b, pi, tw, k in jobs]
        seeds = [61000000 + 100 * b["tid"] + 20 * pi + 10 * (tw == "E") + 2 * k for b, pi, tw, k in jobs]
        res = run(llm, convs, seeds, 2048, ASK_EDIT, temperature=0.7)
        recs = [{"tid": b["tid"], "point": pi, "twin": tw, "attempt": k, "text": b["points"][pi]["prefix_" + tw] + "\n" + r[0].lstrip("\n"), "finish_reason": r[1], "answer": r[4]} for (b, pi, tw, k), r in zip(jobs, res)]
    else:
        jobs = [json.loads(l) for l in open(sys.argv[2])]
        # pad every prompt of a task to that task's longest prompt (chat-template tokens), filler before the request line
        L = {}
        for j in jobs:
            j["_n"] = len(tok.apply_chat_template([{"role": "user", "content": j["body"] + j["request"]}], tokenize=True, add_generation_prompt=True))
            L[j["tid"]] = max(L.get(j["tid"], 0), j["_n"])
        for j in jobs:
            k = L[j["tid"]] - j["_n"]
            pad = ("\n\nPadding (ignore):" + PAD_WORD * max(0, k - 6) + "\n\n") if k > 6 else "\n\n"
            j["prompt"] = j["body"] + pad + j["request"]
            j["prompt_tokens"] = len(tok.apply_chat_template([{"role": "user", "content": j["prompt"]}], tokenize=True, add_generation_prompt=True))
        res = run(llm, [[{"role": "user", "content": j["prompt"]}] for j in jobs], [j["seed"] for j in jobs], 2048, ASK)
        recs = [{"tid": j["tid"], "cond": j["cond"], "prompt_tokens": j["prompt_tokens"], "target_tokens": L[j["tid"]],
                 "text": r[0], "finish_reason": r[1], "output_tokens": r[2], "answer_text": r[3], "answer": r[4]} for j, r in zip(jobs, res)]
    with open(sys.argv[3], "x") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    print(json.dumps({"stage": stage, "jobs": len(recs), "seconds": round(time.time() - t0)}))


if __name__ == "__main__":
    main()
