"""v2 Mistral arm: HF tokenizer mode, prompts rendered with the model's chat template (as in our Mistral qualification)."""
import json, os, sys, time
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main():
    _, jobs_path, out_path = sys.argv[1:4]
    jobs = [json.loads(l) for l in open(jobs_path)]
    params = [SamplingParams(temperature=1.0, top_p=0.9, max_tokens=512, seed=j["seed"]) for j in jobs]
    llm = LLM(model="/model", tokenizer_mode="hf", config_format="hf", load_format="auto",
              max_model_len=8192, gpu_memory_utilization=0.88, seed=0, language_model_only=True)
    tok = llm.get_tokenizer()
    tpl = json.load(open("/model/chat_template.json"))["chat_template"] if os.path.exists("/model/chat_template.json") else None
    prompts = []
    for j in jobs:
        ids = tok.apply_chat_template(j["messages"], chat_template=tpl, tokenize=True, add_generation_prompt=True)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        prompts.append(TokensPrompt(prompt_token_ids=list(ids)))
    t0 = time.time()
    outs = llm.generate(prompts, params)
    with open(out_path, "x") as f:
        for j, o in zip(jobs, outs):
            c = o.outputs[0]
            f.write(json.dumps({"id": j["id"], "text": c.text, "finish_reason": c.finish_reason,
                                "prompt_tokens": len(o.prompt_token_ids), "output_tokens": len(c.token_ids)}) + "\n")
    print(json.dumps({"jobs": len(jobs), "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
