"""Run one DMAD vanilla-MAD arm with DMAD's own code (inference/vllm_debate.run_parallel_debates, unmodified).

Usage: run_arm.py <arm> <model_dir> <dtype> <csv> <out.json> [limit]
Arms:
  faithful   use_confidence False, SamplingParams exactly as vllm_debate.main (shared seed 42, stop "</answer>")
  nostop     as faithful, without the stop string (stop-token ablation, same seed)
  agentseed  as faithful, but every (round, agent) gets its own seed 42 + 1000*round + agent (diversity fix)
  conf       use_confidence True (untrained confidence prompt), otherwise as vllm_debate.main (stop "</confidence>")
  fixed      (amendment 1) agentseed + chat template with add_generation_prompt=True (DMAD omits it)
"""
import json, sys, time, dataclasses
sys.path.insert(0, "/dmad")
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from inference.vllm_debate import run_parallel_debates
from utils.data_utils import load_debate_dataset


class PerAgentSeed:
    """Proxy for vllm.LLM: replaces the single SamplingParams by per-prompt copies with distinct seeds.
    run_parallel_debates builds prompt_batch example-major, agent-minor, one generate() call per round."""
    def __init__(self, llm, n_agents):
        self.llm, self.n, self.round = llm, n_agents, 0
    def generate(self, prompts, params, **kw):
        ps = [params.clone() for _ in prompts]
        for i, p in enumerate(ps):
            p.seed = 42 + 1000 * self.round + (i % self.n)
        self.round += 1
        return self.llm.generate(prompts, ps, **kw)


class GenPromptTokenizer:
    """Proxy for the HF tokenizer: forces add_generation_prompt=True in apply_chat_template (amendment 1)."""
    def __init__(self, tok):
        self._tok = tok
    def apply_chat_template(self, messages, **kw):
        kw["add_generation_prompt"] = True
        return self._tok.apply_chat_template(messages, **kw)
    def __getattr__(self, k):
        return getattr(self._tok, k)


def main():
    arm, model_dir, dtype, csv_path, out = sys.argv[1:6]
    limit = int(sys.argv[6]) if len(sys.argv) > 6 else None
    use_conf = arm == "conf"
    try:
        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        tok_note = "use_fast=False (as DMAD)"
    except Exception as e:  # Llama-3 ships no slow tokenizer
        tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        tok_note = f"use_fast=True (slow failed: {type(e).__name__})"
    llm = LLM(model=model_dir, dtype=dtype, gpu_memory_utilization=0.85, tensor_parallel_size=1, seed=0)
    stop = "</confidence>" if use_conf else "</answer>"
    params = SamplingParams(temperature=1.0, max_tokens=1024, top_p=0.9, top_k=-1,
                            stop=None if arm == "nostop" else stop, include_stop_str_in_output=True, seed=42)
    ex = load_debate_dataset(csv_path, fold="test")
    if limit: ex = ex[:limit]
    model = PerAgentSeed(llm, 5) if arm in ("agentseed", "fixed") else llm
    if arm == "fixed":
        tok = GenPromptTokenizer(tok)
    t0 = time.time()
    res = run_parallel_debates(examples=ex, model=model, tokenizer=tok, sampling_params=params,
                               num_agents=5, num_rounds=5, use_confidence=use_conf, save_full_history=True)
    with open(out, "x") as f:
        json.dump({"arm": arm, "model_dir": model_dir, "dtype": dtype, "tokenizer": tok_note,
                   "seconds": round(time.time() - t0, 1), "results": res}, f)
    print(json.dumps({"arm": arm, "n": len(res), "seconds": round(time.time() - t0, 1), "tokenizer": tok_note}))


if __name__ == "__main__":
    main()
