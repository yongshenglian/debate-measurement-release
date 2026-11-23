"""Debate-settings factorial (reports/debate-settings-20261120-a/PROTOCOL.md).

One harness, two prompt families taken verbatim from the released code:
  dov  : Debate-or-Vote (src/main.py get_new_message, dense; src/evaluator.py GSM8K suffix)
  dmad : DMAD (inference/prompts.py templates; inference/vllm_debate.parse_response), stop "</answer>"
Factors switched per arm: family/wording, chat template vs raw, max new tokens, what peers' content is shown.
Fixed for all arms: Qwen2.5-7B-Instruct, the same 300 GSM8K problems, 5 agents, rounds 0..3, T 1.0, top-p 0.9,
per-(problem, round, agent) seeds shared across arms.

Usage (inside the pinned vLLM image): settings_gen.py <arm> <csv> <out.json> [limit]
"""
import csv, json, sys, time
sys.path.insert(0, "/dmad")
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from inference.prompts import INITIAL_PROMPT, format_multi_agent_prompt
from inference.vllm_debate import parse_response

N_AGENTS, N_DEBATE_ROUNDS = 5, 3
DOV_SUFFIX = ' Make sure to state your final answer in curly brackets at the very end of your response, just like: "{final answer: 123}".'
DMAD_ANSWER_INSTR = ("\nThink step by step first, at the end of your reasoning, provide your final answer in the following format:\n"
                     "<answer>YOUR FINAL ANSWER</answer>")

#            r0 / debate prompt family, debate wording,  chat,  max_tokens, share
ARMS = {
    "S1_dov":          dict(family="dov",  wording="dov",  chat=False, max_tokens=512,  share="full"),
    "S2_dmad":         dict(family="dmad", wording="dmad", chat=True,  max_tokens=1024, share="full"),
    "S3_dmad_dovword": dict(family="dmad", wording="dov",  chat=True,  max_tokens=1024, share="full"),
    "S4_dmad_raw":     dict(family="dmad", wording="dmad", chat=False, max_tokens=1024, share="full"),
    "S5_dmad_512":     dict(family="dmad", wording="dmad", chat=True,  max_tokens=512,  share="full"),
    "S6_dmad_ansonly": dict(family="dmad", wording="dmad", chat=True,  max_tokens=1024, share="answers"),
    "S7_dov_chat":     dict(family="dov",  wording="dov",  chat=True,  max_tokens=512,  share="full"),
}


def seed(qi, rnd, a):
    return 20261120 + 100000 * qi + 1000 * rnd + a


def initial(cfg, q):
    return q + DOV_SUFFIX if cfg["family"] == "dov" else INITIAL_PROMPT.format(question=q, extra_info="")


def debate_msg(cfg, q, texts, i):
    """Prompt for agent i given all agents' previous-round texts (dense: every other agent is a peer)."""
    peers = [j for j in range(len(texts)) if j != i]
    if cfg["wording"] == "dov":   # Debate-or-Vote get_new_message, dense branch, verbatim
        msg = "These are the recent opinions from other agents: "
        for j in peers:
            msg += f"\n\nOne of the agents' response: \n{texts[j]}\n"
        msg += f"\n\nThis was your most recent opinion:\n{texts[i]}\n"
        msg += f"\n\nUse these opinions carefully as additional advice to revise your recent opinion to give your final answer to the question:\n{q}"
        return msg + (DOV_SUFFIX if cfg["family"] == "dov" else DMAD_ANSWER_INSTR)
    parsed = [parse_response(t, False) for t in texts]   # DMAD: (reasoning, answer, confidence)
    others = [{"agent_id": j + 1, "reasoning": "" if cfg["share"] == "answers" else parsed[j][0],
               "final_answer": parsed[j][1]} for j in peers]
    own = {"reasoning": parsed[i][0], "final_answer": parsed[i][1]}
    ex = {"question": q, "label": "", "dataset": "gsm8k"}
    return format_multi_agent_prompt(ex, others, own, use_confidence=False, chat_mode=False)["prompt"]


def main():
    arm, csv_path, out = sys.argv[1:4]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
    cfg = ARMS[arm]
    rows = list(csv.DictReader(open(csv_path)))[:limit]
    tok = AutoTokenizer.from_pretrained("/model")
    wrap = (lambda c: tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True)) \
        if cfg["chat"] else (lambda c: c)
    stop = ["</answer>"] if cfg["family"] == "dmad" else None
    llm = LLM(model="/model", dtype="bfloat16", max_model_len=16384, gpu_memory_utilization=0.88, seed=0)
    t0 = time.time()
    hist = [[] for _ in rows]            # hist[qi] = list of rounds, each a list of N_AGENTS texts
    for rnd in range(N_DEBATE_ROUNDS + 1):
        prompts, params = [], []
        for qi, r in enumerate(rows):
            for a in range(N_AGENTS):
                c = initial(cfg, r["question"]) if rnd == 0 else debate_msg(cfg, r["question"], hist[qi][-1], a)
                prompts.append(wrap(c))
                params.append(SamplingParams(temperature=1.0, top_p=0.9, max_tokens=cfg["max_tokens"], stop=stop,
                                             include_stop_str_in_output=True, seed=seed(qi, rnd, a)))
        outs = llm.generate(prompts, params)
        for qi in range(len(rows)):
            hist[qi].append([outs[qi * N_AGENTS + a].outputs[0].text for a in range(N_AGENTS)])
        print(f"round {rnd} done, {time.time() - t0:.0f}s", flush=True)
    json.dump({"arm": arm, "cfg": cfg, "seconds": round(time.time() - t0),
               "results": [{"question": r["question"], "label": r["label"], "rounds": h} for r, h in zip(rows, hist)]},
              open(out, "x"))


if __name__ == "__main__":
    main()
