"""Du et al. (ICML 2024) GSM8K debate, ported from gsm/gen_gsm.py (composable-models/llm_multiagent_debate @ 9846749)
to a local open model via offline vLLM. Prompts come from the authors' construct_message (imported unchanged) and
their initial-prompt string (copied verbatim). The loop is the same: agents=3, every agent sees the others' latest
answers, and the question order is random.seed(0); random.shuffle(test.jsonl).

Changes for local serving (logged in the protocol):
- the API call is replaced by batched vllm chat with T=1.0, top_p=1.0 (the OpenAI defaults) and max_tokens=2048,
  with a distinct seed per (question, agent, round);
- calls are batched across questions per (round, agent). Contexts are identical to the sequential original:
  construct_message reads index 2*round-1, the previous round's assistant message, so agent i in round r sees
  only round r-1 answers.

Usage: gen_du.py <model_dir> <test.jsonl> <n_questions> <rounds> <out.json>
"""
import json, random, sys, time
sys.path.insert(0, "/du/gsm")
from gen_gsm import construct_message, read_jsonl  # authors' code
from vllm import LLM, SamplingParams

INIT = """Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """


def main():
    model_dir, path, nq, rounds, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    agents = 3
    random.seed(0)
    questions = read_jsonl(path)
    random.shuffle(questions)
    questions = questions[:nq]
    ctx = {q["question"]: [[{"role": "user", "content": INIT.format(q["question"])}] for _ in range(agents)]
           for q in questions}
    llm = LLM(model=model_dir, max_model_len=16384, gpu_memory_utilization=0.85, seed=0)
    t0 = time.time()
    for rnd in range(rounds):
        for i in range(agents):
            batch, keys = [], []
            for qi, q in enumerate(questions):
                c = ctx[q["question"]]
                if rnd != 0:
                    others = c[:i] + c[i + 1:]
                    c[i].append(construct_message(others, q["question"], 2 * rnd - 1))
                batch.append([{"role": m["role"], "content": m["content"]} for m in c[i]]); keys.append(qi)
            params = [SamplingParams(temperature=1.0, top_p=1.0, max_tokens=2048,
                                     seed=100000 * rnd + 1000 * i + qi) for qi in keys]
            outs = llm.chat(batch, params)
            for qi, o in zip(keys, outs):
                ctx[questions[qi]["question"]][i].append(
                    {"role": "assistant", "content": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason})
    res = {q["question"]: (ctx[q["question"]], q["answer"]) for q in questions}
    with open(out, "x") as f:
        json.dump(res, f)
    print(json.dumps({"n": len(res), "rounds": rounds, "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
