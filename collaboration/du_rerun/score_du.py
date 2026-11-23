"""Score Du et al. re-run outputs with the authors' eval (gsm/eval_gsm.py, imported) and a validated extractor."""
import json, re, sys, random, types
sys.modules.setdefault("openai", types.ModuleType("openai"))  # eval_gsm imports openai but its scoring functions never use it
sys.path.insert(0, "/home/lian/swarmworld/research/source/related-work/llm_multiagent_debate/gsm")
import eval_gsm as E  # authors' parse_answer, solve_math_problems, most_frequent

NUM = r"-?[0-9][0-9,]*(?:\.[0-9]+)?"

def du_agent(text):
    """Authors' per-agent parse: last {digits/.,$}, else last number anywhere."""
    a = E.parse_answer(text)
    return a if a is not None else E.solve_math_problems(text)

def du_vote(gt, sols):
    """Authors' compute_accuracy list branch (E.compute_accuracy reads a global `pred_solutions`)."""
    E.pred_solutions = sols
    return E.compute_accuracy(gt, sols)

def boxed_last(text):
    """Validated extractor: content of the last \\boxed{...} (balanced braces), cleaned to one number."""
    out = None
    for m in re.finditer(r"\\boxed\s*\{", text):
        i, depth = m.end(), 1
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0); i += 1
        if depth == 0:
            out = text[m.end():i - 1]
    if out is None:
        return None
    s = re.sub(r"\\text\{[^}]*\}|\\(?:mathrm|textbf)\{([^}]*)\}", lambda g: g.group(1) or "", out)
    s = s.replace("\\$", "").replace("$", "").replace("\\,", "").replace("\\!", "")
    nums = re.findall(NUM, s)
    if len(nums) != 1:
        return None
    return float(nums[0].replace(",", ""))

def gt_val(gt):
    return float(E.solve_math_problems(gt))
