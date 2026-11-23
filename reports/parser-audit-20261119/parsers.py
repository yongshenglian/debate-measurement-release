"""Answer parsers of six multi-agent-debate codebases, transcribed from their released code for GSM8K-style
numeric answers. Each entry: requested final-answer format, parse(text) -> value or None, and how the
codebase scores a parse failure. Sources (fetched 2026-11-19):
  dov    Debate-or-Vote (NeurIPS 2025), src/evaluator.py evaluate_arithmetics, commit 82c929e
  du     Du et al. (ICML 2024) composable-models/llm_multiagent_debate gsm/eval_gsm.py parse_answer + fallback
  wynn   Wynn et al. 2025, TheNormativityLab/talk-aint-cheap src/analysis/evaluation.py parse_gsm8k_answer
  dmad   DMAD (2601.19921), SpaceHunterInf/DMAD inference/vllm_debate.py parse_response + utils/eval_utils clean_answer
  rec    ReConcile, dinobby/ReConcile utils.py parse_json (+ invalid_result answer "0")
  wdh    When Debate Helps (NeurIPS 2026), src/when_debate_helps/extraction.py extract_answer
"""
import ast, re
from decimal import Decimal, InvalidOperation


def _num(s):
    try:
        return float(str(s).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def dov(text):
    try:
        pred = re.findall(r"\{(.*?)\}", text)[-1]
        return float(pred.replace("final answer:", "").strip())
    except Exception:
        return None


def _du_parse(input_str):
    matches = re.findall(r"\{([0-9.,$]*)\}", input_str)
    solution = None
    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            break
    return solution


def du(text):
    a = _du_parse(text)
    if a is None or a == "":
        m = re.findall(r"\d+\.?\d*", text)        # solve_math_problems fallback: last number anywhere
        a = m[-1] if m else None
    return _num(a) if a not in (None, "") else None


def wynn(text):
    matches = re.findall(r"\{([0-9.,$]*)\}", text)
    solution = None
    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            break
    return _num(solution) if solution else None


def dmad(text):
    if "<answer>" in text and "</answer>" in text:
        ans = text.split("<answer>")[-1].split("</answer>")[0].strip()
    else:
        return None                                  # "PARSE_ERROR"
    ans = ans.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", ans)
    if not nums:
        return None
    last = nums[-1]
    return _num(last[:-1] if last.endswith(".") else last)


def dmad_generated(text):
    """DMAD generates with stop='</answer>' (include_stop_str_in_output=True): output ends at the FIRST </answer>."""
    i = text.find("</answer>")
    return dmad(text if i < 0 else text[: i + len("</answer>")])


def rec(text):
    try:
        t = text.replace("\n", " ")
        t = re.search('({.+})', t).group(0)
        t = re.sub(r"(\w)'(\w|\s)", r"\1\\'\2", t)
        r = ast.literal_eval(t)
        return _num(r["answer"])
    except Exception:
        return None


_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_BOXED = re.compile(r"\\boxed\s*\{\s*([^{}]+?)\s*\}", re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _wdh_norm(value):
    cleaned = value.strip().strip("$ ").replace(",", "")
    if not cleaned:
        return None
    try:
        d = Decimal(cleaned)
    except InvalidOperation:
        return None
    return float(d)


def wdh(text):
    if not text or not text.strip():
        return None
    cands = _ANSWER_TAG.findall(text) or _BOXED.findall(text)
    for c in reversed(cands):
        v = _wdh_norm(c)
        if v is not None:
            return v
    nums = _NUMBER.findall(text)
    return _wdh_norm(nums[-1]) if nums else None


# requested final-answer format, and how a parse FAILURE is scored by the codebase
PARSERS = {
    "dov":  {"fn": dov,  "fmt": lambda a: "{final answer: %s}" % a, "on_fail": "wrong"},
    "du":   {"fn": du,   "fmt": lambda a: "\\boxed{%s}" % a,        "on_fail": "correct"},  # compute_accuracy: None -> return 1
    "wynn": {"fn": wynn, "fmt": lambda a: "\\boxed{%s}" % a,        "on_fail": "wrong"},
    "dmad": {"fn": dmad_generated, "fmt": lambda a: "<answer>%s</answer>" % a, "on_fail": "wrong"},
    "rec":  {"fn": rec,  "fmt": None,                                "on_fail": "answer_0"},  # JSON object, see suite
    "wdh":  {"fn": wdh,  "fmt": lambda a: "<answer>%s</answer>" % a, "on_fail": "wrong"},
}
