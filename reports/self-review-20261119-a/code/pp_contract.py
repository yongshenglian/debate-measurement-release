"""Editor prompts, variant validation, peer construction and recipient payloads (no labels)."""
import difflib
import json
import re

from pc_contract import validate_candidate
from pd_contract import EDIT_SYSTEM, INSTRUCTIONS as PD_INSTRUCTIONS, matched_messages  # noqa: F401
import pp_design as design

INSTRUCTIONS = {"private": PD_INSTRUCTIONS["private"], "anchor": PD_INSTRUCTIONS["private"],
                "full": PD_INSTRUCTIONS["full"]}
MESSAGE_CONDITION = {"private": "private", "anchor": "private", "full": "peer"}

_KEEP = ("Keep every value exactly as given, even if it looks wrong; do not correct anything. "
         "Set error_description to \"none\".")
EDIT_TASKS = {
    "key": ("Rewrite the solution as only its essential equations, in the same order, with the "
            "same numbers and the same final answer. Use no words except units. " + _KEEP),
    "asserted": ("Rewrite the solution so that each intermediate value is stated as a plain fact, "
                 "without showing any calculation: no operators and no equations. Keep the same "
                 "values in the same order and the same final answer. " + _KEEP),
}
OPERATION = re.compile(r"\d\s*[-+*/×÷xX]\s*\d")


def edit_messages(task, source, kind):
    if set(task) != {"id", "question"}:
        raise ValueError("public task must contain only id and question")
    payload = {"question": task["question"], "solution": validate_candidate(source),
               "task": EDIT_TASKS[kind]}
    return [{"role": "system", "content": EDIT_SYSTEM},
            {"role": "user", "content": json.dumps(payload, separators=(",", ":"))}]


def variant_ok(kind, source, edit, question, key_value):
    """Label-free validation of an editor variant against its source twin."""
    if edit is None:
        return False
    text = edit["rationale"].strip()
    if not text or edit["answer"] != source["answer"] or len(text) > design.MAX_RATIONALE_CHARS:
        return False
    allowed = set(design.numbers(source["rationale"])) | set(design.numbers(question)) | {
        str(source["answer"])}
    found = design.numbers(text)
    if not set(found) <= allowed or key_value not in found:
        return False
    if kind == "key":
        return "=" in text and len(text) <= design.KEY_MAX_LENGTH_RATIO * len(source["rationale"])
    if kind == "asserted":
        return "=" not in text and not OPERATION.search(text)
    raise ValueError(kind)


def peer_for(source, variant, edits=None):
    """The peer message (rationale, answer) shown for one variant of a twin."""
    source = validate_candidate({"rationale": source["rationale"], "answer": source["answer"]})
    answer = source["answer"]
    if variant == "full":
        return source
    if variant == "confident":
        return validate_candidate({"rationale": source["rationale"].rstrip() + design.CONFIDENT,
                                   "answer": answer})
    if variant == "hedged":
        return validate_candidate({"rationale": source["rationale"].rstrip() + design.HEDGED,
                                   "answer": answer})
    if variant in design.EDIT_KINDS:
        edit = edits[variant]
        return validate_candidate({"rationale": edit["rationale"], "answer": edit["answer"]})
    if variant == "stated1":
        return {"rationale": design.STATED.format(answer), "answer": answer}
    if variant == "stated5":
        return {"rationale": " ".join([design.STATED.format(answer)] * design.STATED_REPEATS),
                "answer": answer}
    raise ValueError(variant)


def payload(task, own, peer, presentation, reference=None):
    """v1 field layout; `anchor` adds a reference_code and no peer."""
    if (presentation == "full") != (peer is not None):
        raise ValueError("only the full presentation carries a peer")
    if (presentation == "anchor") != (reference is not None):
        raise ValueError("only the anchor presentation carries a reference code")
    value = {"task_id": task["id"], "question": task["question"],
             "your_frozen_proposal": validate_candidate(own)}
    if reference is not None:
        value["reference_code"] = str(int(reference))
    value.update({"message_condition": MESSAGE_CONDITION[presentation],
                  "unverified_messages": [] if peer is None else [validate_candidate(peer)],
                  "instruction": INSTRUCTIONS[presentation], "padding": ""})
    return value


def adv_text(cot):
    """Strip GSM8K calculator annotations and the '#### N' line (the answer field carries N)."""
    text = re.sub(r"<<[^>]*>>", "", cot)
    lines = [line.rstrip() for line in text.split("\n") if not line.strip().startswith("####")]
    return "\n".join(lines).strip()


def adv_integer(value):
    value = value.replace(",", "").strip()
    return int(value) if re.fullmatch(r"-?\d+", value) else None


def normalize_question(text):
    return re.sub(r"\s+", " ", text.replace("’", "'")).strip().lower()


def adv_eligible(row, label):
    """Label-aware gsm8k_adv row rule; returns (C, E) candidates or None."""
    if adv_integer(row["answer"]) != label:
        return None
    wrong = adv_integer(row["messy_answer"])
    if wrong is None or wrong == label:
        return None
    c, e = adv_text(row["original_cot"]), adv_text(row["messy_cot"])
    if not (0 < len(c) <= design.MAX_RATIONALE_CHARS and 0 < len(e) <= design.MAX_RATIONALE_CHARS):
        return None
    ratio = len(e) / len(c)
    if not design.ADV_MIN_LENGTH_RATIO <= ratio <= design.ADV_MAX_LENGTH_RATIO:
        return None
    if difflib.SequenceMatcher(None, c, e, autojunk=False).ratio() < design.ADV_MIN_SIMILARITY:
        return None
    return {"rationale": c, "answer": label}, {"rationale": e, "answer": wrong}


def bookkeeping(peer, c_twin):
    """Number counts of one shown peer message."""
    if peer is None:
        return None
    found = design.numbers(peer["rationale"])
    c_numbers = set(design.numbers(c_twin["rationale"])) | {str(c_twin["answer"])}
    return {"answer_occurrences": found.count(str(peer["answer"])) + 1,
            "number_tokens": len(found),
            "numbers_absent_from_C": sum(n not in c_numbers for n in found),
            "chars": len(peer["rationale"])}
