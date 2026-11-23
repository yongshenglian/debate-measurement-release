"""Editor (Gemma) runtime under its three-field contract; Qwen recipients use pc_runtime."""
import json
import time

from kv_runtime import MAX_OUTPUT, SAMPLING, InvalidGeneration, prompt_ids, unique_object
from pc_runtime import Backend
from pd_contract import EDITOR_GRAMMAR_CASES, editor_schema, validate_edit


def validate_editor_record(record, tokenizer):
    expected_prompt = prompt_ids(tokenizer, record["messages"])
    if (record["prompt_token_ids"] != expected_prompt
            or record["input_tokens"] != len(expected_prompt)):
        raise ValueError("prompt token replay mismatch")
    output_ids = record["output_token_ids"]
    if (record["finish_reason"] != "stop" or not 0 < len(output_ids) <= MAX_OUTPUT
            or record["output_tokens"] != len(output_ids)):
        raise ValueError("abnormal or over-limit generation")
    decoded = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    if decoded != record["text"].strip():
        raise ValueError("output text/token replay mismatch")
    return validate_edit(json.loads(decoded, object_pairs_hook=unique_object))


def validate_failed_editor_record(record, tokenizer):
    expected_prompt = prompt_ids(tokenizer, record["messages"])
    if (record["prompt_token_ids"] != expected_prompt
            or record["input_tokens"] != len(expected_prompt)
            or record["output_tokens"] != len(record["output_token_ids"])):
        raise ValueError("failed-record replay mismatch")
    if (tokenizer.decode(record["output_token_ids"], skip_special_tokens=True).strip()
            != record["text"].strip()):
        raise ValueError("failed-record text/token replay mismatch")
    try:
        validate_editor_record(record, tokenizer)
    except (ValueError, json.JSONDecodeError):
        return True
    raise ValueError("record marked failed but satisfies the editor contract")


def editor_grammar_gate(model):
    import llguidance
    import llguidance.hf
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    ll = llguidance.hf.from_tokenizer(tokenizer)
    grammar = llguidance.LLMatcher.grammar_from_json_schema(
        editor_schema(), defaults={"whitespace_flexible": False})
    if llguidance.LLMatcher.validate_grammar(grammar, ll):
        raise ValueError("invalid installed grammar")
    checks = []
    for value, expected in EDITOR_GRAMMAR_CASES:
        matcher = llguidance.LLMatcher(ll, grammar, log_level=0)
        if matcher.get_error() or matcher.get_grammar_warnings():
            raise ValueError("installed grammar warning")
        ids = tokenizer.encode(json.dumps(value, separators=(",", ":")), add_special_tokens=False)
        accepted = bool(matcher.consume_tokens(ids) and matcher.is_accepting()
                        and not matcher.is_error())
        if accepted != expected:
            raise ValueError(f"installed editor grammar failure: {value!r}")
        checks.append({"expected": expected, "accepted": accepted})
    return {"passed": True, "family": "gemma", "schema": "editor", "checks": checks}


class EditorBackend(Backend):
    def __init__(self, model, deadline):
        super().__init__("gemma", model, deadline)
        self.runtime["output_schema"] = editor_schema()

    def generate(self, messages, seed):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        if time.monotonic() >= self.deadline:
            raise TimeoutError("application deadline reached")
        ids = prompt_ids(self.tokenizer, messages)
        started = time.monotonic()
        result = self.llm.generate(
            [{"prompt_token_ids": ids}],
            SamplingParams(**SAMPLING, seed=seed, max_tokens=MAX_OUTPUT,
                           structured_outputs=StructuredOutputsParams(json=editor_schema())),
            use_tqdm=False)[0]
        output = result.outputs[0]
        record = {
            "messages": messages,
            "prompt_token_ids": list(result.prompt_token_ids),
            "output_token_ids": list(output.token_ids),
            "input_tokens": len(result.prompt_token_ids),
            "output_tokens": len(output.token_ids),
            "finish_reason": output.finish_reason,
            "inference_seconds": time.monotonic() - started,
            "text": output.text,
        }
        try:
            candidate = validate_editor_record(record, self.tokenizer)
        except (ValueError, json.JSONDecodeError) as error:
            raise InvalidGeneration(record, type(error).__name__ + ": " + str(error)) from error
        return record, candidate
