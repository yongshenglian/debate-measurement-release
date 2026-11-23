"""Pinned Qwen2.5-7B runtime and independent record validation."""
import json
from pathlib import Path
import time

from kv_task import output_schema, validate_candidate


MAX_INPUT = 4096
MAX_OUTPUT = 1024
SAMPLING = {"temperature": 0.7, "top_p": 0.95, "top_k": 20}


class InvalidGeneration(ValueError):
    """Preserve a replayable backend record when generation is not scorable."""

    def __init__(self, record, reason):
        super().__init__(reason)
        self.record = record
        self.reason = reason


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def prompt_ids(tokenizer, messages):
    kwargs = {"tokenize": True, "add_generation_prompt": True}
    try:
        ids = tokenizer.apply_chat_template(messages, return_dict=False, **kwargs)
    except TypeError:
        # Test doubles and older compatible tokenizers may not expose the newer
        # return_dict argument. Normalize their legacy result below as well.
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(ids, "keys"):
        if "input_ids" not in ids:
            raise ValueError("chat template encoding lacks input_ids")
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise ValueError("unexpected batched chat template result")
        ids = ids[0]
    ids = list(ids)
    if any(type(token) is not int for token in ids):
        raise ValueError("chat template returned non-integer token IDs")
    if not 0 < len(ids) <= MAX_INPUT:
        raise ValueError("input token limit exceeded; no truncation allowed")
    return ids


def validate_record(record, tokenizer):
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
    value = json.loads(decoded, object_pairs_hook=unique_object)
    return validate_candidate(value)


def validate_failed_record(record, tokenizer):
    """Replay a record that failed the normal scorable-output contract."""
    expected_prompt = prompt_ids(tokenizer, record["messages"])
    if (record["prompt_token_ids"] != expected_prompt
            or record["input_tokens"] != len(expected_prompt)):
        raise ValueError("failed-record prompt token replay mismatch")
    output_ids = record["output_token_ids"]
    if (record["output_tokens"] != len(output_ids)
            or not 0 <= len(output_ids) <= MAX_OUTPUT):
        raise ValueError("failed-record output count mismatch")
    decoded = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    if decoded != record["text"].strip():
        raise ValueError("failed-record text/token replay mismatch")
    try:
        validate_record(record, tokenizer)
    except (ValueError, json.JSONDecodeError):
        return True
    raise ValueError("record marked failed but satisfies the scorable contract")


def grammar_gate(model):
    import llguidance
    import llguidance.hf
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True,
                                              trust_remote_code=False)
    ll = llguidance.hf.from_tokenizer(tokenizer)
    grammar = llguidance.LLMatcher.grammar_from_json_schema(
        output_schema(), defaults={"whitespace_flexible": False})
    if llguidance.LLMatcher.validate_grammar(grammar, ll):
        raise ValueError("invalid installed grammar")
    cases = (
        ({"rationale": "work", "answer": "12"}, True),
        ({"answer": "12", "rationale": "work"}, False),
        ({"rationale": "work", "answer": "12.0"}, False),
        ({"rationale": "x" * 1201, "answer": "12"}, False),
        ({"rationale": "work", "answer": "12", "extra": 1}, False),
    )
    checks = []
    for value, expected in cases:
        matcher = llguidance.LLMatcher(ll, grammar, log_level=0)
        if matcher.get_error() or matcher.get_grammar_warnings():
            raise ValueError("installed grammar warning")
        ids = tokenizer.encode(json.dumps(value, separators=(",", ":")),
                               add_special_tokens=False)
        consumed = matcher.consume_tokens(ids)
        accepted = bool(consumed and matcher.is_accepting() and not matcher.is_error())
        if accepted != expected:
            raise ValueError("installed grammar contract failure")
        checks.append({"expected": expected, "accepted": accepted})
    return {"passed": True, "checks": checks}


class Backend:
    def __init__(self, model, cache_dtype, deadline):
        if cache_dtype not in ("bfloat16", "fp8_e4m3"):
            raise ValueError("unsupported cache dtype")
        from vllm import LLM
        self.deadline = deadline
        self.cache_dtype = cache_dtype
        self.llm = LLM(
            model=str(model), dtype="bfloat16", tensor_parallel_size=1,
            max_model_len=8192, max_num_seqs=4, gpu_memory_utilization=0.80,
            attention_backend="FLASHINFER", kv_cache_dtype=cache_dtype,
            enable_prefix_caching=False, enable_chunked_prefill=True,
            limit_mm_per_prompt={"image": 0, "video": 0},
            structured_outputs_config={"backend": "guidance",
                                       "disable_any_whitespace": True},
        )
        self.tokenizer = self.llm.get_tokenizer()
        cfg = self.llm.llm_engine.vllm_config
        actual_cache = str(cfg.cache_config.cache_dtype)
        expected_cache = "bfloat16" if cache_dtype == "bfloat16" else "fp8_e4m3"
        if (expected_cache not in actual_cache
                or str(cfg.model_config.dtype) != "torch.bfloat16"
                or cfg.parallel_config.tensor_parallel_size != 1
                or cfg.attention_config.backend.name != "FLASHINFER"
                or cfg.structured_outputs_config.backend != "guidance"
                or not cfg.structured_outputs_config.disable_any_whitespace):
            raise ValueError("effective runtime configuration mismatch")
        self.runtime = {
            "requested_cache_dtype": cache_dtype,
            "effective_cache_dtype": actual_cache,
            "model_dtype": str(cfg.model_config.dtype),
            "attention_backend": cfg.attention_config.backend.name,
            "tensor_parallel_size": cfg.parallel_config.tensor_parallel_size,
            "max_model_len": cfg.model_config.max_model_len,
            "max_num_seqs": cfg.scheduler_config.max_num_seqs,
            "structured_backend": cfg.structured_outputs_config.backend,
        }

    def generate(self, messages, seed):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        if time.monotonic() >= self.deadline:
            raise TimeoutError("application deadline reached")
        ids = prompt_ids(self.tokenizer, messages)
        started = time.monotonic()
        result = self.llm.generate(
            [{"prompt_token_ids": ids}],
            SamplingParams(
                **SAMPLING, seed=seed, max_tokens=MAX_OUTPUT,
                structured_outputs=StructuredOutputsParams(json=output_schema()),
            ),
            use_tqdm=False,
        )[0]
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
            candidate = validate_record(record, self.tokenizer)
        except (ValueError, json.JSONDecodeError) as error:
            raise InvalidGeneration(record, type(error).__name__ + ": " + str(error)) from error
        return record, candidate
