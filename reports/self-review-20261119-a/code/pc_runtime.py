"""Qwen and Gemma/Mistral runtimes under the integer contract, plus record replay.

Engine settings are copied unchanged from kv_runtime.Backend (Qwen) and
ps_runtime.StrongBackend (Gemma/Mistral); only the output contract differs.
"""
import json
import time

from kv_runtime import MAX_OUTPUT, SAMPLING, InvalidGeneration, prompt_ids, unique_object
from pc_contract import GRAMMAR_CASES, output_schema, validate_candidate

FAMILIES = ("qwen", "gemma", "mistral")
COMMON = dict(dtype="bfloat16", tensor_parallel_size=1, max_model_len=8192,
              kv_cache_dtype="bfloat16", attention_backend="FLASHINFER",
              enable_prefix_caching=False, enable_chunked_prefill=True)
ENGINE = {
    "qwen": dict(COMMON, max_num_seqs=4, gpu_memory_utilization=0.80,
                 limit_mm_per_prompt={"image": 0, "video": 0},
                 structured_outputs_config={"backend": "guidance",
                                            "disable_any_whitespace": True}),
    "strong": dict(COMMON, tokenizer_mode="hf", config_format="hf", load_format="auto",
                   trust_remote_code=False, max_num_seqs=1, gpu_memory_utilization=0.85,
                   language_model_only=True, kernel_config={"linear_backend": "marlin"},
                   quantization="compressed-tensors",
                   structured_outputs_config={"backend": "guidance",
                                              "disable_any_whitespace": True,
                                              "enable_in_reasoning": False}),
}


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
    return validate_candidate(json.loads(decoded, object_pairs_hook=unique_object))


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
    if tokenizer.decode(output_ids, skip_special_tokens=True).strip() != record["text"].strip():
        raise ValueError("failed-record text/token replay mismatch")
    try:
        validate_record(record, tokenizer)
    except (ValueError, json.JSONDecodeError):
        return True
    raise ValueError("record marked failed but satisfies the scorable contract")


def grammar_gate(family, model):
    import llguidance
    from transformers import AutoTokenizer
    if family not in FAMILIES:
        raise ValueError("unsupported family")
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True,
                                              trust_remote_code=False)
    if family == "mistral":
        from mistral_common.guidance.tokenizer import from_mistral_tokenizer
        ll, adapter = from_mistral_tokenizer(tokenizer.tokenizer), "mistral_common.guidance"
    else:
        import llguidance.hf
        ll, adapter = llguidance.hf.from_tokenizer(tokenizer), "llguidance.hf"
    grammar = llguidance.LLMatcher.grammar_from_json_schema(
        output_schema(), defaults={"whitespace_flexible": False})
    if llguidance.LLMatcher.validate_grammar(grammar, ll):
        raise ValueError("invalid installed grammar")
    checks = []
    for value, expected in GRAMMAR_CASES:
        matcher = llguidance.LLMatcher(ll, grammar, log_level=0)
        if matcher.get_error() or matcher.get_grammar_warnings():
            raise ValueError("installed grammar warning")
        ids = tokenizer.encode(json.dumps(value, separators=(",", ":")),
                               add_special_tokens=False)
        accepted = bool(matcher.consume_tokens(ids) and matcher.is_accepting()
                        and not matcher.is_error())
        if accepted != expected:
            raise ValueError(f"installed grammar contract failure: {value!r}")
        checks.append({"expected": expected, "accepted": accepted})
    return {"passed": True, "family": family, "adapter": adapter, "checks": checks}


class Backend:
    def __init__(self, family, model, deadline):
        if family not in FAMILIES:
            raise ValueError("unsupported family")
        from vllm import LLM
        self.deadline = deadline
        self.llm = LLM(model=str(model), **ENGINE["qwen" if family == "qwen" else "strong"])
        self.tokenizer = self.llm.get_tokenizer()
        cfg = self.llm.llm_engine.vllm_config
        checks = [cfg.cache_config.cache_dtype == "bfloat16",
                  str(cfg.model_config.dtype) == "torch.bfloat16",
                  cfg.parallel_config.tensor_parallel_size == 1,
                  cfg.attention_config.backend.name == "FLASHINFER",
                  cfg.structured_outputs_config.backend == "guidance",
                  cfg.structured_outputs_config.disable_any_whitespace]
        if family != "qwen":
            checks += [str(cfg.model_config.quantization).replace("-", "_") == "compressed_tensors",
                       cfg.kernel_config.linear_backend == "marlin"]
        if not all(checks):
            raise ValueError("effective runtime configuration mismatch")
        self.runtime = {
            "family": family, "quantization": str(cfg.model_config.quantization),
            "effective_cache_dtype": str(cfg.cache_config.cache_dtype),
            "model_dtype": str(cfg.model_config.dtype),
            "attention_backend": cfg.attention_config.backend.name,
            "tensor_parallel_size": cfg.parallel_config.tensor_parallel_size,
            "max_model_len": cfg.model_config.max_model_len,
            "max_num_seqs": cfg.scheduler_config.max_num_seqs,
            "structured_backend": cfg.structured_outputs_config.backend,
            "output_schema": output_schema(),
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
            SamplingParams(**SAMPLING, seed=seed, max_tokens=MAX_OUTPUT,
                           structured_outputs=StructuredOutputsParams(json=output_schema())),
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
            candidate = validate_record(record, self.tokenizer)
        except (ValueError, json.JSONDecodeError) as error:
            raise InvalidGeneration(record, type(error).__name__ + ": " + str(error)) from error
        return record, candidate
