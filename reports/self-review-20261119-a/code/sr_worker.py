"""One-GPU worker for self-review (copy of pw_worker): Qwen recipient revisions only."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import traceback

from kv_io import save, sha
from kv_runtime import InvalidGeneration, prompt_ids
from pc_contract import output_schema
from pc_runtime import Backend, grammar_gate

SCHEDULES = {"recipient": "recipient-schedule.json"}
MAX_SECONDS = {"recipient": 7100}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=tuple(SCHEDULES), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if args.gpu not in range(4):
        raise ValueError("invalid worker scope")
    output = Path("/output")
    if any(output.iterdir()):
        raise ValueError("output directory must be empty")
    started = time.monotonic()
    schedule_path = Path("/data") / SCHEDULES[args.phase]
    jobs = json.loads(schedule_path.read_text())["workers"][str(args.gpu)]
    family = "qwen"
    state = {"status": "preflight", "completed_calls": 0, "expected_calls": len(jobs),
             "invalid_generations": 0, "gpu": args.gpu, "phase": args.phase, "family": family}

    def update(**values):
        state.update(values, seconds=time.monotonic() - started,
                     utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        save(output / "status.json", state)

    update()
    try:
        model_manifest = json.loads(Path(f"/data/model-manifest-{family}.json").read_text())
        for item in model_manifest["files"]:
            if sha(Path("/model") / item["path"]) != item["sha256"]:
                raise ValueError("model checksum mismatch")
        import torch
        if (torch.cuda.device_count() != 1
                or torch.cuda.get_device_name(0) != "NVIDIA RTX A6000"
                or torch.cuda.get_device_capability(0) != (8, 6)):
            raise ValueError("expected one Ampere A6000")
        if torch.cuda.mem_get_info()[0] < 46_000 * 1024**2:
            raise ValueError("GPU occupied")
        save(output / "manifest.json", {
            "gpu": args.gpu, "phase": args.phase, "family": family,
            "expected_calls": len(jobs), "schedule_sha256": sha(schedule_path),
            "model_manifest": model_manifest,
            "code_sha256": {path.name: sha(path) for path in Path("/code").glob("*.py")},
            "packages": {name: importlib.metadata.version(name) for name in
                         ("vllm", "torch", "transformers", "tokenizers", "llguidance")},
        })
        save(output / "grammar.json", grammar_gate("qwen", Path("/model")))
        update(status="loading_model")
        schema = output_schema()
        with (output / "events.jsonl").open("x", buffering=1) as stream:
            deadline = started + MAX_SECONDS[args.phase]
            backend = Backend("qwen", Path("/model"), deadline)
            save(output / "runtime.json", backend.runtime)
            update(status="running")
            for call_index, job in enumerate(jobs):
                if ("input_tokens" in job and
                        len(prompt_ids(backend.tokenizer, job["messages"])) != job["input_tokens"]):
                    raise ValueError("frozen prompt token count drift")
                try:
                    record, candidate = backend.generate(job["messages"], job["seed"])
                    failure = None
                except InvalidGeneration as error:
                    record, candidate = error.record, None
                    failure = {"reason": error.reason, "finish_reason": record["finish_reason"]}
                event = {"gpu": args.gpu, "phase": args.phase, "call_index": call_index,
                         "job": job, "schema": schema, "record": record,
                         "candidate": candidate, "generation_failure": failure}
                stream.write(json.dumps(event, separators=(",", ":")) + "\n")
                update(completed_calls=call_index + 1,
                       invalid_generations=state["invalid_generations"] + (failure is not None))
        update(status="completed")
        print(json.dumps(state), flush=True)
    except Exception as error:
        update(status="failed", error=type(error).__name__ + ": " + str(error))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
