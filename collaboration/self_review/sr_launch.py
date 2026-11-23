"""Guarded stage launcher for the self-review study: prepare, schedule, recipients, audit."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

ROOT = Path("/home/lian/swarmworld/research")
HERE = ROOT / "collaboration" / "self_review"
PW = ROOT / "collaboration" / "peer_warning"
sys.path[:0] = [str(PW), str(HERE)]
from pw_launch import CODE as PW_CODE, QWEN, code_unchanged, load_status  # noqa: E402
from kv_io import save, sha  # noqa: E402
from kv_prepare_run import IMAGE  # noqa: E402
from ps_launch import docker, preflight, run  # noqa: E402
import sr_design as design  # noqa: E402

RUN_PREFIX = "self-review-"
SOURCE = ROOT / "reports" / design.SOURCE_RUN
CODE = {**PW_CODE, HERE: ("sr_design.py", "sr_contract.py", "sr_schedule.py", "sr_worker.py",
                          "sr_audit.py")}
TIMEOUT = {"recipient": 7200, "cpu": 3600}
MAX_ALLOCATED_GPU_HOURS = round(design.N_GPUS * TIMEOUT["recipient"] / 3600, 2)


def prepare(run_dir):
    if run_dir.exists():
        raise FileExistsError(run_dir)
    audit = json.loads((SOURCE / "audit" / "audit.json").read_text())
    if not audit["integrity_passed"]:
        raise ValueError("Study 3 audit did not pass")
    for name in ("code", "source", "data", "hidden", "caches", "audit"):
        (run_dir / name).mkdir(parents=True)
    for gpu in range(design.N_GPUS):
        (run_dir / f"recipient-gpu{gpu}").mkdir()
        (run_dir / "caches" / f"recipient-gpu{gpu}").mkdir()
    for source, names in CODE.items():
        for name in names:
            shutil.copyfile(source / name, run_dir / "code" / name)
    shutil.copyfile(HERE / "PROTOCOL.md", run_dir / "PROTOCOL.md")
    for name in ("tasks.json", "public.json", "recipient-schedule.json"):
        shutil.copyfile(SOURCE / "data" / name, run_dir / "source" / name)
    shutil.copyfile(SOURCE / "data" / "model-manifest-qwen.json",
                    run_dir / "data" / "model-manifest-qwen.json")
    shutil.copyfile(SOURCE / "hidden" / "hidden.json", run_dir / "hidden" / "hidden.json")
    save(run_dir / "hidden" / "study3-per-task.json", audit["per_task"])
    record = {"version": design.VERSION, "image": IMAGE, "source_run": design.SOURCE_RUN,
              "source_audit_sha256": sha(SOURCE / "audit" / "audit.json"),
              "prepared": datetime.now().astimezone().isoformat(),
              "protocol_sha256": sha(run_dir / "PROTOCOL.md"),
              "code_sha256": {p.name: sha(p) for p in sorted((run_dir / "code").iterdir())},
              "source_sha256": {p.name: sha(p) for p in sorted((run_dir / "source").iterdir())},
              "hidden_sha256": {p.name: sha(p) for p in sorted((run_dir / "hidden").iterdir())},
              "max_allocated_gpu_hours": MAX_ALLOCATED_GPU_HOURS}
    save(run_dir / "prepare.json", record)
    return record


def cpu_stage(run_dir, script):
    code_unchanged(run_dir)
    models = [(QWEN, "/model", "ro"), (run_dir / "code", "/code", "ro")]
    if script == "sr_schedule.py":
        mounts = [(run_dir / "source", "/source", "ro"), (run_dir / "data", "/output", "rw")]
    else:
        mounts = [(run_dir, "/run", "ro"), (run_dir / "audit", "/output", "rw")]
    return run(docker(f"{run_dir.name}-{script[3:-3]}", mounts + models,
                      ["python3", f"/code/{script}"], TIMEOUT["cpu"], detach=False))


def launch_recipients(run_dir):
    code_unchanged(run_dir)
    schedule = json.loads((run_dir / "data" / "recipient-schedule.json").read_text())
    if (not schedule["passed"]
            or schedule["source_schedule_sha256"] != sha(run_dir / "source" / "recipient-schedule.json")):
        raise RuntimeError("schedule gate did not pass")
    gpus = [int(g) for g, jobs in schedule["workers"].items() if jobs]
    if any(any((run_dir / f"recipient-gpu{gpu}").iterdir()) for gpu in gpus):
        raise RuntimeError("recipient outputs not empty")
    record = {"preflight": preflight({gpu: "qwen" for gpu in gpus}), "containers": {}}
    for gpu in gpus:
        name = f"{run_dir.name}-recipient-gpu{gpu}"
        # Workers see only data/ (schedule and model manifest): no labels, no Study 3 outcomes.
        mounts = [(QWEN, "/model", "ro"), (run_dir / "code", "/code", "ro"),
                  (run_dir / "data", "/data", "ro"), (run_dir / f"recipient-gpu{gpu}", "/output", "rw"),
                  (run_dir / "caches" / f"recipient-gpu{gpu}", "/root/.cache", "rw")]
        command = docker(name, mounts, ["python3", "/code/sr_worker.py", "--phase", "recipient",
                                        "--gpu", str(gpu)], TIMEOUT["recipient"], gpu=gpu)
        record["containers"][str(gpu)] = {"name": name, "command": command,
                                          "container_id": run(command).strip()}
    save(run_dir / "recipient-launch.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "schedule", "recipients", "audit"))
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run.resolve()
    if run_dir.parent != ROOT / "reports" or not run_dir.name.startswith(RUN_PREFIX):
        raise ValueError("unexpected run path")
    if args.stage == "prepare":
        result = prepare(run_dir)
    elif args.stage == "schedule":
        if (run_dir / "data" / "recipient-schedule.json").exists():
            raise FileExistsError("schedule already frozen")
        result = cpu_stage(run_dir, "sr_schedule.py")
    elif args.stage == "recipients":
        result = launch_recipients(run_dir)
    else:
        for gpu in range(design.N_GPUS):
            status = load_status(run_dir / f"recipient-gpu{gpu}")["status"]
            if status != "completed":
                raise RuntimeError(f"recipient-gpu{gpu} is {status}")
        result = cpu_stage(run_dir, "sr_audit.py")
    print(result if isinstance(result, str) else json.dumps(result, indent=2, default=str))
