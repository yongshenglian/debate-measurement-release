"""Strict post-exit audit for the explicit one-agent GSM8K arm."""
import argparse
import hashlib
import json
from pathlib import Path

from verify_gsm8k_source_v2 import extract_answer, questions_from_log


def audit_rows(rows, expected_labels):
    problems = []
    correct = 0
    empty = 0
    for index, (row, label) in enumerate(zip(rows, expected_labels)):
        if list(row) != ["0"]:
            problems.append(f"row {index}: rounds")
            continue
        record = row["0"]
        if record["answer"] != label:
            problems.append(f"row {index}: source label")
        for key in ("responses", "final_answers", "final_answer_iscorr"):
            if len(record[key]) != 1:
                problems.append(f"row {index}: {key} count")
        if len(record["final_answers"]) != 1 or len(record["final_answer_iscorr"]) != 1:
            continue
        answer = record["final_answers"][0]
        recomputed = answer != "" and float(answer) == float(label)
        if bool(record["final_answer_iscorr"][0]) != recomputed:
            problems.append(f"row {index}: individual correctness")
        if bool(record["debate_answer_iscorr"]) != recomputed:
            problems.append(f"row {index}: aggregate correctness")
        if record["debate_answer"] != answer:
            problems.append(f"row {index}: single aggregate answer")
        correct += recomputed
        empty += answer == ""
    return {"correct": correct, "empty_extractions": empty, "problems": problems}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--expected-size", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", cache_dir=args.data_dir)["test"]
    frame = dataset.to_pandas().sample(frac=1, random_state=0).reset_index(drop=True)
    frame = frame.head(args.expected_size)
    expected_questions = frame["question"].tolist()
    expected_labels = [extract_answer(value) for value in frame["answer"]]
    rows = [json.loads(line) for line in args.history.read_text().splitlines()]
    logged_questions = questions_from_log(args.log.read_text())
    row_audit = audit_rows(rows, expected_labels[:len(rows)])
    passed = (
        len(rows) == args.expected_size
        and len(logged_questions) == args.expected_size
        and logged_questions == expected_questions
        and not row_audit["problems"]
    )
    result = {
        "passed": passed,
        "expected_rows": args.expected_size,
        "history_rows": len(rows),
        "logged_questions": len(logged_questions),
        "questions_match": logged_questions == expected_questions,
        "correct": row_audit["correct"],
        "accuracy": row_audit["correct"] / args.expected_size,
        "empty_extractions": row_audit["empty_extractions"],
        "problems": row_audit["problems"],
        "history_sha256": hashlib.sha256(args.history.read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
