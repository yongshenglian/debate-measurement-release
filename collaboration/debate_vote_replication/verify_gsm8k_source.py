"""Verify histories and top-level question logs against the pinned GSM8K cache."""
import argparse
import hashlib
import json
from pathlib import Path
import re


ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
QUESTION_PREFIX = "Question:  "
QUESTION_SUFFIX = (
    ' Make sure to state your final answer in curly brackets at the very end of '
    'your response, just like: "{final answer: 123}".'
)
ARMS = ("dense", "sparse", "central", "single")


def extract_answer(text):
    match = ANSWER_RE.search(text)
    if not match:
        raise ValueError("GSM8K answer lacks #### numeric label")
    return int(match.group(1).replace(",", "").strip())


def questions_from_log(text):
    questions = []
    for line in text.splitlines():
        if not line.startswith(QUESTION_PREFIX):
            continue
        rendered = line[len(QUESTION_PREFIX):].rstrip()
        if not rendered.endswith(QUESTION_SUFFIX):
            raise ValueError("top-level question has unexpected instruction suffix")
        questions.append(rendered[:-len(QUESTION_SUFFIX)])
    return questions


def load_history(path):
    files = list((path / "out" / "history").glob("*.jsonl"))
    if len(files) != 1:
        raise ValueError(f"expected one history under {path}, found {len(files)}")
    return [json.loads(line) for line in files[0].read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--expected-size", type=int, default=300)
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", cache_dir=args.data_dir)["test"]
    frame = dataset.to_pandas().sample(frac=1, random_state=0).reset_index(drop=True)
    frame = frame.head(args.expected_size)
    expected_questions = frame["question"].tolist()
    expected_labels = [extract_answer(value) for value in frame["answer"]]
    manifest = [
        {"index": index, "question": question, "answer": answer}
        for index, (question, answer) in enumerate(zip(expected_questions, expected_labels))
    ]
    digest = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()

    result = {
        "passed": True,
        "dataset_rows": len(dataset),
        "expected_size": args.expected_size,
        "ordered_manifest_sha256": digest,
        "arms": {},
    }
    for arm in ARMS:
        rows = load_history(args.run / arm)
        labels = [row["0"]["answer"] for row in rows]
        labels_match = labels == expected_labels[:len(rows)]
        arm_result = {"history_rows": len(rows), "labels_match_expected_prefix": labels_match}
        if not labels_match:
            result["passed"] = False
        if args.logs:
            logged = questions_from_log((args.logs / f"{arm}.log").read_text())
            questions_match = logged == expected_questions[:len(logged)]
            arm_result.update({
                "logged_questions": len(logged),
                "questions_match_expected_prefix": questions_match,
            })
            if not questions_match or len(logged) < len(rows):
                result["passed"] = False
        result["arms"][arm] = arm_result
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
