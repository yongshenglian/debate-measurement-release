"""Replay saved Debate-or-Vote histories and isolate communication effects."""
import argparse
import json
from pathlib import Path
import random


ARMS = ("dense", "sparse", "central", "single")


def load_arm(root, arm):
    files = list((root / arm / "out" / "history").glob("*.jsonl"))
    if len(files) != 1:
        raise ValueError(f"expected one history for {arm}, found {len(files)}")
    rows = [json.loads(line) for line in files[0].read_text().splitlines()]
    if not rows:
        raise ValueError(f"empty history for {arm}")
    return rows


def bootstrap_difference(pairs, draws=20000, seed=20261117):
    """Task-paired percentile interval for later-minus-round-zero accuracy."""
    rng = random.Random(seed)
    n = len(pairs)
    observed = sum(after - before for before, after in pairs) / n
    samples = []
    for _ in range(draws):
        samples.append(sum(pairs[rng.randrange(n)][1] - pairs[rng.randrange(n)][0]
                           for _ in range(n)) / n)
    # The expression above intentionally draws paired records, but must use the same record's
    # before/after values. Recompute correctly and keep the implementation explicit.
    samples.clear()
    for _ in range(draws):
        indices = [rng.randrange(n) for _ in range(n)]
        samples.append(sum(pairs[i][1] - pairs[i][0] for i in indices) / n)
    samples.sort()
    return {"difference": observed, "ci95": [samples[int(.025 * draws)],
                                               samples[int(.975 * draws)]]}


def summarize(root, expected_rows):
    histories = {arm: load_arm(root, arm) for arm in ARMS}
    if any(len(rows) != expected_rows for rows in histories.values()):
        raise ValueError({arm: len(rows) for arm, rows in histories.items()})
    labels = [[row["0"]["answer"] for row in histories[arm]] for arm in ARMS]
    if any(values != labels[0] for values in labels[1:]):
        raise ValueError("task labels or order differ across arms")

    result = {"passed": True, "tasks": expected_rows, "arms": {}}
    for arm, rows in histories.items():
        round_names = sorted(rows[0], key=int)
        if any(sorted(row, key=int) != round_names for row in rows):
            raise ValueError(f"incomplete rounds in {arm}")
        arm_result = {"rounds": {}}
        baseline = [bool(row["0"]["debate_answer_iscorr"]) for row in rows]
        for round_name in round_names:
            aggregate = [bool(row[round_name]["debate_answer_iscorr"]) for row in rows]
            individual = [bool(value) for row in rows
                          for value in row[round_name]["final_answer_iscorr"]]
            answers = [value for row in rows for value in row[round_name]["final_answers"]]
            item = {"aggregate_correct": sum(aggregate),
                    "aggregate_accuracy": sum(aggregate) / expected_rows,
                    "individual_correct": sum(individual),
                    "individual_total": len(individual),
                    "individual_accuracy": sum(individual) / len(individual),
                    "empty_extractions": sum(value == "" for value in answers)}
            if round_name != "0":
                item["communication_transitions"] = {
                    "wrong_to_right": sum(not before and after
                                          for before, after in zip(baseline, aggregate)),
                    "right_to_wrong": sum(before and not after
                                          for before, after in zip(baseline, aggregate)),
                    "right_to_right": sum(before and after
                                          for before, after in zip(baseline, aggregate)),
                    "wrong_to_wrong": sum(not before and not after
                                          for before, after in zip(baseline, aggregate)),
                }
                item["paired_effect"] = bootstrap_difference(list(zip(baseline, aggregate)))
            arm_result["rounds"][round_name] = item
        result["arms"][arm] = arm_result
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    value = summarize(args.run, args.expected_rows)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))
