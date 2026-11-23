"""Independent, task-paired audit of saved Debate-or-Vote histories."""
import argparse
import json
from pathlib import Path
import random


ARM_SPEC = {
    "dense": {"rounds": [str(i) for i in range(6)], "agents": 5},
    "sparse": {"rounds": [str(i) for i in range(6)], "agents": 5},
    "central": {"rounds": [str(i) for i in range(6)], "agents": 5},
    "single": {"rounds": ["0"], "agents": 1},
}


def load_arm(root, arm):
    files = list((root / arm / "out" / "history").glob("*.jsonl"))
    if len(files) != 1:
        raise ValueError(f"expected one history for {arm}, found {len(files)}")
    rows = [json.loads(line) for line in files[0].read_text().splitlines()]
    if not rows:
        raise ValueError(f"empty history for {arm}")
    return rows


def bootstrap_difference(pairs, draws=20_000, seed=20_261_117):
    """Task-paired percentile interval for after-minus-before accuracy."""
    if not pairs or draws < 1:
        raise ValueError("pairs and draws must be non-empty")
    rng = random.Random(seed)
    n = len(pairs)
    observed = sum(after - before for before, after in pairs) / n
    samples = []
    for _ in range(draws):
        indices = [rng.randrange(n) for _ in range(n)]
        samples.append(sum(pairs[i][1] - pairs[i][0] for i in indices) / n)
    samples.sort()
    lower = samples[int(0.025 * draws)]
    upper = samples[min(draws - 1, int(0.975 * draws))]
    return {"difference": observed, "ci95": [lower, upper]}


def _validate(histories, expected_rows):
    for arm, rows in histories.items():
        if len(rows) != expected_rows:
            raise ValueError(f"{arm}: expected {expected_rows} rows, found {len(rows)}")
        expected_rounds = ARM_SPEC[arm]["rounds"]
        expected_agents = ARM_SPEC[arm]["agents"]
        for row_index, row in enumerate(rows):
            if sorted(row, key=int) != expected_rounds:
                raise ValueError(f"{arm} row {row_index}: incomplete rounds")
            label = row["0"]["answer"]
            for round_name, record in row.items():
                if record["answer"] != label:
                    raise ValueError(f"{arm} row {row_index} round {round_name}: label changed")
                for key in ("final_answers", "final_answer_iscorr", "responses"):
                    if len(record[key]) != expected_agents:
                        raise ValueError(
                            f"{arm} row {row_index} round {round_name}: {key} agent count"
                        )
    reference = [row["0"]["answer"] for row in histories["dense"]]
    for arm, rows in histories.items():
        if [row["0"]["answer"] for row in rows] != reference:
            raise ValueError(f"task labels or order differ in {arm}")


def summarize(root, expected_rows):
    histories = {arm: load_arm(root, arm) for arm in ARM_SPEC}
    _validate(histories, expected_rows)
    result = {"passed": True, "tasks": expected_rows, "arms": {}}
    for arm, rows in histories.items():
        arm_result = {"rounds": {}}
        baseline = [bool(row["0"]["debate_answer_iscorr"]) for row in rows]
        for round_name in ARM_SPEC[arm]["rounds"]:
            aggregate = [bool(row[round_name]["debate_answer_iscorr"]) for row in rows]
            individual = [
                bool(value)
                for row in rows
                for value in row[round_name]["final_answer_iscorr"]
            ]
            answers = [
                value for row in rows for value in row[round_name]["final_answers"]
            ]
            item = {
                "aggregate_correct": sum(aggregate),
                "aggregate_accuracy": sum(aggregate) / expected_rows,
                "individual_correct": sum(individual),
                "individual_total": len(individual),
                "individual_accuracy": sum(individual) / len(individual),
                "empty_extractions": sum(value == "" for value in answers),
            }
            if round_name != "0":
                item["communication_transitions"] = {
                    "wrong_to_right": sum(not before and after for before, after in zip(baseline, aggregate)),
                    "right_to_wrong": sum(before and not after for before, after in zip(baseline, aggregate)),
                    "right_to_right": sum(before and after for before, after in zip(baseline, aggregate)),
                    "wrong_to_wrong": sum(not before and not after for before, after in zip(baseline, aggregate)),
                }
                item["paired_effect"] = bootstrap_difference(list(zip(baseline, aggregate)))
            arm_result["rounds"][round_name] = item
        result["arms"][arm] = arm_result

    result["contrasts"] = {}
    for arm in ("dense", "sparse", "central"):
        rounds = result["arms"][arm]["rounds"]
        best_round = max(rounds, key=lambda name: rounds[name]["aggregate_accuracy"])
        result["contrasts"][arm] = {
            "round0_vote_minus_round0_individual": (
                rounds["0"]["aggregate_accuracy"] - rounds["0"]["individual_accuracy"]
            ),
            "best_communication_round": best_round,
            "best_communication_minus_round0_vote": (
                rounds[best_round]["aggregate_accuracy"] - rounds["0"]["aggregate_accuracy"]
            ),
        }
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
