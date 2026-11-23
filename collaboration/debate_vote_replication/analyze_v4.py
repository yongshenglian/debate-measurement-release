"""Final Debate-or-Vote audit with within-arm causal diagnostics and pairing checks."""
import argparse
import json
from pathlib import Path

from analyze_v3 import ARM_SPEC, load_arm, summarize as summarize_v3


def summarize(root, expected_rows):
    result = summarize_v3(root, expected_rows)
    histories = {arm: load_arm(root, arm) for arm in ARM_SPEC}

    # Replace v3's descriptive best-round block: round zero is not communication.
    result["contrasts"] = {}
    for arm in ("dense", "sparse", "central"):
        rounds = result["arms"][arm]["rounds"]
        later = [str(i) for i in range(1, 6)]
        best_later = max(later, key=lambda name: rounds[name]["aggregate_accuracy"])
        contrast = {
            "best_later_round": best_later,
            "best_later_minus_own_round0": (
                rounds[best_later]["aggregate_accuracy"] - rounds["0"]["aggregate_accuracy"]
            ),
        }
        if arm != "central":
            contrast["round0_vote_minus_round0_individual"] = (
                rounds["0"]["aggregate_accuracy"] - rounds["0"]["individual_accuracy"]
            )
        result["contrasts"][arm] = contrast

    # Candidate-coverage and readout decomposition are defined for five scored
    # proposals. Centralized mode scores only agent zero in the released code.
    for arm in ("dense", "sparse"):
        rows = histories[arm]
        proposal_hit = [any(map(bool, row["0"]["final_answer_iscorr"])) for row in rows]
        initial_vote = [bool(row["0"]["debate_answer_iscorr"]) for row in rows]
        result["arms"][arm]["initial_pool"] = {
            "proposal_hit_count": sum(proposal_hit),
            "proposal_hit_rate": sum(proposal_hit) / expected_rows,
            "all_wrong_count": expected_rows - sum(proposal_hit),
            "recoverable_headroom_count": sum(hit and not vote for hit, vote in zip(proposal_hit, initial_vote)),
        }
        for round_name in [str(i) for i in range(1, 6)]:
            final = [bool(row[round_name]["debate_answer_iscorr"]) for row in rows]
            headroom = [hit and not vote for hit, vote in zip(proposal_hit, initial_vote)]
            recovered = sum(can_recover and after for can_recover, after in zip(headroom, final))
            new_correct = sum((not hit) and after for hit, after in zip(proposal_hit, final))
            damaged = sum(vote and not after for vote, after in zip(initial_vote, final))
            result["arms"][arm]["rounds"][round_name]["coverage_readout_decomposition"] = {
                "recoverable_headroom": sum(headroom),
                "recovered": recovered,
                "new_correct_beyond_initial_coverage": new_correct,
                "damaged_initial_vote": damaged,
                "identity_check": {
                    "observed_gain_count": sum(final) - sum(initial_vote),
                    "decomposed_gain_count": recovered + new_correct - damaged,
                    "passed": sum(final) - sum(initial_vote) == recovered + new_correct - damaged,
                },
            }

    # Separate processes use the same seed but need not share sampled proposals.
    reference = histories["dense"]
    result["round0_cross_arm_identity"] = {}
    for other in ("sparse", "central"):
        rows = histories[other]
        result["round0_cross_arm_identity"][f"dense_vs_{other}"] = {
            "tasks": expected_rows,
            "exact_response_rows": sum(
                reference[i]["0"]["responses"] == rows[i]["0"]["responses"]
                for i in range(expected_rows)
            ),
            "interpretation": "independent stochastic proposal pools; use within-arm round-zero contrasts",
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
