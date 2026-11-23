from analyze_v4 import summarize
from test_analyze_v3 import fixture


def test_best_round_excludes_round_zero_and_decomposition_closes(tmp_path):
    fixture(tmp_path)
    value = summarize(tmp_path, 4)
    assert value["contrasts"]["dense"]["best_later_round"] in {"1", "2", "3", "4", "5"}
    decomposition = value["arms"]["dense"]["rounds"]["1"]["coverage_readout_decomposition"]
    assert decomposition["identity_check"]["passed"] is True
    assert value["arms"]["dense"]["initial_pool"] == {
        "proposal_hit_count": 2,
        "proposal_hit_rate": 0.5,
        "all_wrong_count": 2,
        "recoverable_headroom_count": 0,
    }


def test_cross_arm_identity_is_reported_not_assumed(tmp_path):
    fixture(tmp_path)
    value = summarize(tmp_path, 4)
    assert value["round0_cross_arm_identity"]["dense_vs_sparse"]["exact_response_rows"] == 4
    assert "within-arm" in value["round0_cross_arm_identity"]["dense_vs_central"]["interpretation"]
