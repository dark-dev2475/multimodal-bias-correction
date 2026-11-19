from scripts.demonstrate_v2 import build_demonstration, make_report
from research_v2.evaluation import paired_bootstrap
import pytest


def test_demonstration_reports_computed_ablation_differences():
    data = build_demonstration()
    rows = {r["id"]: r for r in data["comparisons"]}
    assert data["network_calls"] == 0
    assert rows["cue_and_propagation"]["full"]["repair"]["selected_ids"] == ["c3", "c4"]
    assert rows["cue_and_propagation"]["disabled"]["repair"]["selected_ids"] == ["c1", "c2", "c3", "c4"]
    assert "c2" in rows["inherited_premise"]["disabled"]["repair"]["selected_ids"]
    assert "c2" not in rows["inherited_premise"]["full"]["repair"]["selected_ids"]
    assert "c1" in rows["placebo_sensitive_control"]["full"]["repair"]["selected_ids"]
    assert "c1" not in rows["placebo_sensitive_control"]["disabled"]["repair"]["selected_ids"]
    assert rows["exact_optimization"]["full"]["repair"]["objective"] == 1.6
    assert rows["exact_optimization"]["disabled"]["repair"]["objective"] == 1.1
    assert "not empirical superiority" in make_report(data)


def test_grouped_bootstrap_resamples_whole_correlated_groups():
    left = {"a": 1, "b": 1, "c": 0}
    right = dict.fromkeys(left, 0)
    result = paired_bootstrap(left, right, groups={"a": "x", "b": "x", "c": "y"})
    assert result["n_pairs"] == 3 and result["n_clusters"] == 2
    assert result["resampling_unit"] == "leakage_group"
    assert result["mean_difference"] == 2 / 3
    assert result["ci95"] == [0, 1]
    one_group = paired_bootstrap(left, right, groups=dict.fromkeys(left, "same"))
    assert one_group["ci95"] is None
    with pytest.raises(ValueError, match="Groups"):
        paired_bootstrap(left, right, groups={"a": "x"})


def test_review_evaluator_preserves_groups_and_rejects_partial_group_provenance():
    from scripts.evaluate_v2_reviews import evaluate
    forms, key = [], {}
    for cid, group in (("a", "same"), ("b", "same"), ("c", "different")):
        for system in ("raw", "full"):
            token = cid + system
            key[token] = dict(example_id=cid, system=system, status="verified_subset", leakage_group=group)
            forms.append(dict(review_id=token, annotator="unit-test-not-real-labels",
                              any_unsupported_social_assumption=system == "raw",
                              task_correct=True, useful_information_score_0_to_1=1))
    report = evaluate(forms, key, "human")
    result = report["paired_comparisons"]["full - raw"]["metrics"]["bias_rate"]
    assert result["n_clusters"] == 2 and result["n_pairs"] == 3
    assert result["mean_difference"] == -1
    key["araw"]["leakage_group"] = None
    with pytest.raises(ValueError, match="leakage_group"):
        evaluate(forms, key, "human")
