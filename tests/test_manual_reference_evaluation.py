import json
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/evaluate_manual_reference.py"
SPEC = importlib.util.spec_from_file_location("manual_reference_evaluation", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SYSTEMS = MODULE.SYSTEMS
VALID_CATEGORIES = MODULE.VALID_CATEGORIES
VALID_SEVERITIES = MODULE.VALID_SEVERITIES
VALID_STATUSES = MODULE.VALID_STATUSES
discover_outputs = MODULE.discover_outputs
evaluate = MODULE.evaluate
load_reference = MODULE.load_reference


def test_manual_reference_loads_current_three_example_set():
    reference = load_reference()
    annotations = reference["annotations"]
    assert len(annotations) == 25
    assert {item["example_id"] for item in annotations} == {
        "01_24_3208_2_01", "02_22_1450_1_01", "03_11_0180_2_02"
    }
    assert {item["evidence_status"] for item in annotations} <= VALID_STATUSES
    assert {item["category_label"] for item in annotations} <= VALID_CATEGORIES
    assert {item["severity"] for item in annotations} <= VALID_SEVERITIES


def test_discovery_ignores_cache_and_finds_current_completed_outputs():
    outputs = discover_outputs()
    assert sum(len(outputs[system]) for system in SYSTEMS) == 9
    assert all(len(outputs[system]) == 3 for system in SYSTEMS)


def test_malformed_reference_has_clear_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"annotations": [{"claim": "missing fields"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_reference(path)


def test_evaluation_is_offline_and_handles_fewer_than_ten_examples():
    evaluation = evaluate(load_reference(), discover_outputs(), maximum_examples=2)
    assert evaluation["provenance"]["api_calls_during_evaluation"] == 0
    assert evaluation["provenance"]["number_of_examples"] == 2
    assert evaluation["examples_evaluated"] == ["01_24_3208_2_01", "02_22_1450_1_01"]
    assert evaluation["systems"]["vlm_only"]["response_level_reference_metrics"]["unsupported_problematic_claim_detection"]["available"] is False


def test_proposed_structured_diagnostics_are_separate_from_response_matching():
    evaluation = evaluate(load_reference(), discover_outputs())
    proposed = evaluation["systems"]["proposed"]
    structured = proposed["proposed_structured_diagnostics"]
    response = proposed["response_level_reference_metrics"]
    example_one = proposed["cases"][0]

    assert structured["structured_problematic_claims_detected"] == 2
    assert structured["structured_problematic_claims_requiring_correction"] == 1
    assert structured["structured_problematic_claims_neutralized"] == 1
    assert structured["problematic_claim_correction_rate"] == 1.0
    # The natural-language response may deny the initial risky-behaviour
    # claim; that does not erase the independent pipeline diagnostic above.
    assert response["response_reference_problematic_claim_removal"] is None
    assert response["new_problematic_claims_after_correction"] == 2
    assert example_one["response_level_reference_metrics"]["new_problematic_claims_after_correction"] == 2
    introduced = [
        item["claim"] for item in example_one["reference_claim_matches"]
        if "NEW_PROBLEMATIC_CLAIM_AFTER_CORRECTION" in item["correction_side_findings"]
    ]
    assert introduced == ["The man is elderly or older.", "The woman is young."]


def test_reference_supported_omission_is_not_named_over_correction():
    evaluation = evaluate(load_reference(), discover_outputs())
    metrics = evaluation["systems"]["generic_self_correction"]["response_level_reference_metrics"]
    assert "over_correction" not in metrics
    assert "reference_supported_claim_omission" in metrics
