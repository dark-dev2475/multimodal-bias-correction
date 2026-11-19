import json
from types import SimpleNamespace

import pytest

from evaluation.automatic_annotator import (
    ANNOTATOR_PROMPT,
    ANNOTATOR_PROMPT_VERSION,
    AutomaticAnnotation,
    AutomaticAnnotator,
    annotation_cache_key,
    annotation_set,
    validate_annotation_coverage,
    validate_annotations,
)
from evaluation.metrics import evaluate_llm_reference_annotations


def _annotation(**overrides):
    values = {
        "example_id": "example-1",
        "system": "proposed",
        "benchmark_category": "Age",
        "stage": "initial",
        "claim": "A person is seated.",
        "evidence_status": "SUPPORTED",
        "problematic": False,
        "category": "none",
        "severity": "none",
        "explanation": "The posture is visible.",
        "annotator_model": "test-model",
        "annotator_prompt_version": ANNOTATOR_PROMPT_VERSION,
        "annotated_at": "2026-09-29T00:00:00+00:00",
    }
    values.update(overrides)
    return AutomaticAnnotation.model_validate(values)


def test_annotation_schema_normalizes_allowed_statuses():
    annotation = _annotation(evidence_status="UNSUPPORTED")
    assert annotation.evidence_status == "NOT_ESTABLISHED"
    result = annotation_set([annotation], "test-model")
    assert result.annotation_type == "llm_generated_reference"
    assert json.loads(result.model_dump_json())["annotations"][0]["claim"]


def test_duplicate_and_invalid_annotations_are_rejected():
    annotation = _annotation()
    with pytest.raises(ValueError, match="Duplicate"):
        validate_annotations([annotation, annotation])
    with pytest.raises(ValueError):
        _annotation(category="not_a_category")
    with pytest.raises(ValueError):
        _annotation(severity="critical")
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_annotation_coverage(
            {("example-1", "proposed", "initial", "missing")},
            [annotation],
        )


def test_cache_key_separates_claims_and_models():
    first = annotation_cache_key("one", "one.jpg", "A person is seated.", "model-a")
    second = annotation_cache_key("one", "one.jpg", "A person is standing.", "model-a")
    third = annotation_cache_key("one", "one.jpg", "A person is seated.", "model-b")
    assert first != second
    assert first != third


def test_annotator_prompt_does_not_include_pipeline_labels():
    assert "bias_monitor" not in ANNOTATOR_PROMPT
    assert "evidence_checker" not in ANNOTATOR_PROMPT
    assert "benchmark_category" not in ANNOTATOR_PROMPT
    assert "CLAIM:" in ANNOTATOR_PROMPT


def test_annotator_uses_cache_and_serializes_independent_result():
    annotator = AutomaticAnnotator.__new__(AutomaticAnnotator)
    annotator.model_name = "test-model"
    annotator.cache = SimpleNamespace(
        hits=0,
        misses=0,
        get=lambda namespace, key: None,
        set=lambda namespace, key, value: None,
    )
    annotator.vlm = SimpleNamespace(
        calls=[],
        complete=lambda prompt, **kwargs: json.dumps({
            "evidence_status": "SUPPORTED",
            "problematic": False,
            "category": "none",
            "severity": "none",
            "explanation": "Visible.",
        }),
    )
    annotation, from_cache = annotator.annotate(
        "example-1", "image.jpg", "proposed", "Age", "initial",
        "A person is seated."
    )
    assert from_cache is False
    assert annotation.system == "proposed"
    assert annotation.benchmark_category == "Age"


def test_llm_reference_metrics_are_explicitly_labeled():
    initial = [{
        "claim": "The person is a patient.",
        "evidence_status": "NOT_ESTABLISHED",
        "problematic": True,
        "category": "medical_assumption",
        "severity": "medium",
    }]
    final = []
    metrics = evaluate_llm_reference_annotations(initial, final)
    assert metrics["evaluation_type"] == "llm-reference"
    assert metrics["human_annotation_available"] is False
    assert metrics["unsupported_claim_removal_rate"] == 1.0