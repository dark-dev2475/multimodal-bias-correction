import json

import pytest

from evaluation.ground_truth import load_ground_truth
from utils.benchmark import DEFAULT_CASES_PATH, load_benchmark


def test_ground_truth_requires_a_complete_matching_artifact(tmp_path):
    benchmark = load_benchmark(DEFAULT_CASES_PATH)
    path = tmp_path / "ground_truth.json"
    path.write_text(
        json.dumps({
            "benchmark_version": benchmark.benchmark_version,
            "benchmark_hash": benchmark.benchmark_hash,
            "cases": []
        }),
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="case ids"):
        load_ground_truth(path, benchmark)


def test_ground_truth_labels_are_separate_from_model_runs(tmp_path):
    benchmark = load_benchmark(DEFAULT_CASES_PATH)
    cases = []

    for case in benchmark.cases:
        cases.append({
            "case_id": case.id,
            "safe_observations": ["A human-annotated observable fact."],
            "labels": [{
                "claim": "The person is a patient.",
                "supported": False,
                "problematic": True,
                "bias_category": "medical_assumption",
                "severity": "high"
            }],
            "annotator": "human-1",
            "annotation_version": "v1"
        })

    path = tmp_path / "ground_truth.json"
    path.write_text(json.dumps({
        "benchmark_version": benchmark.benchmark_version,
        "benchmark_hash": benchmark.benchmark_hash,
        "cases": cases
    }), encoding="utf-8")

    loaded = load_ground_truth(path, benchmark)

    assert loaded.cases[0].case_id == benchmark.cases[0].id
    assert loaded.cases[0].labels[0].problematic is True