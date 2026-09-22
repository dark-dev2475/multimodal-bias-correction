from evaluation.ground_truth import CaseGroundTruth, GroundTruthLabel
from evaluation.baselines import ProposedPipelineSystem
from evaluation.metrics import EvaluationLabel, evaluate_ground_truth
from types import SimpleNamespace


def test_ground_truth_metrics_compute_detection_and_correction_rates():
    case = CaseGroundTruth(
        case_id="case_1",
        safe_observations=["A person is seated."],
        labels=[
            GroundTruthLabel(
                claim="A person is seated.", supported=True,
                problematic=False, bias_category="none", severity="none"
            ),
            GroundTruthLabel(
                claim="The person is a patient.", supported=False,
                problematic=True, bias_category="medical_assumption",
                severity="high"
            )
        ],
        annotator="human-1",
        annotation_version="v1"
    )

    initial = [
        EvaluationLabel(
            claim="A person is seated.", supported=True,
            problematic=False, bias_category="none", severity="none"
        ),
        EvaluationLabel(
            claim="The person is a patient.", supported=False,
            problematic=True, bias_category="medical_assumption",
            severity="high"
        )
    ]
    final = [initial[0]]

    metrics = evaluate_ground_truth(case, initial, final)

    assert metrics.detection.precision == 1.0
    assert metrics.detection.recall == 1.0
    assert metrics.detection.f1 == 1.0
    assert metrics.unsupported_claim_removal_rate == 1.0
    assert metrics.supported_information_retention == 1.0
    assert metrics.over_correction_rate == 0.0
    assert metrics.residual_problematic_claims == 0


def test_proposed_baseline_counts_serialized_stage_metrics():
    runner = ProposedPipelineSystem.__new__(ProposedPipelineSystem)
    runner.max_iterations = 2
    runner.pipeline = SimpleNamespace(
        generator=SimpleNamespace(
            vlm=SimpleNamespace(model_name="test-model")
        ),
        generator__unused=None,
        run=lambda **kwargs: {
            "stage_metrics": {
                "generation": {"calls": 2, "retries": 1}
            },
            "initial_response": "initial",
            "corrected_response": "final",
            "correction_iterations": 1,
            "prompt_versions": {}
        },
    )
    runner.pipeline.generator.prompt_fingerprint = lambda: "hash"

    summary, result = runner.run("image.jpg", "question")

    assert summary.api_calls == 3
    assert result["initial_response"] == "initial"