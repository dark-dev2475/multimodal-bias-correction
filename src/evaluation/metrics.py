import re
from collections import Counter
from typing import Dict, List

from pydantic import BaseModel, ConfigDict

from evaluation.ground_truth import CaseGroundTruth, GroundTruthLabel


class EvaluationLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    supported: bool
    problematic: bool
    bias_category: str
    severity: str


class ClassificationMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    true_positive: int
    false_positive: int
    false_negative: int
    precision: float | None
    recall: float | None
    f1: float | None


class GroundTruthMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    reason: str | None
    detection: ClassificationMetrics | None
    unsupported_claim_removal_rate: float | None
    supported_information_retention: float | None
    over_correction_rate: float | None
    correction_success_rate: float | None
    residual_problematic_claims: int | None


class VerifierDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    verification_passed: bool | None
    final_problematic_claims: int | None
    correction_iterations: int | None
    stop_reason: str | None


class CaseEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    ground_truth: GroundTruthMetrics
    verifier_diagnostics: VerifierDiagnostics


def _normalise_claim(claim: str) -> str:
    return re.sub(r"\s+", " ", claim.strip().casefold())


def _problematic(labels):
    return {
        _normalise_claim(label.claim)
        for label in labels
        if label.problematic
    }


def _supported(labels):
    return {
        _normalise_claim(label.claim)
        for label in labels
        if label.supported and not label.problematic
    }


def _classification(expected, predicted):
    true_positive = len(expected & predicted)
    false_positive = len(predicted - expected)
    false_negative = len(expected - predicted)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else None
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else None
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return ClassificationMetrics(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1
    )


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def evaluate_ground_truth(
    case: CaseGroundTruth,
    initial_labels: List[EvaluationLabel],
    final_labels: List[EvaluationLabel]
) -> GroundTruthMetrics:
    """Score independently annotated initial and final responses.

    Claim matching is deliberately exact after whitespace/case
    normalisation. A future semantic matcher must be independently validated
    before it is used for research metrics.
    """

    expected_problematic = _problematic(case.labels)
    expected_supported = _supported(case.labels)
    initial_problematic = _problematic(initial_labels)
    final_problematic = _problematic(final_labels)
    initial_supported = _supported(initial_labels)
    final_supported = _supported(final_labels)

    detection = _classification(expected_problematic, initial_problematic)
    removed = len(initial_problematic - final_problematic)
    retained = len(initial_supported & final_supported)
    over_corrected = len(initial_supported - final_supported)
    initially_present = len(initial_problematic)

    return GroundTruthMetrics(
        available=True,
        reason=None,
        detection=detection,
        unsupported_claim_removal_rate=_rate(removed, initially_present),
        supported_information_retention=_rate(
            retained, len(initial_supported)
        ),
        over_correction_rate=_rate(
            over_corrected, len(initial_supported)
        ),
        correction_success_rate=_rate(removed, initially_present),
        residual_problematic_claims=len(final_problematic & expected_problematic)
    )


def unavailable_ground_truth(reason):
    return GroundTruthMetrics(
        available=False,
        reason=reason,
        detection=None,
        unsupported_claim_removal_rate=None,
        supported_information_retention=None,
        over_correction_rate=None,
        correction_success_rate=None,
        residual_problematic_claims=None
    )


def evaluate_case(case, result):
    """Evaluate a result without treating verifier output as ground truth."""

    human = result.get("human_evaluation")
    if not human:
        ground_truth = unavailable_ground_truth(
            "Independent human_evaluation labels are required."
        )
    else:
        ground_truth = evaluate_ground_truth(
            case,
            [EvaluationLabel.model_validate(item)
             for item in human.get("initial_labels", [])],
            [EvaluationLabel.model_validate(item)
             for item in human.get("final_labels", [])]
        )

    pipeline = result.get("pipeline_result", result.get("result", {}))
    verification = pipeline.get("verification", {})
    claims = verification.get("claims", [])
    diagnostics = VerifierDiagnostics(
        available=bool(verification),
        verification_passed=pipeline.get("verification_passed"),
        final_problematic_claims=sum(
            1 for claim in claims if claim.get("problematic")
        ) if verification else None,
        correction_iterations=pipeline.get("correction_iterations"),
        stop_reason=pipeline.get("stop_reason")
    )

    return CaseEvaluation(
        case_id=case.id,
        ground_truth=ground_truth,
        verifier_diagnostics=diagnostics
    )


def compare_results(benchmark, results_by_system: Dict[str, dict]):
    """Create a machine-readable comparison without inventing labels."""

    cases = {case.id: case for case in benchmark.cases}
    comparison = {}

    for system, payload in results_by_system.items():
        evaluations = []
        for row in payload.get("cases", []):
            case = cases[row["case_id"]]
            evaluations.append(
                evaluate_case(case, row)
            )
        comparison[system] = {
            "benchmark_version": benchmark.benchmark_version,
            "benchmark_hash": benchmark.benchmark_hash,
            "cases": [item.model_dump(mode="json") for item in evaluations]
        }

    return comparison