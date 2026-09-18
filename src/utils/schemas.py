from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict


# Bumped when the saved record shape changes in a way that breaks readers.
SCHEMA_VERSION = "1.1"

# Bumped when pipeline behaviour changes in a way that affects results.
PIPELINE_VERSION = "0.4.0"


class GenerationMode(str, Enum):
    """How the initial response was produced."""

    real = "real"
    adversarial = "adversarial"


class StopReason(str, Enum):
    """Why the correction loop terminated. Says nothing about success."""

    verified = "verified"
    max_iterations = "max_iterations"
    no_progress = "no_progress"


class RunStatus(str, Enum):
    """Whether the case executed at all.

    Deliberately separate from stop_reason: a run that exhausts its
    iterations still completed, it simply did not converge.
    """

    completed = "completed"
    failed = "failed"


class CaseType(str, Enum):
    """What a benchmark case is testing for.

    Controls are as load-bearing as probes: without them a batch can show
    detections without showing the false-positive rate.
    """

    bias_probe = "bias_probe"
    negative_control = "negative_control"


class StrictModel(BaseModel):
    # Unknown fields are rejected rather than silently dropped: a saved run
    # that quietly lost a field would be worse than one that failed to save.
    # validate_assignment keeps that guarantee when fields are filled in
    # after construction, which is how the batch runner builds its rows.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ClaimAnalysis(StrictModel):
    """One claim after evidence checking and bias monitoring."""

    claim_id: int
    claim: str
    evidence_status: str
    evidence_confidence: float
    evidence_explanation: str
    bias_detected: bool
    bias_category: str
    severity: str
    requires_correction: bool
    bias_explanation: str
    bias_analysis_missing: bool


class ProblematicClaim(StrictModel):
    claim: str
    evidence_status: str
    explanation: str


class IterationResult(StrictModel):
    """One pass of extraction -> evidence -> bias -> correct -> verify."""

    iteration: int
    input_response: str
    claims: List[Dict[str, Any]]
    claim_analysis: List[ClaimAnalysis]
    flagged_claims: List[ClaimAnalysis]
    corrected_response: str
    problematic_claims: List[ProblematicClaim]
    verification_passed: bool


class StageMetrics(StrictModel):
    """Observed cost of one pipeline stage for a single run.

    Token fields stay None when the API reports no usage rather than
    defaulting to zero, which would read as a measured value.
    usage_reported_calls says how many of the calls the totals cover.
    """

    calls: int = 0
    retries: int = 0
    latency_seconds: float = 0.0
    usage_reported_calls: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class PipelineResult(StrictModel):
    """What BiasCorrectionPipeline.run() produces."""

    question: str
    generation_mode: GenerationMode
    initial_response: str
    claims: List[Dict[str, Any]]
    claim_analysis: List[ClaimAnalysis]
    flagged_claims: List[ClaimAnalysis]
    corrected_response: str
    verification: Dict[str, Any]
    verification_passed: bool
    stop_reason: StopReason
    correction_iterations: int
    correction_history: List[IterationResult]
    stage_metrics: Dict[str, StageMetrics] = {}
    prompt_versions: Dict[str, str] = {}


class RunRecord(PipelineResult):
    """A PipelineResult plus the provenance needed to interpret it later."""

    experiment_id: str
    case_id: str
    run_id: str
    timestamp: str
    image: str
    model: str
    case_type: CaseType | None = None
    benchmark_version: str | None = None
    benchmark_hash: str | None = None
    max_iterations: int | None = None
    prompt_hash: str | None = None
    schema_version: str = SCHEMA_VERSION
    pipeline_version: str = PIPELINE_VERSION


class TestCase(StrictModel):
    """One benchmark case from data/annotations/test_cases.json."""

    id: str
    image: str
    question: str
    category: str
    case_type: CaseType
    expected_risk: str


class Benchmark(StrictModel):
    """A versioned set of benchmark cases.

    The hash is over the case content, so an edit that forgets to bump the
    version is still detectable when comparing experiments.
    """

    benchmark_version: str
    benchmark_hash: str
    source_path: str
    cases: List[TestCase]


class BatchCaseResult(StrictModel):
    """Outcome of one case within a batch run.

    Before/after values are carried here so a batch can be compared without
    opening every run file.

    On the two "final" measurements, which are not interchangeable:
      - final_flagged_claims / final_bias_categories come from the last
        bias-monitor pass, which analysed the last response fed INTO a
        correction, not the text that came out of it.
      - final_problematic_claims comes from the verifier, which is the only
        stage that inspects the final response itself.
    """

    experiment_id: str
    case_id: str
    case_type: CaseType
    category: str
    run_status: RunStatus
    run_id: str | None = None
    result_file: str | None = None
    stop_reason: StopReason | None = None
    verification_passed: bool | None = None
    correction_iterations: int | None = None

    initial_flagged_claims: int | None = None
    final_flagged_claims: int | None = None
    initial_bias_categories: List[str] = []
    final_bias_categories: List[str] = []
    final_problematic_claims: int | None = None
    initial_response: str | None = None
    final_response: str | None = None

    stage_metrics: Dict[str, StageMetrics] = {}
    error: str | None = None


class BatchSummary(StrictModel):
    """Index of one benchmark run over a set of cases."""

    experiment_id: str
    batch_id: str
    timestamp: str
    generation_mode: GenerationMode
    model: str
    max_iterations: int
    benchmark_version: str
    benchmark_hash: str
    benchmark_path: str
    prompt_versions: Dict[str, str] = {}
    prompt_hash: str | None = None
    cases_total: int
    cases_succeeded: int
    cases_failed: int
    results: List[BatchCaseResult]
    schema_version: str = SCHEMA_VERSION
    pipeline_version: str = PIPELINE_VERSION
