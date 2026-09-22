import json
from pathlib import Path
from typing import List

from pydantic import BaseModel, ConfigDict

from utils.schemas import Benchmark


class GroundTruthLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    supported: bool
    problematic: bool
    bias_category: str
    severity: str


class CaseGroundTruth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    safe_observations: List[str]
    labels: List[GroundTruthLabel]
    annotator: str
    annotation_version: str


class GroundTruthSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark_version: str
    benchmark_hash: str
    cases: List[CaseGroundTruth]


def load_ground_truth(path, benchmark: Benchmark) -> GroundTruthSet:
    """Load human labels and verify they describe this benchmark exactly.

    This artifact is intentionally separate from model run records. A model
    result cannot silently become an evaluation label by being copied into it.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    ground_truth = GroundTruthSet.model_validate(data)

    if ground_truth.benchmark_version != benchmark.benchmark_version:
        raise ValueError(
            "Ground truth benchmark version does not match the benchmark: "
            f"{ground_truth.benchmark_version!r} != "
            f"{benchmark.benchmark_version!r}"
        )

    if ground_truth.benchmark_hash != benchmark.benchmark_hash:
        raise ValueError(
            "Ground truth benchmark hash does not match the benchmark. "
            "Re-annotate after changing benchmark cases."
        )

    expected_ids = {case.id for case in benchmark.cases}
    actual_ids = {case.case_id for case in ground_truth.cases}

    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"Ground truth case ids do not match benchmark; "
            f"missing={missing}, unexpected={unexpected}"
        )

    return ground_truth