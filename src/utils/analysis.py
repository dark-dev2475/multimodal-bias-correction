import json
from pathlib import Path
from collections import Counter
from typing import Dict, List

from utils.schemas import CaseType, StrictModel


SUPPORTED = "SUPPORTED"


class CaseMetrics(StrictModel):
    """Before/after measurements for one case.

    Two different "after" numbers are kept deliberately:

      final_flagged
        from the last bias-monitor pass, which analysed the last response fed
        INTO a correction, not the text that came out of it.

      final_problematic
        from the verifier, the only stage that inspects the final response.

    final_problematic is the honest end-state measure; final_flagged is
    useful for seeing what the monitor was still catching mid-loop.
    """

    case_id: str
    case_type: CaseType
    category: str
    run_id: str
    model: str
    stop_reason: str
    verification_passed: bool
    correction_iterations: int

    initial_flagged: int
    final_flagged: int
    final_problematic: int
    flags_resolved: int

    initial_categories: List[str]
    final_categories: List[str]

    supported_initial: int
    supported_final: int

    initial_chars: int
    final_chars: int
    length_ratio: float

    is_false_positive: bool
    residual_after_correction: bool


class GroupMetrics(StrictModel):
    """Aggregates over one case type."""

    cases: int = 0
    total_initial_flagged: int = 0
    total_final_problematic: int = 0
    flags_resolved: int = 0
    resolution_rate: float | None = None
    verified: int = 0
    verified_rate: float | None = None
    mean_iterations: float | None = None
    mean_length_ratio: float | None = None


class ExperimentMetrics(StrictModel):
    """Everything Phase 5 can say about one experiment."""

    experiment_id: str
    runs_analysed: int
    cases: int
    distinct_images: int
    models: List[str]
    benchmark_versions: List[str]
    benchmark_hashes: List[str]
    prompt_hashes: List[str]
    pipeline_versions: List[str]

    probes: GroupMetrics
    controls: GroupMetrics

    stop_reasons: Dict[str, int]
    category_initial: Dict[str, int]
    category_final: Dict[str, int]

    missing_cases: List[str]
    caveats: List[str]

    per_case: List[CaseMetrics]


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_experiment_runs(output_dir, experiment_id, latest_only=True):
    """Saved runs for one experiment.

    Run ids are timestamp-prefixed, so sorting gives chronological order and
    the last entry is the most recent. Only the latest run per case is used
    by default: aggregating repeated runs of the same case would weight it
    more heavily than the others.
    """

    experiment_dir = Path(output_dir) / experiment_id

    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"No such experiment: {experiment_dir}"
        )

    runs = []

    for case_dir in sorted(experiment_dir.iterdir()):

        if not case_dir.is_dir() or case_dir.name.startswith("_"):
            continue

        files = sorted(case_dir.glob("*.json"))

        for path in (files[-1:] if latest_only else files):
            runs.append(json.loads(path.read_text(encoding="utf-8")))

    return runs


def load_batch_case_ids(output_dir, experiment_id):
    """Every case id any batch in this experiment attempted.

    Compared against saved runs to spot cases that failed every time and so
    left no run file behind, which would otherwise vanish from the analysis.
    """

    batch_dir = Path(output_dir) / experiment_id / "_batches"

    if not batch_dir.exists():
        return set()

    attempted = set()

    for path in sorted(batch_dir.glob("*.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        attempted.update(
            row["case_id"] for row in summary.get("results", [])
        )

    return attempted


# ---------------------------------------------------------------------
# Per-case metrics
# ---------------------------------------------------------------------

def _categories(flagged):

    return sorted({
        item.get("bias_category")
        for item in flagged
        if item.get("bias_category") and item["bias_category"] != "none"
    })


def _supported(analysis):

    return sum(
        1 for item in analysis
        if item.get("evidence_status") == SUPPORTED
    )


def case_metrics(run):

    history = run.get("correction_history") or []
    first = history[0] if history else {}
    last = history[-1] if history else {}

    initial_flagged = len(first.get("flagged_claims", []))
    final_flagged = len(last.get("flagged_claims", []))

    final_problematic = sum(
        1 for claim in run.get("verification", {}).get("claims", [])
        if claim.get("problematic")
    )

    initial_text = run.get("initial_response", "")
    final_text = run.get("corrected_response", "")

    case_type = CaseType(run.get("case_type") or CaseType.bias_probe)

    return CaseMetrics(
        case_id=run["case_id"],
        case_type=case_type,
        category=run.get("category", ""),
        run_id=run["run_id"],
        model=run.get("model", ""),
        stop_reason=run["stop_reason"],
        verification_passed=run["verification_passed"],
        correction_iterations=run["correction_iterations"],

        initial_flagged=initial_flagged,
        final_flagged=final_flagged,
        final_problematic=final_problematic,
        flags_resolved=initial_flagged - final_problematic,

        initial_categories=_categories(first.get("flagged_claims", [])),
        final_categories=_categories(last.get("flagged_claims", [])),

        supported_initial=_supported(first.get("claim_analysis", [])),
        supported_final=_supported(last.get("claim_analysis", [])),

        initial_chars=len(initial_text),
        final_chars=len(final_text),
        length_ratio=(
            round(len(final_text) / len(initial_text), 4)
            if initial_text else 0.0
        ),

        # A control case should never have been flagged in the first place.
        is_false_positive=(
            case_type is CaseType.negative_control and initial_flagged > 0
        ),
        residual_after_correction=final_problematic > 0
    )


def _group(rows):

    group = GroupMetrics(cases=len(rows))

    if not rows:
        return group

    group.total_initial_flagged = sum(r.initial_flagged for r in rows)
    group.total_final_problematic = sum(r.final_problematic for r in rows)
    group.flags_resolved = (
        group.total_initial_flagged - group.total_final_problematic
    )

    if group.total_initial_flagged:
        group.resolution_rate = round(
            group.flags_resolved / group.total_initial_flagged, 4
        )

    group.verified = sum(1 for r in rows if r.verification_passed)
    group.verified_rate = round(group.verified / len(rows), 4)
    group.mean_iterations = round(
        sum(r.correction_iterations for r in rows) / len(rows), 4
    )
    group.mean_length_ratio = round(
        sum(r.length_ratio for r in rows) / len(rows), 4
    )

    return group


# ---------------------------------------------------------------------
# Experiment metrics
# ---------------------------------------------------------------------

def _caveats(runs, per_case, distinct_images, models, hashes, missing):
    """Conditions that limit what these numbers can support.

    Emitted with the results rather than left to the reader to remember.
    """

    notes = []

    if len(hashes) > 1:
        notes.append(
            f"Runs span {len(hashes)} benchmark versions ({', '.join(hashes)}). "
            f"These cases are not directly comparable."
        )

    if len(models) > 1:
        notes.append(
            f"Runs span {len(models)} models ({', '.join(models)}). "
            f"Differences may be model effects rather than pipeline effects."
        )

    if distinct_images <= 1:
        notes.append(
            f"All cases use {distinct_images} image. Results describe "
            f"question-framing sensitivity, not visual generalisation."
        )

    if len(per_case) < 10:
        notes.append(
            f"Only {len(per_case)} cases analysed. Too few for rates to be "
            f"stable; treat percentages as illustrative."
        )

    # Every stage currently runs on the same model, so the system is both
    # author and judge of its own output.
    if len(models) == 1:
        notes.append(
            f"Generation and every judging stage use the same model "
            f"({models[0]}). The pipeline is grading its own work; an "
            f"independent judge model would make results defensible."
        )

    if not any(r.case_type is CaseType.negative_control for r in per_case):
        notes.append(
            "No negative-control case in this experiment. False-positive "
            "rate cannot be estimated."
        )

    if missing:
        notes.append(
            f"{len(missing)} case(s) were attempted but have no saved run "
            f"({', '.join(sorted(missing))}); they likely failed every time."
        )

    return notes


def summarise(experiment_id, runs, attempted_case_ids=frozenset()):

    per_case = [case_metrics(run) for run in runs]

    probes = [r for r in per_case if r.case_type is CaseType.bias_probe]
    controls = [
        r for r in per_case if r.case_type is CaseType.negative_control
    ]

    models = sorted({r.model for r in per_case if r.model})
    images = {run.get("image") for run in runs if run.get("image")}
    hashes = sorted({
        run.get("benchmark_hash") for run in runs
        if run.get("benchmark_hash")
    })

    category_initial = Counter()
    category_final = Counter()

    for row in per_case:
        category_initial.update(row.initial_categories)
        category_final.update(row.final_categories)

    missing = sorted(attempted_case_ids - {r.case_id for r in per_case})

    return ExperimentMetrics(
        experiment_id=experiment_id,
        runs_analysed=len(runs),
        cases=len({r.case_id for r in per_case}),
        distinct_images=len(images),
        models=models,
        benchmark_versions=sorted({
            run.get("benchmark_version") for run in runs
            if run.get("benchmark_version")
        }),
        benchmark_hashes=hashes,
        prompt_hashes=sorted({
            run.get("prompt_hash") for run in runs if run.get("prompt_hash")
        }),
        pipeline_versions=sorted({
            run.get("pipeline_version") for run in runs
            if run.get("pipeline_version")
        }),

        probes=_group(probes),
        controls=_group(controls),

        stop_reasons=dict(Counter(r.stop_reason for r in per_case)),
        category_initial=dict(category_initial),
        category_final=dict(category_final),

        missing_cases=missing,
        caveats=_caveats(
            runs, per_case, len(images), models, hashes, missing
        ),
        per_case=per_case
    )
