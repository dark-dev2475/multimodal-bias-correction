import json
import hashlib
from pathlib import Path

from utils.schemas import Benchmark, CaseType, TestCase


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CASES_PATH = (
    PROJECT_ROOT / "data" / "annotations" / "test_cases.json"
)


def resolve_image(case, root=PROJECT_ROOT):
    """Absolute path to a case's image.

    Case files store repo-relative paths so they stay portable.
    """

    path = Path(case.image)

    return path if path.is_absolute() else root / path


def benchmark_hash(cases):
    """Content hash over the cases themselves.

    Catches a benchmark edited without bumping its version, which would
    otherwise make two experiments look comparable when they are not.
    """

    payload = json.dumps(
        [case.model_dump(mode="json") for case in cases],
        sort_keys=True,
        ensure_ascii=False
    )

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def load_benchmark(
    path=DEFAULT_CASES_PATH,
    root=PROJECT_ROOT,
    require_images=True
):
    """Load and validate a versioned benchmark file.

    Images are checked up front because a batch that dies on case 40 of 50
    has already spent the API budget for the first 39.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Test case file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(
            f"{path} must be a JSON object with 'benchmark_version' and "
            f"'cases', got {type(data).__name__}"
        )

    version = data.get("benchmark_version")

    if not version:
        raise ValueError(
            f"{path} is missing 'benchmark_version'. Versioning the "
            f"benchmark is what keeps past experiments comparable."
        )

    entries = data.get("cases")

    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} must contain a non-empty 'cases' list")

    cases = [TestCase.model_validate(entry) for entry in entries]

    # Duplicate ids would make two different cases share an output
    # directory, silently mixing their runs together.
    seen = {}
    for case in cases:
        seen[case.id] = seen.get(case.id, 0) + 1

    duplicates = sorted(i for i, n in seen.items() if n > 1)

    if duplicates:
        raise ValueError(
            f"Duplicate case ids in {path}: {', '.join(duplicates)}"
        )

    if require_images:
        missing = [
            f"{case.id} -> {resolve_image(case, root)}"
            for case in cases
            if not resolve_image(case, root).exists()
        ]

        if missing:
            raise FileNotFoundError(
                "Test cases reference images that do not exist:\n  "
                + "\n  ".join(missing)
            )

    # A benchmark with no control can show detections but not the
    # false-positive rate, so refuse to run one by accident.
    if not any(case.case_type is CaseType.negative_control for case in cases):
        raise ValueError(
            f"{path} has no negative_control case. At least one is "
            f"required to measure false positives."
        )

    return Benchmark(
        benchmark_version=version,
        benchmark_hash=benchmark_hash(cases),
        source_path=str(path),
        cases=cases
    )
