import json
import secrets
from pathlib import Path
from datetime import datetime, timezone

from utils.schemas import RunRecord


def new_run_id():
    """Sortable, collision-resistant identifier for a single run.

    The timestamp prefix keeps runs in chronological order on disk; the
    random suffix keeps two runs started in the same second distinct.
    """

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    return f"{stamp}-{secrets.token_hex(3)}"


class ExperimentLogger:

    def __init__(self, output_dir="data/outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_run(
        self,
        case_id,
        image_path,
        model,
        result,
        experiment_id="adhoc",
        max_iterations=None,
        prompt_hash=None,
        run_id=None,
        case_type=None,
        benchmark_version=None,
        benchmark_hash=None
    ):
        """Persist a run to <experiment_id>/<case_id>/<run_id>.json.

        Experiments are kept in separate trees and every run gets its own
        file, so neither re-running a case nor starting a new experiment can
        overwrite earlier results.
        """

        run_id = run_id or new_run_id()

        record = RunRecord.model_validate({
            "experiment_id": experiment_id,
            "case_id": case_id,
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image": str(image_path),
            "model": model,
            "case_type": case_type,
            "benchmark_version": benchmark_version,
            "benchmark_hash": benchmark_hash,
            "max_iterations": max_iterations,
            "prompt_hash": prompt_hash,
            **result
        })

        case_dir = self.output_dir / experiment_id / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        output_file = case_dir / f"{run_id}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                record.model_dump(mode="json"),
                f,
                indent=2,
                ensure_ascii=False
            )

        return output_file

    def list_runs(self, case_id, experiment_id="adhoc"):
        """Every saved run for a case within one experiment, oldest first."""

        case_dir = self.output_dir / experiment_id / case_id

        if not case_dir.exists():
            return []

        return sorted(case_dir.glob("*.json"))

    def list_experiments(self):
        """Experiment ids that currently have saved runs."""

        if not self.output_dir.exists():
            return []

        return sorted(
            d.name for d in self.output_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        )
