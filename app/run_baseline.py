import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from evaluation.baselines import BaselineSystem, create_system
from utils.benchmark import DEFAULT_CASES_PATH, load_benchmark, resolve_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one comparable baseline over benchmark cases."
    )
    parser.add_argument("--system", choices=[s.value for s in BaselineSystem],
                        required=True)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--max-iterations", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark = load_benchmark(args.cases)
    selected = benchmark.cases

    if args.only:
        wanted = set(args.only)
        unknown = wanted - {case.id for case in selected}
        if unknown:
            raise SystemExit(f"Unknown case ids: {', '.join(sorted(unknown))}")
        selected = [case for case in selected if case.id in wanted]

    runner = create_system(args.system, args.max_iterations)
    rows = []

    for case in selected:
        started = time.monotonic()
        result = runner.run(resolve_image(case), case.question)
        if isinstance(result, tuple):
            summary, pipeline_result = result
        else:
            summary, pipeline_result = result, None

        row = {
            "case_id": case.id,
            "category": case.category,
            "case_type": case.case_type.value,
            "benchmark_version": benchmark.benchmark_version,
            "benchmark_hash": benchmark.benchmark_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "result": summary.as_dict()
        }
        if pipeline_result is not None:
            row["pipeline_result"] = pipeline_result
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "system": args.system,
        "model": rows[0]["result"]["model"] if rows else None,
        "benchmark_version": benchmark.benchmark_version,
        "benchmark_hash": benchmark.benchmark_hash,
        "cases": rows
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(rows)} {args.system} result(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())