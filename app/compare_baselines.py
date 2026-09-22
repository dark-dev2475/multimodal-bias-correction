import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from evaluation.baselines import BaselineSystem, create_system
from evaluation.metrics import compare_results
from evaluation.provenance import provenance
from utils.benchmark import DEFAULT_CASES_PATH, load_benchmark, resolve_image
from utils.schemas import PIPELINE_VERSION


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run and compare all three benchmark systems."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--max-iterations", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark = load_benchmark(args.cases)
    cases = benchmark.cases
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {case.id for case in cases}
        if unknown:
            raise SystemExit(f"Unknown case ids: {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case.id in wanted]

    raw = {}
    for system_name in [system.value for system in BaselineSystem]:
        runner = create_system(system_name, args.max_iterations)
        rows = []
        for case in cases:
            result = runner.run(resolve_image(case), case.question)
            if isinstance(result, tuple):
                summary, pipeline_result = result
            else:
                summary, pipeline_result = result, None

            row = {
                "case_id": case.id,
                "category": case.category,
                "case_type": case.case_type.value,
                "result": summary.as_dict()
            }
            if pipeline_result is not None:
                row["pipeline_result"] = pipeline_result
            rows.append(row)

        raw[system_name] = {
            "system": system_name,
            "cases": rows,
            "provenance": provenance(
                model=(runner.generator.vlm.model_name
                       if hasattr(runner, "generator")
                       else runner.pipeline.generator.vlm.model_name),
                benchmark_version=benchmark.benchmark_version,
                benchmark_hash=benchmark.benchmark_hash,
                configuration={"max_iterations": args.max_iterations},
                pipeline_version=PIPELINE_VERSION
            )
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_version": benchmark.benchmark_version,
        "benchmark_hash": benchmark.benchmark_hash,
        "systems": raw,
        "evaluation": compare_results(benchmark, raw)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Saved comparison report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())