import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from pipeline.pipeline import BiasCorrectionPipeline
from evaluation.metrics import evaluate_case
from utils.benchmark import DEFAULT_CASES_PATH, load_benchmark, resolve_image
from utils.schemas import PIPELINE_VERSION


ABLATIONS = {
    "full": {},
    "without_claim_extraction": {"enable_claim_extraction": False},
    "without_evidence_checking": {"enable_evidence_checking": False},
    "without_bias_monitor": {"enable_bias_monitor": False},
    "without_verification": {"enable_verification": False}
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the proposed pipeline ablation study."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--ablation", choices=["all"] + list(ABLATIONS),
                        default="all")
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

    names = list(ABLATIONS) if args.ablation == "all" else [args.ablation]
    report = {
        "benchmark_version": benchmark.benchmark_version,
        "benchmark_hash": benchmark.benchmark_hash,
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ablations": {}
    }

    for name in names:
        runner = BiasCorrectionPipeline(
            **ABLATIONS[name],
        )
        rows = []
        for case in cases:
            result = runner.run(
                image_path=resolve_image(case),
                question=case.question,
                max_iterations=args.max_iterations
            )
            rows.append({
                "case_id": case.id,
                "category": case.category,
                "case_type": case.case_type.value,
                "result": result,
                "evaluation": evaluate_case(
                    case, {"pipeline_result": result}
                ).model_dump(mode="json")
            })

        report["ablations"][name] = {
            "configuration": {
                **ABLATIONS[name],
                "max_iterations": args.max_iterations
            },
            "model": runner.generator.vlm.model_name,
            "cases": rows
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Saved ablation report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())