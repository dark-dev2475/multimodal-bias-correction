import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from evaluation.ground_truth import load_ground_truth
from evaluation.metrics import compare_results
from utils.benchmark import DEFAULT_CASES_PATH, load_benchmark


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score annotated results against independent ground truth."
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark = load_benchmark(args.cases)
    load_ground_truth(args.ground_truth, benchmark)
    payload = json.loads(args.results.read_text(encoding="utf-8"))

    if "systems" in payload:
        systems = payload["systems"]
    else:
        system = args.system or payload.get("system", "unknown")
        systems = {system: payload}

    report = {
        "benchmark_version": benchmark.benchmark_version,
        "benchmark_hash": benchmark.benchmark_hash,
        "evaluation": compare_results(benchmark, systems)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Saved evaluation report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())