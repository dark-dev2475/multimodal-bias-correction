import io
import sys
import json
import argparse
import traceback
import contextlib
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from tqdm import tqdm

from pipeline.pipeline import BiasCorrectionPipeline
from pipeline.generator import MODES
from utils.benchmark import DEFAULT_CASES_PATH, load_benchmark, resolve_image
from utils.logger import ExperimentLogger, new_run_id
from utils.schemas import BatchCaseResult, BatchSummary, RunStatus


def parse_args():

    parser = argparse.ArgumentParser(
        description="Run the bias-correction pipeline over every benchmark case."
    )

    parser.add_argument(
        "--experiment-id",
        default="phase4_baseline_v1",
        help=(
            "Groups runs into one experiment. Results are written under "
            "<output-dir>/<experiment-id>/, so a new id never disturbs "
            "an earlier experiment."
        )
    )

    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--mode", choices=MODES, default="real")
    parser.add_argument("--max-iterations", type=int, default=2)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "outputs"
    )

    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="CASE_ID",
        help="Run only these case ids."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run at most this many cases."
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the batch on the first failing case."
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show each case's full pipeline trace instead of capturing it."
    )

    parser.add_argument("--no-save", action="store_true")

    return parser.parse_args()


def select_cases(cases, only, limit):

    if only:
        wanted = set(only)
        unknown = wanted - {case.id for case in cases}

        if unknown:
            raise SystemExit(
                f"Unknown case ids: {', '.join(sorted(unknown))}"
            )

        cases = [case for case in cases if case.id in wanted]

    return cases[:limit] if limit else cases


def categories_of(flagged):

    return sorted({
        item["bias_category"]
        for item in flagged
        if item.get("bias_category") and item["bias_category"] != "none"
    })


def run_case(pipeline, case, args, logger, benchmark):
    """Run one case. Model failures are recorded, not raised."""

    row = BatchCaseResult(
        experiment_id=args.experiment_id,
        case_id=case.id,
        case_type=case.case_type,
        category=case.category,
        run_status=RunStatus.completed
    )

    # The pipeline is chatty; capturing keeps the progress bar readable and
    # gives us something to print if the case fails.
    buffer = io.StringIO()
    sink = contextlib.nullcontext() if args.verbose \
        else contextlib.redirect_stdout(buffer)

    try:
        with sink:
            result = pipeline.run(
                image_path=resolve_image(case),
                question=case.question,
                max_iterations=args.max_iterations
            )

        # Terminating without converging is still a completed run; only an
        # exception means the case failed to execute.
        row.stop_reason = result["stop_reason"]
        row.verification_passed = result["verification_passed"]
        row.correction_iterations = result["correction_iterations"]
        row.stage_metrics = result["stage_metrics"]

        row.initial_response = result["initial_response"]
        row.final_response = result["corrected_response"]

        history = result["correction_history"]
        first = history[0]["flagged_claims"] if history else []
        last = history[-1]["flagged_claims"] if history else []

        row.initial_flagged_claims = len(first)
        row.initial_bias_categories = categories_of(first)
        row.final_flagged_claims = len(last)
        row.final_bias_categories = categories_of(last)

        row.final_problematic_claims = sum(
            1 for claim in result["verification"].get("claims", [])
            if claim.get("problematic")
        )

        if not args.no_save:
            run_id = new_run_id()
            path = logger.save_run(
                case_id=case.id,
                image_path=resolve_image(case),
                model=pipeline.generator.vlm.model_name,
                result=result,
                experiment_id=args.experiment_id,
                max_iterations=args.max_iterations,
                prompt_hash=pipeline.generator.prompt_fingerprint(),
                run_id=run_id,
                case_type=case.case_type,
                benchmark_version=benchmark.benchmark_version,
                benchmark_hash=benchmark.benchmark_hash
            )
            row.run_id = run_id
            row.result_file = str(path)

    except Exception as exc:
        row.run_status = RunStatus.failed
        row.error = f"{type(exc).__name__}: {exc}"

        if not args.verbose:
            tqdm.write(buffer.getvalue()[-2000:])

        tqdm.write(f"\n{case.id} FAILED: {row.error}")
        tqdm.write(traceback.format_exc(limit=3))

        if args.stop_on_error:
            raise

    return row


def print_summary(summary):

    print("\n" + "=" * 82)
    print("BENCHMARK SUMMARY")
    print("=" * 82)
    print(f"Experiment: {summary.experiment_id}")
    print(f"Batch:      {summary.batch_id}")
    print(f"Benchmark:  {summary.benchmark_version} "
          f"(hash {summary.benchmark_hash})")
    print(f"Mode:       {summary.generation_mode.value}")
    print(f"Model:      {summary.model}")
    print(f"Cases:      {summary.cases_succeeded} completed, "
          f"{summary.cases_failed} failed, of {summary.cases_total}")
    print()

    print(f"{'case':<10} {'type':<17} {'flags':>9} {'probs':>6} "
          f"{'iters':>5}  {'status':<10} {'stop_reason'}")
    print("-" * 82)

    for row in summary.results:
        if row.run_status is RunStatus.failed:
            print(f"{row.case_id:<10} {row.case_type.value:<17} "
                  f"{'-':>9} {'-':>6} {'-':>5}  {'failed':<10} -")
            continue

        flags = f"{row.initial_flagged_claims}->{row.final_flagged_claims}"
        print(f"{row.case_id:<10} {row.case_type.value:<17} {flags:>9} "
              f"{row.final_problematic_claims:>6} "
              f"{row.correction_iterations:>5}  "
              f"{row.run_status.value:<10} {row.stop_reason.value}")


def main():

    args = parse_args()

    benchmark = load_benchmark(args.cases)
    cases = select_cases(benchmark.cases, args.only, args.limit)

    if not cases:
        raise SystemExit("No cases selected.")

    pipeline = BiasCorrectionPipeline(generation_mode=args.mode)
    logger = ExperimentLogger(output_dir=args.output_dir)

    batch_id = new_run_id()
    rows = []

    for case in tqdm(cases, desc="cases", unit="case"):
        rows.append(run_case(pipeline, case, args, logger, benchmark))

    summary = BatchSummary(
        experiment_id=args.experiment_id,
        batch_id=batch_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        generation_mode=args.mode,
        model=pipeline.generator.vlm.model_name,
        max_iterations=args.max_iterations,
        benchmark_version=benchmark.benchmark_version,
        benchmark_hash=benchmark.benchmark_hash,
        benchmark_path=benchmark.source_path,
        prompt_versions=pipeline._prompt_versions(),
        prompt_hash=pipeline.generator.prompt_fingerprint(),
        cases_total=len(rows),
        cases_succeeded=sum(
            1 for r in rows if r.run_status is RunStatus.completed),
        cases_failed=sum(
            1 for r in rows if r.run_status is RunStatus.failed),
        results=rows
    )

    print_summary(summary)

    if not args.no_save:
        batch_dir = args.output_dir / args.experiment_id / "_batches"
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_file = batch_dir / f"{batch_id}.json"

        batch_file.write_text(
            json.dumps(summary.model_dump(mode="json"),
                       indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        print(f"\nBatch summary: {batch_file}")

    return 1 if summary.cases_failed else 0


if __name__ == "__main__":
    sys.exit(main())
