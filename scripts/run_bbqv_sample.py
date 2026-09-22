#!/usr/bin/env python3
"""Run the existing proposed pipeline on a small BBQ-V sample."""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
from contextlib import redirect_stdout
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from evaluation.bbqv_loader import DEFAULT_MANIFEST_PATH, load_bbqv_manifest
from pipeline.pipeline import BiasCorrectionPipeline
from utils.logger import ExperimentLogger, new_run_id


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run the existing proposed pipeline on BBQ-V examples."
    )
    parser.add_argument("--num-examples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "outputs",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(args)


def select_examples(examples, num_examples, seed):
    if num_examples < 1:
        raise ValueError("--num-examples must be at least 1")
    if num_examples > len(examples):
        raise ValueError(
            f"Requested {num_examples} examples, but the manifest contains "
            f"only {len(examples)}"
        )
    selected = list(examples)
    random.Random(seed).shuffle(selected)
    return selected[:num_examples]


def run_sample(example, pipeline, logger, benchmark, args):
    output = io.StringIO()
    stream = None if args.verbose else redirect_stdout(output)
    if stream is None:
        result = pipeline.run(
            image_path=example.image_path,
            question=example.question,
            max_iterations=args.max_iterations,
        )
    else:
        with stream:
            result = pipeline.run(
                image_path=example.image_path,
                question=example.question,
                max_iterations=args.max_iterations,
            )

    run_id = new_run_id()
    result_path = logger.save_run(
        case_id=example.example_id,
        image_path=example.image_path,
        model=pipeline.generator.vlm.model_name,
        result=result,
        experiment_id="bbqv_phase3_sample",
        max_iterations=args.max_iterations,
        prompt_hash=pipeline.generator.prompt_fingerprint(),
        run_id=run_id,
        benchmark_version=benchmark.benchmark_version,
        benchmark_hash=benchmark.benchmark_hash,
    )
    return result, result_path, output.getvalue()


def main(argv=None):
    args = parse_args(argv)
    benchmark = load_bbqv_manifest(args.manifest)
    selected = select_examples(benchmark.examples, args.num_examples, args.seed)
    pipeline = BiasCorrectionPipeline()
    logger = ExperimentLogger(output_dir=args.output_dir)

    for example in selected:
        result, result_path, trace = run_sample(
            example, pipeline, logger, benchmark, args
        )
        if not args.verbose:
            print(trace)
        evidence = result["claim_analysis"]
        evidence_counts = {}
        for item in evidence:
            status = item["evidence_status"]
            evidence_counts[status] = evidence_counts.get(status, 0) + 1
        print(json.dumps({
            "example_id": example.example_id,
            "category": example.category,
            "image_path": str(example.image_path),
            "question_used_by_existing_pipeline": example.question,
            "initial_response": result["initial_response"],
            "extracted_claims": len(result["claims"]),
            "evidence_summary": evidence_counts,
            "bias_flags": len(result["flagged_claims"]),
            "correction_happened": result["corrected_response"] != result["initial_response"],
            "final_response": result["corrected_response"],
            "verification": result["verification"],
            "iterations": result["correction_iterations"],
            "api_calls": sum(
                metrics.get("calls", 0) + metrics.get("retries", 0)
                for metrics in result["stage_metrics"].values()
            ),
            "result_path": str(result_path),
            "warnings": [
                item["claim"]
                for item in evidence
                if item.get("bias_analysis_missing")
            ],
        }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()