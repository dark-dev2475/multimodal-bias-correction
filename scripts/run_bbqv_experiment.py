#!/usr/bin/env python3
"""Run a small, comparable BBQ-V experiment across all three systems."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from evaluation.baselines import BaselineSystem, create_system
from evaluation.bbqv_loader import DEFAULT_MANIFEST_PATH, load_bbqv_manifest
from evaluation.provenance import git_commit, provenance
from utils.schemas import PIPELINE_VERSION


DEFAULT_SYSTEMS = [system.value for system in BaselineSystem]
EXPERIMENT_VERSION = "bbqv-experiment-v1"


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run the three existing systems on a BBQ-V subset."
    )
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--systems", nargs="+", choices=DEFAULT_SYSTEMS, default=DEFAULT_SYSTEMS)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "outputs" / "bbqv_experiment",
    )
    parser.add_argument("--no-cache", action="store_true")
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


def _runner_model(runner):
    if hasattr(runner, "generator"):
        return runner.generator.vlm.model_name
    return runner.pipeline.generator.vlm.model_name


def _runner_prompt_hash(runner):
    if hasattr(runner, "generator"):
        return runner.generator.prompt_fingerprint()
    return runner.pipeline.generator.prompt_fingerprint()


def _system_prompt_version(system_name):
    if system_name == BaselineSystem.vlm_only.value:
        return "initial_generation_v1"
    if system_name == BaselineSystem.generic_self_correction.value:
        return "generic_correction_v1"
    return "pipeline_" + PIPELINE_VERSION


def _cache_key(system_name, runner, example, max_iterations):
    payload = {
        "system": system_name,
        "model": _runner_model(runner),
        "prompt_version": _system_prompt_version(system_name),
        "prompt_hash": _runner_prompt_hash(runner),
        "example_id": example.example_id,
        "image_path": str(example.image_path),
        "question": example.question,
        "max_iterations": max_iterations,
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _cache_path(output_dir, system_name, cache_key):
    return output_dir / "_cache" / system_name / f"{cache_key}.json"


def _run_one(system_name, runner, example, benchmark, args, cache_dir):
    cache_key = _cache_key(system_name, runner, example, args.max_iterations)
    cache_path = _cache_path(cache_dir, system_name, cache_key)
    if not args.no_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        cached["cache_hit"] = True
        return cached

    started = time.monotonic()
    try:
        result = runner.run(example.image_path, example.question)
    except Exception as exc:
        return {
            "example_id": example.example_id,
            "image_path": str(example.image_path),
            "category": example.category,
            "source": example.source,
            "split": example.split,
            "system": system_name,
            "question": example.question,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "cache_key": cache_key,
            "cache_hit": False,
        }
    if isinstance(result, tuple):
        summary, pipeline_result = result
    else:
        summary, pipeline_result = result, None

    row = {
        "example_id": example.example_id,
        "image_path": str(example.image_path),
        "category": example.category,
        "source": example.source,
        "split": example.split,
        "system": system_name,
        "question": example.question,
        "status": "completed",
        "result": summary.as_dict(),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "cache_key": cache_key,
        "cache_hit": False,
    }
    if pipeline_result is not None:
        row["pipeline_result"] = pipeline_result

    if not args.no_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    return row


def run_experiment(args):
    benchmark = load_bbqv_manifest(args.manifest)
    examples = select_examples(benchmark.examples, args.num_examples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runners = {
        system_name: create_system(system_name, args.max_iterations)
        for system_name in args.systems
    }
    raw = {system_name: [] for system_name in args.systems}
    for system_name, runner in runners.items():
        for example in examples:
            row = _run_one(system_name, runner, example, benchmark, args, args.output_dir)
            system_dir = args.output_dir / system_name / example.example_id
            system_dir.mkdir(parents=True, exist_ok=True)
            output_path = system_dir / f"{system_name}.json"
            output_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
            row["output_path"] = str(output_path)
            raw[system_name].append(row)

    common_configuration = {
        "num_examples": len(examples),
        "selection_seed": args.seed,
        "max_iterations": args.max_iterations,
        "systems": args.systems,
        "category_metadata_sent_to_model": False,
        "ground_truth_sent_to_model": False,
        "automated_verifier_used_as_ground_truth": False,
    }
    systems = {}
    for system_name, rows in raw.items():
        runner = runners[system_name]
        systems[system_name] = {
            "system": system_name,
            "cases": rows,
            "provenance": provenance(
                model=_runner_model(runner),
                benchmark_version=benchmark.benchmark_version,
                benchmark_hash=benchmark.benchmark_hash,
                configuration=common_configuration,
                prompt_versions={"system": _system_prompt_version(system_name)},
                pipeline_version=PIPELINE_VERSION,
            ),
        }

    report = {
        "experiment_version": EXPERIMENT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_version": benchmark.benchmark_version,
        "benchmark_hash": benchmark.benchmark_hash,
        "systems": systems,
        "evaluation": {
            "status": "collection_only",
            "ground_truth_available": False,
            "metrics_computed": False,
        },
    }
    comparison_path = args.output_dir / "baseline_comparison.json"
    comparison_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "generated_at": report["generated_at"],
        "source_manifest": str(args.manifest),
        "benchmark_version": benchmark.benchmark_version,
        "benchmark_hash": benchmark.benchmark_hash,
        "selection_seed": args.seed,
        "selected_example_ids": [example.example_id for example in examples],
        "selected_image_paths": [str(example.image_path) for example in examples],
        "systems": args.systems,
        "model_names": {
            system_name: _runner_model(runner)
            for system_name, runner in runners.items()
        },
        "configuration": common_configuration,
        "pipeline_commit": git_commit(),
        "outputs": str(comparison_path),
    }
    manifest_path = args.output_dir / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return report, manifest, comparison_path, manifest_path


def main(argv=None):
    args = parse_args(argv)
    report, manifest, comparison_path, manifest_path = run_experiment(args)
    print(f"Selected images: {len(manifest['selected_example_ids'])}")
    print(f"Systems: {', '.join(manifest['systems'])}")
    print(f"Outputs: {len(manifest['selected_example_ids']) * len(manifest['systems'])}")
    print(f"Comparison: {comparison_path}")
    print(f"Manifest: {manifest_path}")
    return report


if __name__ == "__main__":
    main()