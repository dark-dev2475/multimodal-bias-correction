#!/usr/bin/env python3
"""Create independent LLM-generated reference annotations from saved runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from evaluation.automatic_annotator import (
    AutomaticAnnotation,
    AutomaticAnnotator,
    annotation_set,
    validate_annotation_coverage,
)
from pipeline.claim_extractor import ClaimExtractor
from utils.free_policy import DEFAULT_JUDGE

VALID_SYSTEMS = {"vlm_only", "generic_self_correction", "proposed"}


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--annotator-model",
        default=DEFAULT_JUDGE,
        help="Independent vision model used for automatic reference annotations.",
    )
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args(args)


def _result_files(input_path: Path):
    return sorted(
        path for path in input_path.glob("**/*.json")
        if path.name not in {"baseline_comparison.json", "experiment_manifest.json"}
        and "_cache" not in path.parts
    )


def _claims_from_pipeline(record: dict):
    pipeline = record.get("pipeline_result", {})
    history = pipeline.get("correction_history", [])
    initial = history[0].get("claims", []) if history else pipeline.get("claims", [])
    final = pipeline.get("claims", [])
    return (
        [item["claim"] for item in initial],
        [item["claim"] for item in final],
    )


class ClaimExtractionCache:
    def __init__(self):
        self.extractor = ClaimExtractor()

    def extract(self, response):
        return [claim.claim for claim in self.extractor.extract(response).claims]


def _claims_for_record(record: dict, extractor: ClaimExtractionCache):
    pipeline = record.get("pipeline_result")
    if pipeline is not None:
        return _claims_from_pipeline(record)
    result = record["result"]
    initial = extractor.extract(result["initial_response"])
    final = extractor.extract(result["final_response"])
    return initial, final


def _load_records(input_path: Path, max_examples: int | None):
    records = []
    seen_examples = set()
    for path in _result_files(input_path):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "completed":
            continue
        system = record.get("system")
        if system not in VALID_SYSTEMS:
            raise ValueError(f"Unexpected system in {path}: {system}")
        example_id = record.get("example_id")
        if not example_id:
            raise ValueError(f"Missing example_id in {path}")
        if max_examples is not None and example_id not in seen_examples:
            if len(seen_examples) >= max_examples:
                continue
            seen_examples.add(example_id)
        elif max_examples is not None and example_id not in seen_examples:
            continue
        records.append(record)
    if not records:
        raise ValueError(f"No completed system outputs found under {input_path}")
    return records


def run(args):
    records = _load_records(args.input, args.max_examples)
    extractor = ClaimExtractionCache()
    annotator = AutomaticAnnotator(model_name=args.annotator_model)
    annotations: list[AutomaticAnnotation] = []
    expected_keys = set()

    for record in records:
        initial_claims, final_claims = _claims_for_record(record, extractor)
        stages = {
            "initial": initial_claims,
            "final": final_claims,
        }
        for stage, claims in stages.items():
            for claim in claims:
                key = (record["example_id"], record["system"], stage, claim)
                if key in expected_keys:
                    raise ValueError(f"Duplicate claim input: {key}")
                expected_keys.add(key)
                annotation, _ = annotator.annotate(
                    example_id=record["example_id"],
                    image_path=record["image_path"],
                    system=record["system"],
                    benchmark_category=record["category"],
                    stage=stage,
                    claim=claim,
                )
                annotations.append(annotation)

    validate_annotation_coverage(expected_keys, annotations)
    result = annotation_set(annotations, args.annotator_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    distributions = {}
    for annotation in annotations:
        distributions[annotation.evidence_status] = distributions.get(annotation.evidence_status, 0) + 1
    return result, annotator, distributions


def main(argv=None):
    args = parse_args(argv)
    result, annotator, distributions = run(args)
    problematic = sum(annotation.problematic for annotation in result.annotations)
    initial = sum(annotation.stage == "initial" for annotation in result.annotations)
    final = sum(annotation.stage == "final" for annotation in result.annotations)
    print(f"Annotations: {len(result.annotations)}")
    print(f"Initial claims: {initial}")
    print(f"Final claims: {final}")
    print(f"Problematic claims: {problematic}")
    print(f"Evidence status distribution: {distributions}")
    print(f"Annotator model: {result.annotator_model}")
    print(f"API calls: {annotator.api_calls}")
    print(f"Cache hits: {annotator.cache_hits}")
    print(f"Cache misses: {annotator.cache_misses}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
