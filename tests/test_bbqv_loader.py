import json
from types import SimpleNamespace

import pytest

from evaluation.bbqv_loader import BBQVBenchmark, BBQVExample, load_bbqv_manifest
from scripts.run_bbqv_sample import parse_args, run_sample, select_examples
from utils.logger import ExperimentLogger


def _write_manifest(tmp_path, image_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "benchmark_version": "bbqv-subset-v1",
        "source": "ucf-crcv/BBQ-V",
        "split": "test",
        "selection_seed": 42,
        "entries": [{
            "example_id": "example-1",
            "image_path": str(image_path),
            "category": "Age",
            "source": "ucf-crcv/BBQ-V",
            "split": "test",
            "question": "What is visible?",
        }],
    }), encoding="utf-8")
    return path


def test_manifest_loading_and_example_parsing(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    benchmark = load_bbqv_manifest(_write_manifest(tmp_path, image))
    example = benchmark.examples[0]
    assert example.example_id == "example-1"
    assert example.category == "Age"
    assert example.image_path == image
    assert benchmark.benchmark_hash


def test_missing_image_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bbqv_manifest(_write_manifest(tmp_path, tmp_path / "missing.jpg"))


def test_sample_selection_and_runner_configuration_are_deterministic():
    examples = [type("Example", (), {"example_id": str(i)})() for i in range(5)]
    first = [item.example_id for item in select_examples(examples, 3, 42)]
    second = [item.example_id for item in select_examples(examples, 3, 42)]
    assert first == second
    args = parse_args(["--num-examples", "1", "--seed", "7"])
    assert args.num_examples == 1
    assert args.seed == 7


def test_runner_saves_existing_pipeline_result_schema(tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    example = BBQVExample(
        example_id="example-1",
        image_path=image,
        category="Age",
        source="ucf-crcv/BBQ-V",
        split="test",
        metadata={},
        question="What is visible?",
    )
    benchmark = BBQVBenchmark(
        benchmark_version="bbqv-subset-v1",
        benchmark_hash="hash",
        source="ucf-crcv/BBQ-V",
        split="test",
        selection_seed=42,
        examples=(example,),
    )

    class FakePipeline:
        generator = SimpleNamespace(
            vlm=SimpleNamespace(model_name="test-model"),
            prompt_fingerprint=lambda: "prompt-hash",
        )

        def run(self, **kwargs):
            return {
                "question": kwargs["question"],
                "generation_mode": "real",
                "initial_response": "A visible object.",
                "claims": [],
                "claim_analysis": [],
                "flagged_claims": [],
                "corrected_response": "A visible object.",
                "verification": {"claims": []},
                "verification_passed": True,
                "stop_reason": "verified",
                "correction_iterations": 1,
                "correction_history": [],
                "stage_metrics": {},
                "prompt_versions": {},
                "configuration": {},
            }

    args = parse_args(["--num-examples", "1", "--verbose", "--output-dir", str(tmp_path / "out")])
    result, result_path, _ = run_sample(
        example,
        FakePipeline(),
        ExperimentLogger(tmp_path / "out"),
        benchmark,
        args,
    )
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["verification_passed"] is True
    assert saved["case_id"] == "example-1"
    assert saved["benchmark_version"] == "bbqv-subset-v1"
    assert saved["benchmark_hash"] == "hash"
    assert saved["initial_response"] == "A visible object."
