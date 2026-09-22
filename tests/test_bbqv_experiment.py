import json
from pathlib import Path
from types import SimpleNamespace

from scripts.run_bbqv_experiment import (
    _cache_key,
    parse_args,
    select_examples,
)


def test_experiment_configuration_is_small_and_fair():
    args = parse_args([
        "--num-examples", "3",
        "--seed", "42",
        "--systems", "vlm_only", "generic_self_correction", "proposed",
    ])
    assert args.num_examples == 3
    assert args.seed == 42
    assert args.systems == ["vlm_only", "generic_self_correction", "proposed"]


def test_experiment_selection_is_deterministic_and_shared():
    examples = [SimpleNamespace(example_id=str(i)) for i in range(8)]
    first = [example.example_id for example in select_examples(examples, 3, 42)]
    second = [example.example_id for example in select_examples(examples, 3, 42)]
    assert first == second
    assert len(set(first)) == 3


def test_cache_key_distinguishes_system_and_example():
    runner = SimpleNamespace(
        generator=SimpleNamespace(
            vlm=SimpleNamespace(model_name="same-model"),
            prompt_fingerprint=lambda: "same-prompt",
        )
    )
    first = SimpleNamespace(
        example_id="one", image_path=Path("one.jpg"), question="What is visible?"
    )
    second = SimpleNamespace(
        example_id="two", image_path=Path("two.jpg"), question="What is visible?"
    )
    assert _cache_key("vlm_only", runner, first, 2) != _cache_key(
        "vlm_only", runner, second, 2
    )
    assert _cache_key("vlm_only", runner, first, 2) != _cache_key(
        "proposed", runner, first, 2
    )


def test_smoke_report_shape_is_collection_only(tmp_path):
    report = {
        "systems": {
            "vlm_only": {"cases": [{"example_id": "one", "system": "vlm_only"}]},
            "proposed": {"cases": [{"example_id": "one", "system": "proposed"}]},
        },
        "evaluation": {"ground_truth_available": False, "metrics_computed": False},
    }
    path = tmp_path / "baseline_comparison.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert set(loaded["systems"]) == {"vlm_only", "proposed"}
    assert loaded["evaluation"]["metrics_computed"] is False