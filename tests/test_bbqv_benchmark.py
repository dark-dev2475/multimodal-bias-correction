import json

from PIL import Image

from scripts.create_bbqv_benchmark import PROJECT_ROOT, _select_candidates


MANIFEST_PATH = PROJECT_ROOT / "data" / "bbqv" / "benchmark_manifest.json"


def test_bbqv_manifest_contains_real_unique_images():
    assert MANIFEST_PATH.exists()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = manifest["entries"]

    assert 30 <= len(entries) <= 50
    assert len({entry["example_id"] for entry in entries}) == len(entries)
    assert len({entry["image_path"] for entry in entries}) == len(entries)
    assert len({entry["category"] for entry in entries}) >= 2

    for entry in entries:
        image_path = PROJECT_ROOT / entry["image_path"]
        assert image_path.exists()
        with Image.open(image_path) as image:
            image.verify()


def test_bbqv_selection_is_deterministic():
    candidates = {
        "Age": [
            {"example_id": "age-1", "category": "Age"},
            {"example_id": "age-2", "category": "Age"},
        ],
        "Religion": [
            {"example_id": "religion-1", "category": "Religion"},
            {"example_id": "religion-2", "category": "Religion"},
        ],
        "SES": [
            {"example_id": "ses-1", "category": "SES"},
            {"example_id": "ses-2", "category": "SES"},
        ],
    }
    first = [item["example_id"] for item in _select_candidates(candidates, 5, 42)]
    second = [item["example_id"] for item in _select_candidates(candidates, 5, 42)]
    assert first == second