#!/usr/bin/env python3
"""Create a deterministic local subset of the official BBQ-V test split."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASET_NAME = "ucf-crcv/BBQ-V"
SPLIT = "test"
SEED = 42
TARGET_SIZE = 40
SCRIPT_VERSION = "1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "bbqv"
IMAGE_ROOT = OUTPUT_ROOT / "images"
MANIFEST_PATH = OUTPUT_ROOT / "benchmark_manifest.json"
METADATA_PATH = OUTPUT_ROOT / "dataset_metadata.json"


def _parse_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, str):
        return value if isinstance(value, dict) else {}
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return {"raw": value}
    return parsed if isinstance(parsed, dict) else {"raw": value}


def _image_digest(image: Any) -> str:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _label_name(dataset: Any, field_name: str, value: Any) -> str:
    feature = dataset.features[field_name]
    names = getattr(feature, "names", None)
    if names is not None and isinstance(value, int):
        return names[value]
    return str(value)


def _collect_candidates(dataset: Any, target_size: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    """Collect only the category-diverse images needed for this subset.

    Dataset rows are grouped from scalar columns first. Images are decoded and
    hashed only while filling the requested subset, avoiding a full image pass.
    """
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for dataset_index, category_value in enumerate(dataset["category"]):
        category = _label_name(dataset, "category", category_value)
        grouped_indices[category].append(dataset_index)

    rng = random.Random(seed)
    for indices in grouped_indices.values():
        rng.shuffle(indices)

    categories = sorted(grouped_indices)
    positions = {category: 0 for category in categories}
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_images: set[str] = set()
    selected_count = 0

    while selected_count < target_size:
        available = [
            category
            for category in categories
            if positions[category] < len(grouped_indices[category])
        ]
        if not available:
            break
        category_counts = {category: len(candidates[category]) for category in categories}
        category = min(available, key=lambda item: (category_counts[item], item))
        dataset_index = grouped_indices[category][positions[category]]
        positions[category] += 1
        row = dataset[dataset_index]
        image = row["file_name"]
        digest = _image_digest(image)
        if digest in seen_images:
            continue
        seen_images.add(digest)
        candidates[category].append(
            {
                "dataset_index": dataset_index,
                "example_id": row["id"],
                "category": category,
                "image": image,
                "image_digest": digest,
                "additional_metadata": _parse_metadata(row.get("additional_metadata")),
                "question_polarity": _label_name(dataset, "question_polarity", row["question_polarity"]),
                "context": row.get("context"),
                "question": row.get("question"),
            }
        )
        selected_count += 1

    return candidates


def _select_candidates(
    candidates: dict[str, list[dict[str, Any]]], target_size: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    categories = sorted(candidates)
    if not categories:
        raise ValueError("BBQ-V contains no selectable categories")

    shuffled = {category: list(rows) for category, rows in candidates.items()}
    for rows in shuffled.values():
        rng.shuffle(rows)

    target_size = min(target_size, sum(len(rows) for rows in shuffled.values()))
    selected: list[dict[str, Any]] = []
    positions = {category: 0 for category in categories}

    while len(selected) < target_size:
        available = [
            category
            for category in categories
            if positions[category] < len(shuffled[category])
        ]
        if not available:
            break
        counts = Counter(item["category"] for item in selected)
        category = min(available, key=lambda item: (counts[item], item))
        selected.append(shuffled[category][positions[category]])
        positions[category] += 1

    return sorted(selected, key=lambda item: item["example_id"])


def create_benchmark(target_size: int = TARGET_SIZE, seed: int = SEED) -> dict[str, Any]:
    dataset = load_dataset(DATASET_NAME, split=SPLIT)
    candidates = _collect_candidates(dataset, target_size, seed)
    selected = _select_candidates(candidates, target_size, seed)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    for item in selected:
        filename = f"bbqv_{item['example_id']}.jpg"
        image_path = IMAGE_ROOT / filename
        if not image_path.exists():
            item["image"].convert("RGB").save(image_path, format="JPEG", quality=95)
        manifest_entries.append(
            {
                "example_id": item["example_id"],
                "image_path": str(image_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "category": item["category"],
                "source": DATASET_NAME,
                "split": SPLIT,
                "dataset_index": item["dataset_index"],
                "selection_seed": seed,
                "additional_metadata": item["additional_metadata"],
                "question_polarity": item["question_polarity"],
                "context": item["context"],
                "question": item["question"],
            }
        )

    category_counts = dict(sorted(Counter(item["category"] for item in selected).items()))
    dataset_category_counts = dict(
        sorted({category: len(rows) for category, rows in candidates.items()}.items())
    )
    benchmark = {
        "benchmark_version": "bbqv-subset-v1",
        "source": DATASET_NAME,
        "split": SPLIT,
        "selection_seed": seed,
        "target_size": target_size,
        "selected_size": len(manifest_entries),
        "entries": manifest_entries,
    }
    MANIFEST_PATH.write_text(json.dumps(benchmark, indent=2) + "\n", encoding="utf-8")

    dataset_version = getattr(getattr(dataset, "info", None), "version", None)
    metadata = {
        "dataset_name": "BBQ-V",
        "hugging_face_dataset": DATASET_NAME,
        "split": SPLIT,
        "dataset_size": len(dataset),
        "number_of_categories": len(dataset_category_counts),
        "category_counts": dataset_category_counts,
        "selected_benchmark_size": len(manifest_entries),
        "selected_category_counts": category_counts,
        "selection_seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_revision": str(dataset_version) if dataset_version else None,
        "git_commit": _git_commit(),
        "script_version": SCRIPT_VERSION,
        "image_deduplication": "SHA-256 over decoded PNG pixels",
        "ground_truth_policy": "BBQ-V categories are benchmark labels, not claim-level ground truth.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"manifest": benchmark, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result = create_benchmark(args.target_size, args.seed)
    print(f"Selected images: {result['manifest']['selected_size']}")
    print(f"Category distribution: {result['metadata']['selected_category_counts']}")
    print(f"Unique images: {len({entry['image_path'] for entry in result['manifest']['entries']})}")
    missing = [
        entry["image_path"]
        for entry in result["manifest"]["entries"]
        if not (PROJECT_ROOT / entry["image_path"]).exists()
    ]
    print(f"Missing images: {len(missing)}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    if result["manifest"]["entries"]:
        print("Example entry:")
        print(json.dumps(result["manifest"]["entries"][0], indent=2))


if __name__ == "__main__":
    main()