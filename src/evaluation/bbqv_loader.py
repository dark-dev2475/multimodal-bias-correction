"""Load the local BBQ-V benchmark manifest without changing pipeline semantics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "bbqv" / "benchmark_manifest.json"


@dataclass(frozen=True)
class BBQVExample:
    example_id: str
    image_path: Path
    category: str
    source: str
    split: str
    metadata: dict[str, Any]
    question: str


@dataclass(frozen=True)
class BBQVBenchmark:
    benchmark_version: str
    benchmark_hash: str
    source: str
    split: str
    selection_seed: int
    examples: tuple[BBQVExample, ...]


def _resolve_image_path(raw_path: str | Path, root: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else root / path


def _manifest_hash(data: dict[str, Any]) -> str:
    payload = json.dumps(data["entries"], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def load_bbqv_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
    root: str | Path = PROJECT_ROOT,
    require_images: bool = True,
) -> BBQVBenchmark:
    """Load a legacy subset or explicitly grouped multi-question manifest."""
    manifest_path = Path(path)
    root_path = Path(root)
    if not manifest_path.exists():
        raise FileNotFoundError(f"BBQ-V manifest not found: {manifest_path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("BBQ-V manifest must contain a non-empty 'entries' list")

    examples = []
    allow_shared_images = data.get("allow_shared_images", False)
    if not isinstance(allow_shared_images, bool):
        raise ValueError("allow_shared_images must be a boolean")
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for entry in entries:
        example_id = entry.get("example_id")
        image_path = _resolve_image_path(entry.get("image_path", ""), root_path)
        if not example_id:
            raise ValueError("BBQ-V manifest entry is missing example_id")
        if example_id in seen_ids:
            raise ValueError(f"Duplicate BBQ-V example_id: {example_id}")
        if image_path in seen_paths and not allow_shared_images:
            raise ValueError(f"Duplicate BBQ-V image_path: {image_path}")
        if require_images and not image_path.exists():
            raise FileNotFoundError(
                f"BBQ-V image for {example_id} does not exist: {image_path}"
            )

        seen_ids.add(example_id)
        seen_paths.add(image_path)
        examples.append(
            BBQVExample(
                example_id=example_id,
                image_path=image_path,
                category=entry["category"],
                source=entry["source"],
                split=entry["split"],
                metadata={
                    "additional_metadata": entry.get("additional_metadata", {}),
                    "question_polarity": entry.get("question_polarity"),
                    "context": entry.get("context"),
                    "dataset_index": entry.get("dataset_index"),
                    "selection_seed": entry.get("selection_seed"),
                    "question_family": entry.get("question_family"),
                    "leakage_group": entry.get("leakage_group"),
                    "source_split": entry.get("source_split", data.get("source_split")),
                    "source_revision": data.get("source_revision"),
                    "pixel_sha256": entry.get("pixel_sha256"),
                },
                question=entry["question"],
            )
        )

    return BBQVBenchmark(
        benchmark_version=data.get("benchmark_version", "bbqv-subset-v1"),
        benchmark_hash=_manifest_hash(data),
        source=data.get("source", "ucf-crcv/BBQ-V"),
        split=data.get("split", "test"),
        selection_seed=data.get("selection_seed", 42),
        examples=tuple(examples),
    )
