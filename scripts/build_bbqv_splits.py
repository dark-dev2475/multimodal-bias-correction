"""Expand the pinned, already-cached BBQ-V corpus without downloading it again."""

import argparse
import ast
import hashlib
import io
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.bbqv_splits import family_id, group_records, assign_groups, audit_partitions, content_hash

REVISION = "a1c78b8f73bc40408993414e3d94714a6a9169d3"
DEFAULT_CACHE = Path.home() / ".cache/huggingface/datasets/ucf-crcv___bbq-v/default/0.0.0" / REVISION


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def image_signatures(raw):
    with Image.open(io.BytesIO(raw)) as image:
        fmt = image.format
        rgb = image.convert("RGB")
        prefix = f"{rgb.width}:{rgb.height}:RGB:".encode()
        pixel_hash = hashlib.sha256(prefix + rgb.tobytes()).hexdigest()
        small = list(rgb.convert("L").resize((9, 9), Image.Resampling.LANCZOS).tobytes())
        bits = 0
        for y in range(8):
            for x in range(8):
                bits = (bits << 1) | int(small[y * 9 + x] > small[y * 9 + x + 1])
        for y in range(8):
            for x in range(8):
                bits = (bits << 1) | int(small[y * 9 + x] > small[(y + 1) * 9 + x])
        return {"pixel_sha256": pixel_hash, "dhash128": f"{bits:032x}",
                "width": rgb.width, "height": rgb.height,
                "extension": {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(fmt, ".img")}


def source_dataset(cache):
    # Lazy imports keep split/audit helpers testable without heavy dependencies.
    from datasets import Dataset, Image as DatasetImage, concatenate_datasets
    info = json.loads((cache / "dataset_info.json").read_text(encoding="utf-8"))
    shards = sorted(cache.glob("*.arrow"))
    if not shards:
        raise ValueError("No cached Arrow shards found. Use the existing authenticated dataset cache.")
    ds = concatenate_datasets([Dataset.from_file(str(p)) for p in shards])
    if len(ds) != info["splits"]["test"]["num_examples"]:
        raise ValueError("Cached shard count does not match dataset metadata")
    return ds.cast_column("file_name", DatasetImage(decode=False)), info, shards


def raw_bytes(row):
    item = row["file_name"]
    return item["bytes"] if item.get("bytes") is not None else Path(item["path"]).read_bytes()


def build(args):
    ds, info, shards = source_dataset(args.source_cache)
    source_identity = {"revision": args.source_cache.name, "rows": len(ds),
                       "shards": [{"name": p.name, "bytes": p.stat().st_size} for p in shards],
                       "info_sha256": content_hash(info)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "source_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if index["source_identity"] != source_identity:
            raise ValueError("Existing index belongs to a different source; use a new output directory")
        records = index["records"]
    else:
        records = []
    encoded = {r["encoded_sha256"]: {k: r[k] for k in (
        "pixel_sha256", "dhash128", "width", "height", "extension")} for r in records}
    print(f"Indexing {len(ds)} source rows; resuming at {len(records)}", flush=True)
    for i in range(len(records), len(ds)):
        row = ds[i]
        raw = raw_bytes(row)
        byte_hash = hashlib.sha256(raw).hexdigest()
        if byte_hash not in encoded:
            encoded[byte_hash] = image_signatures(raw)
        meta = ast.literal_eval(row["additional_metadata"]) if isinstance(row["additional_metadata"], str) else row["additional_metadata"]
        records.append({"example_id": row["id"], "dataset_index": i,
            "question_family": family_id(row["id"]), "category": ds.features["category"].int2str(row["category"]),
            "question_polarity": ds.features["question_polarity"].int2str(row["question_polarity"]),
            "context": row["context"], "question": row["question"], "additional_metadata": meta,
            "encoded_sha256": byte_hash, "encoded_bytes": len(raw), **encoded[byte_hash]})
        if (i + 1) % 250 == 0 or i + 1 == len(ds):
            write_json(index_path, {"source_identity": source_identity, "records": records})
            print(f"Indexed {i + 1}/{len(ds)}; distinct encoded images {len(encoded)}", flush=True)
    exposed = set()
    for path in args.exposed_manifest:
        exposed.update(e["example_id"] for e in json.loads(path.read_text(encoding="utf-8"))["entries"])
    missing_exposed = exposed - {r["example_id"] for r in records}
    if missing_exposed:
        raise ValueError("Previously exposed IDs missing from this source revision")
    print("Building transitive leakage groups...", flush=True)
    groups = group_records(records, args.near_duplicate_distance)
    partitions = assign_groups(groups, exposed, args.seed)
    audit = audit_partitions(partitions, exposed, args.near_duplicate_distance)
    if not audit["passed"]:
        raise ValueError("Leakage audit failed")
    representatives = {}
    for r in records:
        representatives.setdefault(r["pixel_sha256"], r)
    image_root = args.output_dir / "images"
    image_root.mkdir(exist_ok=True)
    required = sum(r["encoded_bytes"] for key, r in representatives.items()
                   if not (image_root / (key + r["extension"])).exists())
    free = shutil.disk_usage(args.output_dir).free
    if required + 2 * 1024 ** 3 > free:
        raise RuntimeError(f"Export needs {required} new bytes plus a 2 GiB reserve; free={free}")
    print(f"Exporting {len(representatives)} distinct pixel images; {required / 1024 ** 2:.1f} MiB new data", flush=True)
    image_paths = {}
    for i, (key, row) in enumerate(representatives.items()):
        path = image_root / (key + row["extension"])
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["encoded_sha256"]:
                raise ValueError(f"Existing exported image has changed: {path.name}")
        else:
            raw = raw_bytes(ds[row["dataset_index"]])
            path.write_bytes(raw)
        image_paths[key] = path.resolve()
        if (i + 1) % 500 == 0:
            print(f"Images ready: {i + 1}/{len(representatives)}", flush=True)
    for split, rows in partitions.items():
        entries = []
        for row in rows:
            path = image_paths[row["pixel_sha256"]]
            try:
                path_string = path.relative_to(ROOT).as_posix()
            except ValueError:
                path_string = str(path)
            entries.append({**row, "image_path": path_string, "source": "ucf-crcv/BBQ-V",
                            "source_split": "test", "split": split, "selection_seed": args.seed,
                            "exported_image_sha256": representatives[row["pixel_sha256"]]["encoded_sha256"]})
        write_json(args.output_dir / f"{split}_manifest.json", {
            "benchmark_version": "bbqv-grouped-v2.1", "source": "ucf-crcv/BBQ-V",
            "source_revision": args.source_cache.name, "source_split": "test", "split": split,
            "partition_role": "development_no_weight_training" if split == "train" else split,
            "allow_shared_images": True, "selection_seed": args.seed,
            "selected_size": len(entries), "entries": entries})
    metadata = {"version": "bbqv-grouped-v2.1", "created_at": datetime.now(timezone.utc).isoformat(),
        "source_identity": source_identity, "source_dataset": "ucf-crcv/BBQ-V", "source_split": "test",
        "source_rows": len(records), "unique_pixel_images": len(representatives),
        "question_families": len({r["question_family"] for r in records}), "leakage_groups": len(groups),
        "largest_group_rows": max(map(len, groups.values())), "ratios_requested": [0.7, 0.15, 0.15],
        "seed": args.seed, "near_duplicate_dhash128_hamming_threshold": args.near_duplicate_distance,
        "previously_exposed_ids": sorted(exposed), "network_downloads": 0, "weight_training_performed": False,
        "exported_image_bytes": sum(r["encoded_bytes"] for r in representatives.values()),
        "partitions": {s: {"examples": len(rows), "unique_images": len({r["pixel_sha256"] for r in rows}),
                            "groups": len({r["leakage_group"] for r in rows}),
                            "fraction": len(rows) / len(records),
                            "missing_categories": sorted({r["category"] for r in records} - {r["category"] for r in rows}),
                            "category_counts": dict(sorted(Counter(r["category"] for r in rows).items()))}
                       for s, rows in partitions.items()}, "audit": audit,
        "license": "CC BY-NC 4.0", "source_url": "https://huggingface.co/datasets/ucf-crcv/BBQ-V",
        "interpretation": "Custom partitions of a pinned official test release; train is development/calibration only. Exact and detected near-duplicate images and question families are group-disjoint."}
    metadata["partition_fingerprint"] = content_hash({s: [r["example_id"] for r in rows]
                                                      for s, rows in partitions.items()})
    write_json(args.output_dir / "split_metadata.json", metadata)
    print(json.dumps(metadata["partitions"], indent=2), flush=True)
    print(f"Audit passed: {audit['passed']}; output: {args.output_dir}", flush=True)
    return metadata


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/bbqv_v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--near-duplicate-distance", type=int, default=4)
    p.add_argument("--exposed-manifest", nargs="+", type=Path,
                   default=[ROOT / "data/bbqv/benchmark_manifest.json"])
    return p.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
