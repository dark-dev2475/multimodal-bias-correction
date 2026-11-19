"""Read-only audit of generated partitions, coverage and optional image hashes."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.bbqv_loader import load_bbqv_manifest
from evaluation.bbqv_splits import audit_partitions, content_hash


def audit(directory, verify_images=False):
    read = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))
    metadata = read("split_metadata.json")
    partitions = {s: read(s + "_manifest.json")["entries"] for s in ("train", "validation", "test")}
    for split in partitions:
        load_bbqv_manifest(directory / (split + "_manifest.json"))
    result = audit_partitions(partitions, metadata["previously_exposed_ids"],
                              metadata["near_duplicate_dhash128_hamming_threshold"])
    source_ids = {r["example_id"] for r in read("source_index.json")["records"]}
    ids = [r["example_id"] for rows in partitions.values() for r in rows]
    result["exact_source_coverage"] = set(ids) == source_ids and len(ids) == len(source_ids)
    result["fingerprint_matches"] = metadata["partition_fingerprint"] == content_hash(
        {s: [r["example_id"] for r in rows] for s, rows in partitions.items()})
    result["verified_image_files"] = 0
    if verify_images:
        images = {r["image_path"]: r["exported_image_sha256"] for rows in partitions.values() for r in rows}
        for name, expected in images.items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            with path.open("rb") as handle:
                actual = hashlib.file_digest(handle, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Exported image checksum mismatch: {path}")
        result["verified_image_files"] = len(images)
    result["passed"] &= result["exact_source_coverage"] and result["fingerprint_matches"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "data/bbqv_v2")
    parser.add_argument("--verify-images", action="store_true")
    args = parser.parse_args()
    result = audit(args.directory, args.verify_images)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
