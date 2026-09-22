#!/usr/bin/env python3
"""Inspect the official BBQ-V dataset from Hugging Face.

This script follows the Phase 1 requirement: verify access, inspect the schema,
and confirm that images are actually available before any benchmark integration.

Usage examples:
    python scripts/inspect_bbqv.py --split train[:5]
    python scripts/inspect_bbqv.py --split validation[:5]
    python scripts/inspect_bbqv.py --limit 3
"""

from __future__ import annotations

import argparse
import os
from typing import Any

try:
    from datasets import load_dataset
except Exception as exc:  # pragma: no cover - only for user guidance
    raise SystemExit(
        "datasets is not installed. Install it with: python -m pip install datasets pillow"
    ) from exc


DATASET_NAME = "ucf-crcv/BBQ-V"


def _check_auth_hint() -> None:
    print("=" * 80)
    print("BBQ-V access required")
    print("=" * 80)
    print("The dataset is gated on the Hugging Face Hub and requires authentication.")
    print("To access it:")
    print("1. Create or sign in to a Hugging Face account at https://huggingface.co")
    print("2. Visit: https://huggingface.co/datasets/ucf-crcv/BBQ-V")
    print("3. Request access / accept the required conditions for the dataset.")
    print("4. Authenticate locally with either:")
    print("   - huggingface-cli login")
    print("   - or set HF_TOKEN in your environment")
    print("5. Re-run this script after access is granted.")
    print("=" * 80)


def _safe_value_repr(value: Any, max_len: int = 300) -> str:
    try:
        return repr(value)[:max_len]
    except Exception:
        return "<unrepresentable>"


def inspect(split: str, limit: int) -> None:
    token = os.environ.get("HF_TOKEN")
    print(f"HF_TOKEN set: {bool(token)}")
    print(f"Loading dataset: {DATASET_NAME}")

    try:
        ds = load_dataset(DATASET_NAME, split=split)
    except Exception as exc:  # pragma: no cover - user guidance path
        print("\nDataset load failed.")
        print(f"Exception type: {type(exc).__name__}")
        print(f"Message: {exc}")
        _check_auth_hint()
        raise SystemExit(1) from exc

    print(f"Dataset loaded successfully. Number of examples: {len(ds)}")
    print("Features:")
    for name, feature in ds.features.items():
        print(f"  - {name}: {feature}")

    print("\nSample rows:")
    for i in range(min(limit, len(ds))):
        ex = ds[i]
        print(f"--- sample {i} ---")
        for key, value in ex.items():
            if key == "image":
                print(f"{key}: image_type={type(value).__name__}, size={getattr(value, 'size', None)}, mode={getattr(value, 'mode', None)}")
            elif isinstance(value, (str, int, float, bool)) or value is None:
                print(f"{key}: {_safe_value_repr(value)}")
            else:
                print(f"{key}: {type(value).__name__} {_safe_value_repr(value)}")

    print("\nImage export check:")
    for i in range(min(2, len(ds))):
        ex = ds[i]
        image = ex.get("image")
        if image is None:
            print(f"sample {i}: no image field found")
            continue
        try:
            image.save(f"/tmp/bbqv_inspect_{i}.png")
            print(f"sample {i}: image saved successfully to /tmp/bbqv_inspect_{i}.png")
        except Exception as exc:  # pragma: no cover
            print(f"sample {i}: image export failed: {type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the official BBQ-V dataset schema and images.")
    parser.add_argument("--split", default="train[:5]", help="Dataset split selector. Default: train[:5]")
    parser.add_argument("--limit", type=int, default=3, help="Number of sample rows to print. Default: 3")
    args = parser.parse_args()
    inspect(args.split, args.limit)


if __name__ == "__main__":
    main()
