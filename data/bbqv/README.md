# BBQ-V Local Benchmark Subset

This directory contains a reproducible local subset of real images from the official `ucf-crcv/BBQ-V` Hugging Face dataset.

- Split: `test`
- Selection: deterministic category-diverse sampling
- Default target size: 40 images
- Default seed: 42
- Source images: real images exported from BBQ-V
- This is a subset, not the complete dataset.

The BBQ-V category identifies the benchmark scenario or bias probe. It is **not** claim-level ground truth. Later human annotations must independently label generated claims as supported, unsupported, or uncertain and as problematic or not problematic. Human annotations must remain separate from `benchmark_manifest.json`.

To recreate the subset after authenticating to Hugging Face:

```powershell
python scripts/create_bbqv_benchmark.py --target-size 40 --seed 42
```