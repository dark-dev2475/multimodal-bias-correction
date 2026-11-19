# BBQ-V expanded project partitions — Version 2

This directory expands the original **40-example pilot to 14,578 examples**
using the already-authorised, cached `ucf-crcv/BBQ-V` dataset. No examples were
fabricated or oversampled. The original `data/bbqv/` pilot is unchanged.

## Actual contents

| Project partition | Examples | Distinct images | Connected groups | Categories |
|---|---:|---:|---:|---:|
| Train / development | 12,364 | 3,780 | 30 | 9 |
| Validation | 1,142 | 455 | 30 | 8 |
| Held-out test | 1,072 | 442 | 38 | 9 |
| Total | 14,578 | 4,677 | 98 | 9 |

**Train means method development/calibration, not model-weight training.**
The project remains training-free. The dataset has no open-ended gold-answer
field, so these manifests are not a ready-made supervised fine-tuning dataset.
Use independent annotations for task-correctness and bias-outcome evaluation.

These are **custom project partitions of the source's `test` release**, not
official train/validation/test splits. They are not directly interchangeable
with published full-benchmark scores.

## Provenance and storage

- Dataset: [official BBQ-V card](https://huggingface.co/datasets/ucf-crcv/BBQ-V).
- Source release/cache revision: `a1c78b8f73bc40408993414e3d94714a6a9169d3`.
- The inspected cached Arrow shards and their metadata contain **14,578 rows**.
  The current public card describes **14,144 pairs**. This discrepancy is
  explicitly retained: these results describe the pinned cached release, not
  a claim that today's advertised release contains 14,578 pairs.
- Images are exported once per decoded RGB pixel hash, using original encoded
  bytes; no resizing, augmentation, or synthetic image generation is performed.
- Exported images occupy 1,867,284,458 bytes (about 1.74 GiB). No download was
  required. `images/` is Git-ignored; use the builder to reproduce it from the cache.
- Dataset licence: **CC BY-NC 4.0**. Respect the source access terms, attribution,
  and non-commercial restriction. Do not assume locally available gated data
  is suitable for unrestricted redistribution.

## Leakage-control procedure

1. Group the first two official ID fields (`biasCategory_questionIndex`). This
   keeps all example, image, and polarity variants of a question family together.
2. Union examples sharing an exact decoded RGB pixel SHA-256.
3. Union images with 128-bit horizontal/vertical dHash Hamming distance at most 4.
4. Take transitive connected components across all three link types.
5. Force every component containing any of the original 40 exposed example IDs
   into train/development. Those 13 components contain 11,638 examples.
6. Allocate remaining components deterministically (seed 42), targeting 70/15/15
   with a category-balance objective. Prioritise test category coverage if the
   last unexposed group for a category would otherwise go only to validation.

The largest component has 10,030 rows. Consequently the achieved ratio is
**84.81% / 7.83% / 7.35%**, not 70/15/15. We do not break connected groups just
to obtain attractive percentages.

**Validation has no Sexual_orientation examples.** All but one component for
that category are already exposed; the remaining 44 examples are reserved for
test. Do not claim all-category validation or tune a category-specific threshold
for that category from validation. Freeze global settings using development
data, then report this coverage limitation with the final test results.

dHash is a heuristic: this process blocks detected near-duplicate links, not
every possible semantic duplicate or pretraining exposure. It may also merge
unrelated images with similar hashes. The grouping is deliberately conservative.

## Files and reproduction

- `source_index.json`: all source row metadata, original encoding hashes, pixel
  hashes, perceptual hashes, dimensions, and source indices; also enables resume.
- `{train,validation,test}_manifest.json`: loader-compatible examples and split,
  source-revision, question-family, and leakage-group provenance.
- `split_metadata.json`: exact counts, category coverage, overlap audit, seed,
  source identity, and partition fingerprint.
- `images/`: the 4,677 deduplicated local images referenced by the manifests.

From the repository root in PowerShell:

```powershell
venv/Scripts/python.exe scripts/build_bbqv_splits.py
python scripts/audit_bbqv_splits.py --verify-images
python scripts/run_v2_experiment.py --manifest data/bbqv_v2/train_manifest.json --num-examples 3 --systems raw generic full no_interventions --output-dir data/outputs/v2/demonstration_dev
```

The existing virtual environment has `datasets`, Arrow and Pillow. The builder
uses the pinned local cache and never silently downloads another revision.
It resumes indexing, verifies existing exports, and reserves at least 2 GiB
free disk space before export. Pass `--source-cache` for another authenticated
cache location. Use a fresh output directory for a different source revision.

The final audit verified **4,677 file checksums**, exact source-ID coverage,
and the saved partition fingerprint. Cross-split ID, pixel-hash, question-family,
component, and detected near-duplicate overlaps were all zero. All 40 exposed
IDs remained in development.

At the initial dataset-expansion handoff, only development examples had been
model-evaluated. A subsequent frozen baseline study selected 36 test examples;
11 completed before provider access blocked the run. See
`output/version2/baseline_comparison/REPORT.md`. These observed test outcomes must
not be used for tuning while still claiming that sample is held out. Validation
has not been model-evaluated. Reading manifests for split audits is not an outcome
evaluation. Run a dry run before planning large paid experiments:

```powershell
python scripts/run_v2_experiment.py --manifest data/bbqv_v2/test_manifest.json --num-examples 1072 --systems raw generic full --dry-run
```

Do not remove `--dry-run` until the method is frozen and the inference budget is
accepted. Report confidence intervals using **leakage-group resampling**, not
1,072 independent test observations. The V2 runner now carries group IDs into
the private review key, and the evaluator automatically uses a paired cluster
bootstrap when those IDs are present. Fewer than two groups produces no interval.

Partition fingerprint:
`a4e23a967ebc5c60c6d95eed5908d374c0bac1a0489235d0e1c0cf7d39e61668`.
