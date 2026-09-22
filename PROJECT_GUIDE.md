# Multimodal Bias Correction

## 1. Project Summary

This project studies inference-time bias correction for vision-language models (VLMs).

Research question:

> Can a multimodal AI system detect unsupported or bias-relevant assumptions in its generated output during inference and revise the output using visual evidence, without additional training or full model retraining?

The system does not fine-tune a model. It analyzes and revises a response after the initial VLM generation.

## 2. Core Architecture

```text
Image + question
      |
      v
Initial VLM generation
      |
      v
Claim extraction
      |
      v
Batch visual evidence checking
      |
      v
Bias monitoring
      |
      v
Self-correction when required
      |
      v
Final verification
      |
      v
Structured result + provenance
```

The implementation is in `src/pipeline/`:

- `generator.py`: sends the normal image/question prompt to the VLM.
- `claim_extractor.py`: separates a response into claims.
- `evidence_checker.py`: checks claims against the image.
- `bias_monitor.py`: identifies unsupported or bias-relevant assumptions.
- `corrector.py`: rewrites responses when flagged claims require correction.
- `verifier.py`: checks the final response as a diagnostic.
- `pipeline.py`: coordinates the complete six-stage workflow.

The initial prompt is intentionally conservative. It asks for information directly supported by visual evidence and prohibits unsupported inferences about medical status, occupation, socioeconomic status, religion, nationality, relationships, personality, and intentions.

## 3. Repository Map

```text
app/
  main.py                    Manual/local pipeline entry point
  run_benchmark.py           Existing benchmark runner
  run_baseline.py            Run one legacy benchmark system
  compare_baselines.py       Existing three-system comparison utility
  run_ablation.py            Ablation experiment runner
  evaluate_results.py        Ground-truth evaluation entry point
  analyze.py                 Saved-run analysis utility

src/models/
  vlm.py                     OpenRouter VLM wrapper, retries, usage logging

src/pipeline/
  pipeline.py                Main correction pipeline
  generator.py               Initial VLM generation
  claim_extractor.py         Claim extraction
  evidence_checker.py        Visual evidence checks
  bias_monitor.py            Bias-relevant claim analysis
  corrector.py               Self-correction
  verifier.py                Final verification

src/evaluation/
  baselines.py               VLM-only, generic correction, proposed systems
  bbqv_loader.py             Local BBQ-V manifest adapter
  ground_truth.py            Human ground-truth contract and validation
  metrics.py                 Claim-level evaluation metrics
  provenance.py              Git, pricing, and experiment provenance

src/utils/
  schemas.py                 Pydantic result and benchmark schemas
  benchmark.py               Existing benchmark loading and hashing
  logger.py                  Saved run logging
  chache.py                  File cache used by pipeline stages
  analysis.py                Aggregation of saved experiment outputs

scripts/
  inspect_bbqv.py            Inspect official BBQ-V schema/access
  create_bbqv_benchmark.py   Create deterministic local BBQ-V subset
  run_bbqv_sample.py         One-image BBQ-V integration runner
  run_bbqv_experiment.py     Phase 4 three-system experiment runner

 data/
  annotations/               Existing benchmark and human-label artifacts
  bbqv/                      Local BBQ-V subset and manifest
  raw/images/                Existing project image data
  cache/                     Local stage cache; ignored by Git
  outputs/                   Generated results; ignored by Git

tests/
  test_bbqv_benchmark.py     BBQ-V subset integrity tests
  test_bbqv_loader.py        Adapter, runner, and output-schema tests
  test_bbqv_experiment.py    Experiment selection/cache/report tests
  test_*.py                  Existing pipeline and evaluation tests
```

## 4. BBQ-V Benchmark Work

The primary real-image benchmark is the official Hugging Face dataset:

- Dataset: `ucf-crcv/BBQ-V`
- Repository: `https://github.com/UCF-CRCV/BBQ-Vision`
- Split used: `test`
- Full verified split size: `14,578`
- Verified categories: Age, Disability_status, Gender_identity, Nationality, Physical_appearance, Race_ethnicity, Religion, SES, Sexual_orientation

The local subset is in `data/bbqv/`:

- `images/`: 40 exported real BBQ-V images
- `benchmark_manifest.json`: IDs, paths, categories, source metadata, questions, and selection details
- `dataset_metadata.json`: dataset size, category counts, seed, timestamp, and provenance
- `README.md`: subset policy and ground-truth separation

The subset was selected deterministically with seed `42`, target size `40`, category diversity, stable filenames, and image-content deduplication.

Important interpretation rule:

> A BBQ-V category is benchmark metadata, not claim-level ground truth.

For example, a `Disability_status` category does not automatically make every disability-related claim false or problematic. Later human annotation must label each generated claim as supported, unsupported, or uncertain, and as problematic or not problematic.

## 5. Completed Phases

### Phase 1: Dataset inspection

Completed. The official BBQ-V dataset was authenticated, loaded, and inspected. The real schema and image representation were verified rather than guessed.

### Phase 2: Local subset

Completed. Forty real images were exported locally. The manifest is reproducible and all selected images are unique and readable.

### Phase 3: Pipeline integration

Completed. One real image was sent through the existing pipeline without changing the core architecture.

Observed integration example:

- Example ID: `02_22_1450_1_01`
- Claims extracted: `3`
- Unsupported claims detected: `2`
- Bias flags: `1`
- Correction: triggered
- Final verification: passed
- API calls: `6`

Saved result:

`data/outputs/bbqv_phase3_sample/02_22_1450_1_01/20260922T173751Z-438405.json`

### Phase 4: Experiment runner

The three-system runner is implemented in `scripts/run_bbqv_experiment.py`.

It supports:

1. `vlm_only`: image and question to the VLM, with no correction stages.
2. `generic_self_correction`: initial VLM response followed by a generic review prompt.
3. `proposed`: the existing extraction, evidence, bias monitoring, correction, and verification pipeline.

The runner uses the same selected examples, questions, model configuration, and image paths for all systems. BBQ-V categories and human labels are not sent to any system.

The required three-image smoke run was attempted but was blocked by the OpenRouter free-model daily quota:

```text
429 RateLimitError
free-models-per-day
remaining: 0
```

No fake or partial model outputs were created. The experiment can be resumed with the same command after the quota resets or credits are added.

## 6. Baselines and Fairness

The comparison is descriptive until independent human annotations exist.

All systems must use:

- the same BBQ-V image subset
- the same image for each corresponding example
- the same model
- the same relevant question
- the same selection seed
- recorded prompt/configuration provenance

The experiment runner stores system-specific cache keys containing the system, model, prompt/version, image/example, question, and iteration configuration. A VLM-only cache entry cannot be reused as a proposed-pipeline result.

The automated verifier is a diagnostic only. It is not ground truth and must not be used to claim accuracy, precision, recall, or superiority.

## 7. Ground Truth and Evaluation

Human annotations are intentionally separate from model outputs and verifier diagnostics.

The evaluation layer supports:

- claim-level precision, recall, and F1
- false positives and false negatives
- unsupported claim removal
- supported information retention
- over-correction
- correction success
- residual problematic claims
- latency, iterations, API calls, cache usage, and estimated cost

Research claims should only be made after independent human annotation of initial and final outputs.

## 8. Reproducibility

Important recorded fields include:

- dataset and split
- benchmark version and manifest hash
- selected example IDs
- random seed
- model name
- prompt version/hash
- maximum iterations
- Git commit
- timestamp
- API calls and latency when available
- cache status
- pricing configuration

The local Hugging Face dataset cache is external to the repository. No Hugging Face token or API key belongs in source files, manifests, documentation, or commits.

## 9. How to Run

Use the project virtual environment on Windows:

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
```

Run the tests:

```powershell
venv\Scripts\python.exe -m pytest -q
```

Inspect BBQ-V access/schema:

```powershell
venv\Scripts\python.exe scripts\inspect_bbqv.py --split "test[:2]" --limit 2
```

Recreate the local subset:

```powershell
venv\Scripts\python.exe scripts\create_bbqv_benchmark.py --target-size 40 --seed 42
```

Run one BBQ-V integration sample:

```powershell
venv\Scripts\python.exe scripts\run_bbqv_sample.py --num-examples 1 --seed 42
```

Run the Phase 4 smoke experiment after API quota is available:

```powershell
venv\Scripts\python.exe scripts\run_bbqv_experiment.py `
  --num-examples 3 `
  --seed 42 `
  --systems vlm_only generic_self_correction proposed `
  --max-iterations 2
```

The smoke experiment writes to `data/outputs/bbqv_experiment/`:

```text
data/outputs/bbqv_experiment/
  vlm_only/<example_id>/vlm_only.json
  generic_self_correction/<example_id>/generic_self_correction.json
  proposed/<example_id>/proposed.json
  baseline_comparison.json
  experiment_manifest.json
```

## 10. Professor Presentation Flow

A concise presentation can follow this order:

1. **Problem:** VLMs can produce visually unsupported assumptions about people.
2. **Research question:** Can inference-time analysis correct those assumptions without retraining?
3. **Method:** Generate, extract claims, check visual evidence, monitor bias, correct, verify.
4. **Benchmark:** Real images from official BBQ-V, with a reproducible 40-image subset.
5. **Baselines:** Raw VLM, generic self-correction, and the structured pipeline.
6. **Evidence so far:** One real BBQ-V image completed the full pipeline; one bias flag triggered correction and final verification passed.
7. **Research validity:** BBQ-V categories are metadata; independent claim-level human labels are required for performance claims.
8. **Current limitation:** The three-image baseline smoke comparison is pending OpenRouter quota availability.
9. **Next step:** Run the three-system three-image smoke test, inspect fairness and outputs, then run the full subset only after validation.

## 11. Current Status

The project has a working research pipeline, real-image BBQ-V subset, adapter, one-image integration, evaluation scaffolding, provenance tracking, and API-free tests.

The next evidence-producing action is the three-image smoke comparison after the OpenRouter quota becomes available. The full 30–50 image benchmark and final research metrics must wait until that smoke comparison is verified and independent human annotations are prepared.
