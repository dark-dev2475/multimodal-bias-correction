# Multimodal Bias Correction

Inference-time auditing and correction of unsupported social assumptions in
vision-language model responses, without updating model weights.

**Free-only policy:** demo, final pipeline, baselines and automatic annotation
use free vision models; paid IDs and remote unverified endpoints are rejected.
No purchase is required. See [V2 section 14](README_VERSION_2.md#14-free-only-demo-final-pipeline-and-comparison).
Historical paid-model records below are archived, not the current configuration.

- **[Version 2 — Evidence Intervention and Graph Repair](README_VERSION_2.md):**
  research method, equations, code changes, runnable demos, local/hosted model
  options, evaluation protocol, and limitations.
- [Version 1 project guide](PROJECT_GUIDE.md): the existing six-stage system,
  dataset preparation, original baselines, and annotation workflow.
- [Four runnable V2 mechanism demonstrations](output/version2/NOVELTY_DEMONSTRATION.md):
  controlled ablations, exact optimisation, and a separately labelled live pilot.
- [Expanded BBQ-V partitions](data/bbqv_v2/README.md): 12,364 development,
  1,142 validation, and 1,072 held-out test examples with leakage audits.
- [Paper-based baseline comparison](output/version2/baseline_comparison/REPORT.md):
  implemented Self-Refine/Intent-Aware VQA adaptations; 11/36 cases evaluated
  before provider credit/access blocked completion. No performance match claimed.

Quick start (scripted software demonstration, no API key or network):

```powershell
python scripts/run_v2.py --demo --output data/outputs/v2/demo.json
python scripts/demonstrate_v2.py
python -m pytest -q tests/test_research_v2.py
```

The demo is hand-authored and is not a claim of measured model performance.
The partition named `train` is for method development; no model weights are trained.

Live demo/final pipeline (free OpenRouter key in local `.env`):

```powershell
python scripts/run_v2.py --image data/raw/images/wheelchair.jpg --output data/outputs/v2/free_live_demo.json
python scripts/run_v2_baseline_study.py
python scripts/export_v2_baseline_study.py --allow-incomplete
```

The study is resumable by rerunning the same command after quota resets. Free
availability is not guaranteed. The scripted demo works offline regardless.

### Reference-first study: original 40 images

[Protocol, metrics and commands](output/version2/REFERENCE_STUDY_PROTOCOL.md).
One reference JSON per image is prepared before output collection; AI drafts need
human review. Output generation never receives these references. Testing runs
separately with `python scripts/evaluate_reference_study.py`. This is development
evaluation, not published-score reproduction or proof of superiority.
