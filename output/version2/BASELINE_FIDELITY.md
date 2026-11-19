# Baseline fidelity and limits — Version 2

## What is being compared

This implementation distinguishes an algorithmic **VQA adaptation**, a
**same-task measured comparison**, and an **exact published-score reproduction**.
They are different achievements. Only the first two are attempted in this bounded
study. A method's name is not evidence that its original published score was reproduced.

| System | Implemented behavior | Important deviation |
|---|---|---|
| raw | One answer from image, question and context | Project generator prompt |
| generic | One feedback-free rewrite of that answer | Simple project baseline |
| self_refine_vqa | Separate feedback and revision calls, accumulated history, early stop, up to four cycles | Original zero-shot VQA prompts; different task and backbone |
| intent_aware_vqa | Own debiasing/explanation prompt, three binary feedback aspects, early stop at 3/3, up to three repairs | Zero-shot, open-ended multimodal adaptation; not original prompts or text-only sample |
| eigr | Existing graph, intervention planner, exact repair and verification | No changes after inspecting test outcomes |
| eigr_no_interventions | Same V2 with probes disabled | Ablation, not an external baseline |
| always_abstain | Fixed uncertainty response | Diagnostic for trivial success on ambiguous questions |

The [Self-Refine paper](https://arxiv.org/html/2303.17651v2) and
[official implementation](https://github.com/madaan/self-refine) motivate the
feedback/revision loop with retained history. Its reported tasks and prompts are
not the BBQ-V task used here. These are clean-room task-specific implementations,
not a vendored copy of the authors' code.

[Intent-Aware Self-Correction](https://arxiv.org/html/2503.06011v1) uses
text-only multiple-choice BBQ, a balanced 2,118-example ambiguous-context subset,
three repetitions, and model-specific settings. Our VQA adaptation preserves its
three-aspect feedback and stopping logic but does not reproduce its few-shot
prompts, conversation setup, selected sample, or numerical table. Its headline
accuracy therefore is **not a threshold that can honestly be matched using our
BBQ-V scores**.

Self-Debias and Constitutional AI involve training; they are not reproduced by
this frozen-model experiment. Reflexion and the survey remain related-work
references, not implemented baselines. No result for them should appear as an
experiment in this project.

## Frozen bounded protocol and availability fallback

- 36 held-out examples, four per category, selected before inference with seed
  20261102. Group-diverse selection gives 30 connected groups and distinct images.
- Generator: `inclusionai/ling-3.0-flash-vl` for all active methods.
- Originally selected evaluator: `openai/gpt-4o-mini-2024-07-18`. Its registry entry was checked
  in the provider registry and its image-input support against
  [official model documentation](https://developers.openai.com/api/docs/models/gpt-4o-mini).
- Same image/context/question for each method. Shared initial draft for raw,
  generic, Self-Refine, EIGR and the ablation. Intent-Aware generates its own initial
  answer because its initial instruction is part of the method.
- Nine-call ceiling including shared generation for Self-Refine/EIGR; at most
  eight calls for Intent-Aware's independent path. Actual calls are reported.
  Equal ceilings are not identical token/latency expenditure; this is not an
  exact realised-cost-matched comparison.
- The original model judge was to evaluate anonymous individual answers in shuffled order.
  Identical answers on the same case share one judgment. The judge receives no
  system names, internal graphs, feedback, expected labels or benchmark category.
- Primary metrics: judge-rated task correctness, usefulness, and conservative
  worst-case unsupported-assumption rate. Missing output/labels are not dropped.
- Paired bootstrap resamples complete leakage groups. No favourable point
  estimate or zero-width bootstrap interval is automatically declared a match.
- No training or selection of the best seed. No V2 prompt/threshold tuning on
  these test outcomes. The full test set is not run.
- A $1 advertised request-charge budget is guarded by full-context reservations.
  Price snapshots, actual reported charges and conservative unknown-usage charges
  are recorded. This does not cover taxes or external account fees. A persistent
  journal protects the budget across interrupted calls/resumes.

`protocol.json` is written before model inference; prompts and implementation
are content-hashed. Completion/failure records are retained. Resume refuses a
different protocol or source version. It does not retry away failed observations.

The paid evaluator returned HTTP 402. A free Gemma evaluator returned upstream
HTTP 429. A free `nvidia/nemotron-3.5-super-vl:free` evaluator then passed an
availability check and was frozen in a **new protocol**. It evaluates an anonymous
batch of unique answers per image, in shuffled order, with exact ID coverage
validation. Batching reduces requests but may introduce within-batch anchoring;
it is disclosed rather than treated as identical to isolated judging.

Completed generator records from the original attempt, including failures, were
reused only after verifying their model, selected cases, token limit and generation
source hashes. Generation/repair modules and thresholds did not change. Windows
atomic-write retries and fail-fast provider-access handling were added to the
study runner; the original runner/source snapshot is preserved in the first attempt.

The continuation was capped at $0.06. Added to the original interrupted run's
$0.931714 conservative accounting/reservations, this keeps the combined allowance
below $1. These figures are **not a claim of actual charges**: failed requests with
unknown usage retain a full-context reservation. Many declined requests likely did
not incur those amounts. Actual API-reported usage remains in the raw call logs.

The generator subsequently returned HTTP 402 as well. The continuation stopped
with **11/36 cases complete**, and the remaining 25 were not scored. The API account
reported zero purchased credits. Access must be restored before the frozen run can
finish; no paper or baseline-performance match has been established.

## Reproduction commands

```powershell
python scripts/run_v2_baseline_study.py --dry-run
python scripts/run_v2_baseline_study.py --judge-model nvidia/nemotron-3.5-super-vl:free --judge-mode batch --judge-tokens 3000 --reuse-generations-from data/outputs/v2/baseline_study_36 --output-dir data/outputs/v2/baseline_study_36_free_judge --max-cost-usd 0.06 --workers 2
python scripts/export_v2_baseline_study.py --study data/outputs/v2/baseline_study_36_free_judge --allow-incomplete
```

The runner is intentionally separate from the old collection CLI, preserving
earlier experiments. Raw records live in `data/outputs/v2/baseline_study_36/`
(interrupted paid-evaluator attempt) and `data/outputs/v2/baseline_study_36_free_judge/`
(incomplete free-evaluator continuation).
Exported results and unfilled human-review forms live in
`output/version2/baseline_comparison/`. The automatic annotations are explicitly
labelled as model estimates. Do not give the private system key or automated
labels to a human reviewer doing a blinded independent evaluation.

## What would establish a stronger claim

For published-score reproduction, obtain the authors' exact data selection and
prompts, compatible frozen model snapshots, decoding settings, repeated seeds and
original metrics. Record unavailable details rather than filling them in silently.

For a claim about this multimodal method, use human-reviewed labels, larger
pre-specified samples, multiple backbones, grounded answerable controls and
reported call/token/cost tradeoffs. Low bias obtained through generic abstention
is not sufficient. The current bounded study can reveal failures or preliminary
differences, but it is not a publication-ready equivalence or superiority test.
