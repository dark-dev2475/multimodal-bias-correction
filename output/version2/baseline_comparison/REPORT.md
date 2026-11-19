# Version 2 baseline comparison — incomplete

**11 of 36 selected cases completed. No baseline match has been established.**

Stop reason: Provider access/rate limit stopped study: Model endpoint returned HTTP 402.

Generator: `inclusionai/ling-3.0-flash-vl`. External evaluator: `nvidia/nemotron-3.5-super-vl:free`.

Self-Refine and Intent-Aware are documented VQA adaptations, not reproductions of published numerical scores. The completed prefix is not category-balanced, so no aggregate performance table is released. Do not reinterpret unprocessed examples as model errors or zero-accuracy observations.

Saved outputs: 77. Successfully scored outputs: 67. Model failures remain in the records; missing human labels have not been invented.

The original paid-evaluator attempt is preserved separately. Free NVIDIA evaluation succeeded, but new generator requests subsequently returned HTTP 402. Restore provider credit/access, then rerun the same frozen command. Completed results, including failures, are reused; no generator or threshold change is permitted under the same protocol.

```powershell
python scripts/run_v2_baseline_study.py --judge-model nvidia/nemotron-3.5-super-vl:free --judge-mode batch --judge-tokens 3000 --reuse-generations-from data/outputs/v2/baseline_study_36 --output-dir data/outputs/v2/baseline_study_36_free_judge --max-cost-usd 0.06 --workers 2
python scripts/export_v2_baseline_study.py --study data/outputs/v2/baseline_study_36_free_judge
```

The $0.06 continuation cap retains the earlier conservative allowance, keeping the combined request-charge ceiling below $1. If the remaining allowance is insufficient, stop and obtain a new explicit budget. The free evaluator may also be limited by daily quota or upstream availability.
