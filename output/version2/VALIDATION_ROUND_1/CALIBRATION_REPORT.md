# Validation round 1 — controlled text calibration

**Follow-up caution:** binary calibration passes did not establish real-data reliability. Exact three-class results and a retained real-data false entailment are documented in VALIDATION_SUMMARY.md and checker_reliability_audit.json.

Assistant-authored synthetic text contrast cases; not human image labels or held-out BBQ-V evidence.

A pass permits only further TEXT diagnostic use, not image-grounded accuracy claims. This is a small synthetic stress suite, not an external benchmark or human annotation.

| Checker | Valid / 24 | Binary support correct / 24 | False entailments / 16 | Entailed recall / 8 | Gate |
|---|---:|---:|---:|---:|---|
| nli_deberta | 24/24 | 24/24 | 0/16 | 8/8 | True |
| qwen_legacy | 24/24 | 13/24 | 11/16 | 8/8 | False |
| qwen_pointwise | 24/24 | 24/24 | 0/16 | 8/8 | True |

Gate frozen before testing: complete valid coverage, zero false entailments, and at least 7/8 entailed cases recognized. Missing and failed cases fail the gate.

The legacy judge is tested with its original large review schema but without images. The pointwise prompt asks one entailment question and returns one label. Differences can reflect task/schema complexity, not an isolated causal diagnosis. No old labels are replaced.

Independent checker: [official DeBERTa NLI model card](https://huggingface.co/cross-encoder/nli-deberta-v3-small). It was already pretrained by its authors; this project performs no training. All learned checkers remain fallible; this suite does not establish fairness or novelty.
