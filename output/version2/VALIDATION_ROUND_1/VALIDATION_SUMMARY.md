# Validation round 1 — results and next required step

**Validation tools and diagnostics are complete; scientific validation is not.** The old experiment is preserved. No paper-score match or superiority is established.

## What was completed

- Audited all 40 original evidence graphs and reference records structurally.
- Prepared 40 separate human-reference forms; all remain pending human review.
- Ran 24 frozen synthetic text cases through three checker configurations (72 results).
- Independently analysed all 237 valid saved outputs, retaining 43 failed attempts as unscored.
- Added no training, paid inference, or human intervention to the inference pipeline.

## Controlled text calibration — not project accuracy

| Checker | Binary support decisions correct | False support decisions | Exact three-class decisions |
|---|---:|---:|---:|
| Original Qwen review format | 13/24 | 11/16 | Not comparable: legacy unsupported label combines cases |
| Short pointwise Qwen prompt | 24/24 | 0/16 | 16/24 |
| Independent DeBERTa NLI | 24/24 | 0/16 | 22/24 |

The binary test asks only whether support exists. The pointwise model called all eight contradictions unknown; DeBERTa called two unknown claims contradicted. These errors are retained. The 24 hand-authored cases share eight fact families and are not independent held-out BBQ-V examples. No claim of 100% multimodal accuracy is justified.

## Real-data failure after the calibration pass

For case `02_11_0780_2_01`, the independent NLI checker said the answer “The available evidence does not establish an answer” entails “The office is shared between two individuals in the image.” This is a false entailment. It shows that passing the small synthetic gate is insufficient for real-data fact coverage. The checkpoint is **not approved for validated semantic scoring**. Outputs remain unchanged.

Reference-text fractions are retained only as exploratory diagnostics. They do not inspect images, classify social bias, or determine task correctness. Justified uncertainty can receive low text entailment; incomplete or inconsistent reference facts also affect scores.

## Why cue removal did not run

39/40 graphs had no tagged cue. The remaining graph had no path from a tagged cue to its social claim. Of 696 observations, 578 repeat an earlier text/source entry. The planner therefore had no eligible intervention, rather than insufficient API budget. These findings identify upstream graph quality as an experimental bottleneck, not a proven causal model failure.

## What a person must do next

Open HUMAN_REFERENCE_REVIEW.md and review its 40 linked images/JSON forms, preferably without seeing system answers. Approve or correct the evidence, answerability, required facts and neutral reference answer. Do not mark AI drafts as human-approved without actually reviewing them.

Then separately review blinded system answers, including uncertainty responses and explicit answerable controls. Human offline evaluation does not violate the proposed autonomous inference requirement: no human is inserted into the prediction/correction pipeline.

Before another performance study: freeze the corrected reference version, test cue/link extraction separately, expand checker calibration to real abstention/fact-coverage failures, and use a genuinely held-out answerability-balanced set. Those steps are not completed here.

## Files to open

- CALIBRATION_REPORT.md / calibration_comparison.png: controlled text tests.
- FAILURE_MECHANISM_AUDIT.md: actual graph prerequisites and duplicate counts.
- HUMAN_REFERENCE_REVIEW.md: all 40 human-review links.
- independent_reference_text_40.json: per-EIGR-unit/fact diagnostic.
- INDEPENDENT_TEXT_REPORT.md: all-method text diagnostic, not accuracy.
- checker_reliability_audit.json: exact retained calibration errors and real-data counterexample.
