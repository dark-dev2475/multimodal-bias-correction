# Version 2 — local 40-image development study

**Provisional: AI-drafted references and same-model AI judgments, not human ground truth.**

Generator/judge: Qwen3-VL-2B-Instruct, Q4_K_M weights, Q8_0 vision projector, local llama.cpp Vulkan. No paid API calls.

EIGR selection/intervention algorithms are unchanged. All methods use JSON-schema decoding; task-derived source/quotation and judgment-ID constraints are enforced where applicable. Context quotes are restricted to full context or exact sentence spans. This is a transport adaptation, not a published reproduction.

## Coverage and failures

| Method | Planned | Outputs | Failures | Reviewed | Safe & correct / all planned* |
|---|---:|---:|---:|---:|---:|
| raw | 40 | 40 | 0 | 0 | 0/40 |
| eigr | 40 | 40 | 38 | 0 | 0/40 |
| self_refine_vqa | 40 | 15 | 4 | 0 | 0/40 |
| intent_aware_vqa | 40 | 0 | 0 | 0 | 0/40 |
| reflexion_vqa | 40 | 0 | 0 | 0 | 0/40 |
| constitutional_critique_vqa | 40 | 0 | 0 | 0 | 0/40 |
| always_abstain | 40 | 0 | 0 | 0 | 0/40 |

*Observed safe-correct AI labels divided by **all 40 planned examples**, including failures and missing labels in the denominator. This is a conservative operational lower bound, not a validated accuracy measurement.

## Descriptive AI estimates on reviewed responses only

| Method | Task correct | Unsupported social assertion | Required-fact recall | Token F1 (lexical only) |
|---|---:|---:|---:|---:|
| raw | not available | not available | not available | 37.3% |
| eigr | not available | not available | not available | 10.7% |
| self_refine_vqa | not available | not available | not available | 36.3% |
| intent_aware_vqa | not available | not available | not available | not available |
| reflexion_vqa | not available | not available | not available | not available |
| constitutional_critique_vqa | not available | not available | not available | not available |
| always_abstain | not available | not available | not available | not available |

## Interpretation limits

- The references were frozen before this model produced answers and were never supplied to inference.
- References remain pending human review. Generator and semantic judge share weights: errors may be correlated.
- Reference answerability: {'insufficient_evidence': 39, 'answerable': 1}. This heavily ambiguous subset cannot establish preservation of usefulness on answerable tasks.
- All 40 images are previously exposed development data, not a new held-out test set.
- Failed EIGR outputs are not credited as successful abstentions. Review failures remain missing, never silently fixed to improve scores.
- Self-Refine, Intent-Aware, Reflexion and Constitutional critique are inference-only VQA adaptations under a maximum nine-call budget (including the shared draft where applicable). Constitutional AI's full training method is not reproduced.
- No match with published scores, superiority, or statistically established novelty is claimed.

## Files to show

- `EXAMPLES.md`: all 40 reference/draft/EIGR answers with status and JSON links.
- `all_outputs.json`: every collected method output, including EIGR evidence graphs and repair certificates.
- `metrics.json`: coverage, failures, lexical diagnostics and explicitly labelled AI estimates.
- `per_response_scores.json`: separate testing scores for each planned method/image pair.
- Study `manual_annotations/` and `blind_review/`: reference drafts and blank human-review forms.
