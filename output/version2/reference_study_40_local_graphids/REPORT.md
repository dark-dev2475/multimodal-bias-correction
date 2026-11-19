# Version 2 — local graph-ID engineering study

**Development-informed change:** Claim identity/link constraints added after observing graph failures on these original development cases. Dependencies restricted to earlier draft units after a separate demo produced a cycle. This ordered-DAG representation cannot encode forward-premise dependencies. Not held-out validation or evidence of generalization.

**Provisional: AI-drafted references and same-model AI judgments, not human ground truth.**

Generator/judge: Qwen3-VL-2B-Instruct, Q4_K_M weights, Q8_0 vision projector, local llama.cpp Vulkan. No paid API calls.

EIGR selection/intervention algorithms are unchanged. All methods use JSON-schema decoding; task-derived source/quotation and judgment-ID constraints are enforced where applicable. Context quotes are restricted to full context or exact sentence spans. This is a transport adaptation, not a published reproduction.

## Coverage and failures

| Method | Planned | Outputs | Failures | Reviewed | Safe & correct / all planned* |
|---|---:|---:|---:|---:|---:|
| raw | 40 | 40 | 0 | 4 | 4/40 |
| eigr | 40 | 40 | 0 | 4 | 4/40 |
| self_refine_vqa | 40 | 18 | 6 | 4 | 4/40 |
| intent_aware_vqa | 40 | 0 | 0 | 0 | 0/40 |
| reflexion_vqa | 40 | 0 | 0 | 0 | 0/40 |
| constitutional_critique_vqa | 40 | 0 | 0 | 0 | 0/40 |
| always_abstain | 40 | 0 | 0 | 0 | 0/40 |

*Observed safe-correct AI labels divided by **all 40 planned examples**, including failures and missing labels in the denominator. This is a conservative operational lower bound, not a validated accuracy measurement.

## Descriptive AI estimates on reviewed responses only

| Method | Task correct | Unsupported social assertion | Required-fact recall | Token F1 (lexical only) |
|---|---:|---:|---:|---:|
| raw | 100.0% | 0.0% | 100.0% | 37.3% |
| eigr | 100.0% | 0.0% | 100.0% | 22.0% |
| self_refine_vqa | 100.0% | 0.0% | 100.0% | 36.7% |
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

## Change from the initial local study

Observation source first; empty context forbids context source; context quotes restricted to full context or exact sentences; judgment IDs/counts bound to requested IDs. Does not verify visual truth. Claim IDs/texts assigned deterministically in input unit order; dependencies limited to earlier claims; contradiction IDs limited to other known claims; evidence references must have observation-ID syntax. Core DAG/existence checks retained. Ordered-DAG constraint can miss forward-premise links.

The initial local run is preserved separately, including all graph failures. No benchmark answers or reference labels were used to choose support labels or repair outputs.

## Completion and mechanism audit

All 280 planned method/image attempts recorded: **False**. Every valid output has an AI review: **False**. Failed inference attempts are not valid outputs; missing semantic labels remain missing. Human approval remains pending.

| Mechanism | Cases |
|---|---:|
| Saved evidence graph | 40 |
| Non-empty dependency edges | 9 |
| Measured cue-removal/control contrast | 0 |
| Verification rejected at least one selected claim | 16 |
| Final answer differs from draft (not necessarily an improvement) | 30 |

Metric denominators are explicit in metrics.json. False-abstention estimates concern only the single draft-labelled answerable case. See mechanism_audit.json and the separate LOCAL_DEMO_AUDIT.md counterexample before making claims about the mechanism.

Compact exports: annotations_40.json, final_responses_40.json, and reference_similarity_40.json. These are evaluation artifacts and are never passed into inference.
