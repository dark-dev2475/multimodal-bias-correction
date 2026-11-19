# Version 2 — 40-image study / bounded AI review

Automatic stage: **complete_with_recorded_failures**. Performance validation: **not validated**.

**JUDGE RELIABILITY FAILED A SPOT CHECK.** Spot checks found false support, missed bias, incorrect correctness and over-credited fact coverage. Numerical labels are trace data, not reliable accuracy/fairness measurements. See JUDGE_RELIABILITY_AUDIT.md. Human approval is still pending.

**Development-informed change:** Claim identity/link constraints added after observing graph failures on these original development cases. Dependencies restricted to earlier draft units after a separate demo produced a cycle. This ordered-DAG representation cannot encode forward-premise dependencies. Not held-out validation or evidence of generalization.

**Provisional: AI-drafted references and same-model AI judgments, not human ground truth.**

Generator/judge: Qwen3-VL-2B-Instruct, Q4_K_M weights, Q8_0 vision projector, local llama.cpp Vulkan. No paid API calls.

EIGR selection/intervention algorithms are unchanged. All methods use JSON-schema decoding; task-derived source/quotation and judgment-ID constraints are enforced where applicable. Context quotes are restricted to full context or exact sentence spans. This is a transport adaptation, not a published reproduction.

## Coverage and failures

| Method | Planned | Recorded attempts | Failures | Reviewed | Judge labelled safe/correct / all planned* |
|---|---:|---:|---:|---:|---:|
| raw | 40 | 40 | 0 | 40 | 29/40 |
| eigr | 40 | 40 | 0 | 40 | 36/40 |
| self_refine_vqa | 40 | 40 | 11 | 29 | 21/40 |
| intent_aware_vqa | 40 | 40 | 1 | 39 | 32/40 |
| reflexion_vqa | 40 | 40 | 14 | 26 | 16/40 |
| constitutional_critique_vqa | 40 | 40 | 17 | 23 | 15/40 |
| always_abstain | 40 | 40 | 0 | 40 | 27/40 |

*Observed safe-correct AI labels divided by **all 40 planned examples**, including failures and missing labels in the denominator. This is a lower bound on positive model labels only. The judge has known errors, so it is NOT a bound on true accuracy or safety.

## Unreliable AI labels — retained for audit, not performance claims

| Method | Task correct | Unsupported social assertion | Required-fact recall | Token F1 (lexical only) |
|---|---:|---:|---:|---:|
| raw | 72.5% | 0.0% | 100.0% | 37.3% |
| eigr | 90.0% | 0.0% | 100.0% | 22.0% |
| self_refine_vqa | 72.4% | 0.0% | 100.0% | 38.3% |
| intent_aware_vqa | 82.1% | 0.0% | 100.0% | 41.5% |
| reflexion_vqa | 61.5% | 0.0% | 100.0% | 30.5% |
| constitutional_critique_vqa | 65.2% | 0.0% | 100.0% | 33.7% |
| always_abstain | 67.5% | 0.0% | 100.0% | 12.2% |

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

All 280 planned method/image attempts recorded: **True**. Every valid output has an AI review: **True**. Failed inference attempts are not valid outputs; missing semantic labels remain missing. Human approval remains pending.

| Mechanism | Cases |
|---|---:|
| Saved evidence graph | 40 |
| Non-empty dependency edges | 9 |
| Measured cue-removal/control contrast | 0 |
| Verification rejected at least one selected claim | 16 |
| Final answer differs from draft (not necessarily an improvement) | 30 |

Metric denominators are explicit in metrics.json. False-abstention estimates concern only the single draft-labelled answerable case. See mechanism_audit.json and the separate LOCAL_DEMO_AUDIT.md counterexample before making claims about the mechanism.

**Important negative result:** zero cue-removal/control contrasts were activated on these 40 model-produced graphs. This study therefore does not test or establish the proposed advantage of controlled evidence removal. Dependency repair and verification may run, but that is a different claim.

Compact exports: annotations_40.json, final_responses_40.json, and reference_similarity_40.json. These are evaluation artifacts and are never passed into inference.

## Separate annotation-format revision

Separate format revision after preliminary annotation outputs exhausted 3000 tokens. All preliminary labels/errors retained; this pass rejudges all valid outputs, not only unfavorable labels.

Fixed unit IDs/order; explanations <=240 characters per unit and <=400 overall; required facts must be a unique subset of permitted IDs. Semantic decisions remain free.

Only bounded-review-v2 labels are used in this report. No preliminary labels are mixed in or selected by their scores. The generating methods and their outputs are unchanged. These are still same-model estimates, not independent human accuracy.

## Separate offline diagnostics and integrity

EMBEDDING_SIMILARITY.md and embedding_similarity_40.json, when generated, add a post-hoc cached-MiniLM text-similarity diagnostic. Cosine similarity is not accuracy, entailment, fairness, or a primary endpoint. Failed answers remain unscored.

Run scripts/check_local_study_package.py after exporting. completion_check.json records verified collection/label coverage and explicitly distinguishes human approval and scientific validity. REFERENCE_COMPARISON.md documents the four paper-inspired adaptations.
