# Reference-first evaluation — Version 2

Status: incomplete_no_performance_conclusion

Original 40 are exposed DEVELOPMENT data, not held-out test evidence. Similarity is not fairness/accuracy.

Annotations: {'pending_human_review': 40}

Human labels and AI-estimated labels are kept separate. Pending records are not completed annotations.

| Method | Planned | Reviewed | Missing outputs | Failed outputs |
|---|---:|---:|---:|---:|
| raw | 40 | 0 | 37 | 0 |
| eigr | 40 | 0 | 37 | 3 |
| self_refine_vqa | 40 | 0 | 37 | 2 |
| intent_aware_vqa | 40 | 0 | 37 | 3 |
| reflexion_vqa | 40 | 0 | 37 | 3 |
| constitutional_critique_vqa | 40 | 0 | 37 | 3 |
| always_abstain | 40 | 0 | 37 | 0 |

No published-score match or superiority is established. See metrics.json for explicitly labelled descriptive diagnostics.

Hypothesis: evidence interventions and dependency repair improve safe-correct answers while retaining required facts. This is not proven novelty.
