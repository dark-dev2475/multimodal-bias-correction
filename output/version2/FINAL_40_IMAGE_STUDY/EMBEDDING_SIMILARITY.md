# Offline embedding similarity — exploratory diagnostic

Exploratory post-hoc text similarity, not accuracy/fairness/entailment. Negated or biased sentences can be highly similar. References are AI-assisted drafts pending human review; the 40 cases are development data.

No training or network requests. A cached MiniLM encoder compares final answer text to reference text; it does not inspect the image. This analysis was added after collection began and is not a preregistered endpoint.

| Method | Scored / planned | Mean cosine (scored only) |
|---|---:|---:|
| raw | 40/40 | 0.6892 |
| eigr | 40/40 | 0.3819 |
| self_refine_vqa | 29/40 | 0.6935 |
| intent_aware_vqa | 39/40 | 0.7371 |
| reflexion_vqa | 26/40 | 0.6277 |
| constitutional_critique_vqa | 23/40 | 0.6246 |
| always_abstain | 40/40 | 0.1764 |

Cosine is not a percentage of correct responses. Failed/missing outputs are unscored, never assigned a successful score. Compare coverage before interpreting averages. Long text may be truncated at the encoder limit; per-response flags record this.

Model: [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Model-file and input hashes are saved in the JSON exports.
