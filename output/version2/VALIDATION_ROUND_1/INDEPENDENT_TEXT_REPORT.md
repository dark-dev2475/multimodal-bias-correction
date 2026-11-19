# Independent reference-text diagnostic

**REAL-DATA RELIABILITY CHECK FAILED:** a generic abstention was wrongly judged to entail an office-sharing fact. These are exploratory labels only. See checker_reliability_audit.json and VALIDATION_SUMMARY.md.

Independent-model TEXT entailment against AI-drafted reference facts. Not image-grounded accuracy, fairness, or a replacement for human labels. Non-entailment can reflect incomplete references or justified uncertainty.

The NLI model passed the frozen binary support gate on 24 synthetic text cases. This does not validate it on images or real social-bias decisions. The original model labels and scores are unchanged. This report is a separate post-hoc diagnostic.

| Method | Valid outputs / 40 | Reference-entailing units | Answer-entailing required facts |
|---|---:|---:|---:|
| raw | 40/40 | 0.117 (n=40) | 0.107 (n=38) |
| eigr | 40/40 | 0.146 (n=40) | 0.107 (n=38) |
| self_refine_vqa | 29/40 | 0.132 (n=29) | 0.164 (n=27) |
| intent_aware_vqa | 39/40 | 0.087 (n=39) | 0.112 (n=37) |
| reflexion_vqa | 26/40 | 0.167 (n=26) | 0.135 (n=26) |
| constitutional_critique_vqa | 23/40 | 0.109 (n=23) | 0.091 (n=22) |
| always_abstain | 40/40 | 0.000 (n=40) | 0.026 (n=38) |

These are fractions of text-entailment judgments, NOT task accuracy or bias rates. An uncertainty answer can legitimately be non-entailed by a short fact list. Required facts may be over-inclusive in the unapproved reference drafts. Missing or overlength pairs are unscored, never silently truncated. Different denominators must be considered.

Useful-fact coverage is directional: the answer must express the fact; merely placing a fact in the reference is not evidence that the answer contains it. The inverse direction checks support for an answer sentence. Neither direction proves visual truth.
