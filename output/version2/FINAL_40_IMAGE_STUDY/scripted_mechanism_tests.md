# Version 2 mechanism demonstration

These fixtures test software behavior; they do not measure bias reduction on real images.

Seven scripted cases × five configurations. No model or network is used.

| Case | Configuration | Status | Kept IDs | Residual problematic fixture claims |
|---|---|---|---|---|
| cue_and_propagation | full | verified_subset | c3, c4 | 0 |
| cue_and_propagation | no_interventions | verified_subset | c1, c2, c3, c4 | 2 |
| cue_and_propagation | no_dependency_closure | verified_subset | c3, c4 | 0 |
| cue_and_propagation | no_matched_control | verified_subset | c3, c4 | 0 |
| cue_and_propagation | no_visual_verification | unverified_ablation | c3, c4 | 0 |
| inherited_premise | full | verified_subset | c3, c4 | 0 |
| inherited_premise | no_interventions | verified_subset | c1, c2, c3, c4 | 2 |
| inherited_premise | no_dependency_closure | verified_subset | c2, c3, c4 | 1 |
| inherited_premise | no_matched_control | verified_subset | c3, c4 | 0 |
| inherited_premise | no_visual_verification | unverified_ablation | c3, c4 | 0 |
| placebo_sensitive_control | full | verified_subset | c1, c3 | 0 |
| placebo_sensitive_control | no_interventions | verified_subset | c1, c3 | 0 |
| placebo_sensitive_control | no_dependency_closure | verified_subset | c1, c3 | 0 |
| placebo_sensitive_control | no_matched_control | verified_subset | c3 | 0 |
| placebo_sensitive_control | no_visual_verification | unverified_ablation | c1, c3 | 0 |
| explicit_context_control | full | verified_subset | c1, c3, c4 | 0 |
| explicit_context_control | no_interventions | verified_subset | c1, c3, c4 | 0 |
| explicit_context_control | no_dependency_closure | verified_subset | c1, c3, c4 | 0 |
| explicit_context_control | no_matched_control | verified_subset | c1, c3, c4 | 0 |
| explicit_context_control | no_visual_verification | unverified_ablation | c1, c3, c4 | 0 |
| visual_verifier_veto | full | verified_subset | c3 | 0 |
| visual_verifier_veto | no_interventions | verified_subset | c3 | 0 |
| visual_verifier_veto | no_dependency_closure | verified_subset | c3 | 0 |
| visual_verifier_veto | no_matched_control | verified_subset | c3 | 0 |
| visual_verifier_veto | no_visual_verification | unverified_ablation | c1, c3 | 1 |
| unsupported_abstention | full | abstained | none | 0 |
| unsupported_abstention | no_interventions | abstained | none | 0 |
| unsupported_abstention | no_dependency_closure | abstained | none | 0 |
| unsupported_abstention | no_matched_control | abstained | none | 0 |
| unsupported_abstention | no_visual_verification | abstained | none | 0 |
| conflict_resolution | full | verified_subset | c1 | 0 |
| conflict_resolution | no_interventions | verified_subset | c1 | 0 |
| conflict_resolution | no_dependency_closure | verified_subset | c1 | 0 |
| conflict_resolution | no_matched_control | verified_subset | c1 | 0 |
| conflict_resolution | no_visual_verification | unverified_ablation | c1 | 0 |
