# Why interventions did not activate

Audited all 40 saved EIGR graphs without changing them.

| Immediate prerequisite failure | Cases |
|---|---:|
| no_tagged_cues | 39 |
| no_cue_linked_social_inference | 1 |

The graphs contain 696 observations, including 578 repeated text/source entries. 21 graphs reach the 32-observation cap. 18 contain observations exactly matching a draft claim.

These are measured structural facts. Overlap can also occur for genuinely visible claims, so it is a review flag, not automatic proof of hallucination. The root cause inside the model has not been causally isolated.

The planner requires a tagged cue, a social-inference claim, and an evidence path linking them. None of the 40 graphs meets all three prerequisites. Merely increasing the call budget cannot fix these upstream labels/links. No cue removal was artificially forced.

Next experimental requirement: independently check observation truth, cue recall, social-claim recall and evidence links before a new end-to-end comparison. Text calibration alone does not validate visual extraction. Human reference forms cover all 40 images.
