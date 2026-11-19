# EIGR V2: preliminary local experiment

**Execution results only — not accuracy, bias-reduction scores, or a match to published paper results.**

Forty previously exposed development images were evaluated using the same free local Qwen3-VL-2B model. The reference-inspired methods are our local VQA adaptations, not replications of the papers. Actual model-call counts differ. All failed attempts are retained.

## Recorded output coverage

An output counted here passed the experiment's output requirements; it is not necessarily correct or unbiased. Abstentions count as outputs.

| Local method | Attempts | Outputs returned without recorded failure | Recorded failures | Output coverage (not accuracy) |
| --- | ---: | ---: | ---: | ---: |
| Raw model | 40 | 40 | 0 | 100% |
| EIGR V2 | 40 | 40 | 0 | 100% |
| Self-Refine-inspired VQA [1] | 40 | 29 | 11 | 72.5% |
| Intent-Aware-inspired VQA [2] | 40 | 39 | 1 | 97.5% |
| Reflexion-inspired VQA [3] | 40 | 26 | 14 | 65% |
| Constitutional critique-inspired VQA (no training) [4] | 40 | 23 | 17 | 57.5% |
| Always-abstain control | 40 | 40 | 0 | 100% |
| Total | 280 | 237 | 43 | 84.6% |

EIGR returned 15 retained-claim subsets and 25 abstentions. Its internal `verified_subset` status is not independent evidence of correctness. The raw and always-abstain controls also achieved 100% output coverage, illustrating why coverage cannot establish superior answer quality. Baseline failures concern these implementations and settings, not the published methods in general.

## Architecture observations

| Observation | Recorded count | Interpretation |
| --- | ---: | --- |
| Saved EIGR evidence graphs | 40/40 | Graph records were produced |
| Graphs with dependency edges | 9/40 | Dependency representation was used in some cases |
| Answers changed from initial draft | 30/40 | Revision occurred; improvement is not established |
| Verification calls rejecting at least one claim | 16 | Internal filtering occurred; rejection correctness is unvalidated |
| Cases with measured evidence-removal contrasts | 0/40 | The proposed intervention mechanism did not activate |
| Human-approved reference annotations | 0/40 | All reference drafts still need review |

## Conclusion suitable for presentation

> We implemented a free, training-free multimodal correction prototype and completed 280 method-image attempts on 40 development images. EIGR produced an output on all 40 cases, including 25 abstentions. These are engineering results, not validated evidence that EIGR matches or outperforms published methods. Evidence-removal activation and reliable semantic evaluation remain unresolved.

We intentionally do not report the saved automated judge's apparent accuracy percentages as performance results: spot checks found incorrect support, bias, and fact-coverage judgments. Published-score replication and superiority are both **not established**.

## Paper references and scope

These papers inspired the four named baseline adaptations. **Every numerical result in the table comes from our local experiment, not from these publications.** Raw model and always-abstain are controls; EIGR V2 is our prototype.

[1] A. Madaan et al., "Self-Refine: Iterative Refinement with Self-Feedback," 2023, arXiv:2303.17651. [Paper](https://arxiv.org/abs/2303.17651). Uses iterative model-generated feedback and revision without additional training.

[2] P. Anantaprayoon, M. Kaneko, and N. Okazaki, "Intent-Aware Self-Correction for Mitigating Social Biases in Large Language Models," 2025, arXiv:2503.06011. [Paper](https://arxiv.org/abs/2503.06011). Uses explicit debiasing intent and multi-aspect feedback.

[3] N. Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning," 2023, arXiv:2303.11366. [Paper](https://arxiv.org/abs/2303.11366). Uses verbal feedback and episodic reflection memory without weight updates.

[4] Y. Bai et al., "Constitutional AI: Harmlessness from AI Feedback," 2022, arXiv:2212.08073. [Paper](https://arxiv.org/abs/2212.08073). The full method includes supervised and reinforcement-learning training. Our row uses only a principle-guided critique/revision adaptation and does not reproduce Constitutional AI training.

## Sources in this workspace

- [Recorded method counts and mechanism audit](FINAL_40_IMAGE_STUDY/metrics.json)
- [Completion and validation flags](FINAL_40_IMAGE_STUDY/completion_check.json)
- [Original judge reliability audit](FINAL_40_IMAGE_STUDY/JUDGE_RELIABILITY_AUDIT.md)
- [Subsequent validation and real-data checker failure](VALIDATION_ROUND_1/VALIDATION_SUMMARY.md)

Prepared from existing artifacts on 2026-11-05. No inference rerun, fabricated measurements, or original-result edits were made for this table.
