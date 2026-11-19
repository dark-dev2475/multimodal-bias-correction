# What the four comparisons mean

These are **paper-inspired methods rerun on our task**, not reproductions of
the papers' numerical results. No published-score match or superiority has
been established. The local judge has documented errors; read
JUDGE_RELIABILITY_AUDIT.md before interpreting any score.

| Reference | Central idea in the paper | Implementation in this study | Important deviation |
|---|---|---|---|
| [Self-Refine, 2023](https://arxiv.org/abs/2303.17651) | A model iteratively critiques and revises its own output without additional training. | Shared draft → feedback → revision, with accumulated attempt history. | Original zero-shot multimodal prompts, local 2B model, different task/data; at most four repair rounds. |
| [Intent-Aware Self-Correction, 2025](https://arxiv.org/abs/2503.06011) | Clarifies debiasing intent in instructions, responses and multi-aspect feedback. | Method-specific debiasing draft, brief evidence explanation, coherence/comprehensiveness/objectivity scores, up to three repairs. | Open-ended VQA, no copied few-shot examples or private reasoning, different model/data. |
| [Reflexion, 2023](https://arxiv.org/abs/2303.11366) | Reflective language feedback and episodic memory guide later attempts without weight updates. | Evaluate → brief reflection → revised answer; memory persists within one image/task only. | No external task environment or cross-task learning; not the paper's coding or agent benchmark. |
| [Constitutional AI, 2022](https://arxiv.org/abs/2212.08073) | Principles guide critique/revision and subsequent supervised and AI-feedback reinforcement training. | Only principle-guided critique and revision, with four task-specific grounding/fairness principles. | **No training stage.** This is a constitutional-critique control, not full Constitutional AI. |

Source descriptions were checked against the authors' arXiv abstracts on
2026-11-04; the implementation details above come from this repository's code.
The self-correction survey is background, not a runnable algorithm. Self-Debias
is outside this four-control experiment.

## What is held constant

The same 40 original, previously exposed development images/questions, frozen
reference drafts, local Qwen3-VL-2B model, deterministic settings, and a maximum
nine-call budget per method/image are used. Intent-Aware generates its own
draft within its eight-call method budget; the other model-based methods reuse
the raw draft plus at most eight further calls. An upper bound is not identical
actual compute; traces retain all calls, failures and cache provenance.

Raw generation and always-abstain are additional controls. The references label
39 questions ambiguous and one answerable, so an abstention-heavy method can
look good without preserving much useful information. Neither this sample nor
the unreliable judge establishes generalization or useful-answer preservation.

## Proposed EIGR contribution versus demonstrated result

EIGR implements an explicit evidence/claim graph, cue removal with matched
ledger controls, dependency-constrained subset repair, and model-based visual
verification. This combination is a **research hypothesis**, not established
novelty. Being training-free alone is not new: Self-Refine and Reflexion already
avoid weight updates.

In this local ordered-DAG follow-up, 40 EIGR outputs were produced, dependency
edges appeared in nine cases, and **zero measured cue-removal contrasts ran**.
The experiment therefore does not establish a controlled-removal advantage.
The restricted graph decoder was added after development failures; it cannot
represent dependencies on later draft units. All earlier failures are preserved.

For a defensible effectiveness claim, the next study needs independently
reviewed references/labels, a new held-out and answerability-balanced sample,
and mechanism ablations that actually activate the proposed interventions.
Those experiments have not been completed here.
