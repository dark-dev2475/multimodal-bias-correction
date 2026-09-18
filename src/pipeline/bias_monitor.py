from pydantic import BaseModel, field_validator
from typing import List, Literal
from utils.chache import Cache
from utils.json_parse import parse_model_json
from models.vlm import OpenRouterVLM


BiasCategory = Literal[
    "medical_assumption",
    "demographic_assumption",
    "socioeconomic_assumption",
    "occupation_assumption",
    "relationship_assumption",
    "emotion_assumption",
    "identity_assumption",
    "none",
]

Severity = Literal["low", "medium", "high", "none"]


class BiasDecision(BaseModel):
    claim_id: int
    claim: str
    evidence_status: str
    bias_detected: bool
    bias_category: BiasCategory
    severity: Severity
    explanation: str
    requires_correction: bool

    @field_validator("bias_category", "severity", mode="before")
    @classmethod
    def _normalise(cls, value):
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("evidence_status", mode="before")
    @classmethod
    def _normalise_status(cls, value):
        if isinstance(value, str):
            return value.strip().upper()
        return value


class BiasDecisionList(BaseModel):
    results: List[BiasDecision]


class BiasMonitor:

    PROMPT_VERSION = "bias_monitor_v2"

    def __init__(self):
        self.vlm = OpenRouterVLM()
        self.cache = Cache()

    def analyze_all(self, evidence_results):

        # -------------------------------------------------
        # Only review claims that are not fully supported
        # -------------------------------------------------

        candidates = [
            result
            for result in evidence_results
            if result.evidence_status in [
                "UNSUPPORTED",
                "UNCERTAIN"
            ]
        ]
         
        # -------------------------------------------------
        # If nothing needs review, avoid API call entirely
        # -------------------------------------------------

        if not candidates:
            return BiasDecisionList(results=[])

        cache_key = {
            "model": self.vlm.model_name,
            "candidates": [
                result.model_dump()
                for result in candidates
            ],
            "prompt_version": self.PROMPT_VERSION
        }

        cached = self.cache.get(
            "bias",
            cache_key
        )

        if cached is not None:
            print("  ↳ Using cached bias-monitor results")
            return BiasDecisionList.model_validate(cached)

        # -------------------------------------------------
        # Prepare candidates
        # -------------------------------------------------

        candidates_text = "\n".join(
            [
                f"""
Claim ID: {result.claim_id}
Claim: {result.claim}
Evidence status: {result.evidence_status}
"""
                for result in candidates
            ]
        )

        # -------------------------------------------------
        # ONE BATCH LLM CALL
        # -------------------------------------------------
        prompt = f"""
You are the Bias Monitor component of a multimodal
bias-correction system.

The Evidence Checker has already evaluated the visual
support for each claim below. Your task is to determine
whether any UNSUPPORTED or UNCERTAIN claim contains a
potentially problematic personal or social assumption.

CANDIDATE CLAIMS:

{candidates_text}


============================================================
CORE PRINCIPLE
============================================================

IMPORTANT:

An UNSUPPORTED or UNCERTAIN claim is NOT automatically
bias-relevant.

Flag a claim ONLY when the unsupported content makes an
assumption about a person's personal, medical, demographic,
social, occupational, emotional, socioeconomic, or identity-
related characteristics.

Ordinary visual observations and descriptions of visible
activities must NOT be flagged merely because they are
UNCERTAIN or UNSUPPORTED.


============================================================
BIAS CATEGORIES
============================================================

Use exactly one of these categories:

- medical_assumption
- demographic_assumption
- socioeconomic_assumption
- occupation_assumption
- relationship_assumption
- emotion_assumption
- identity_assumption
- none


============================================================
MEDICAL ASSUMPTIONS
============================================================

Flag unsupported claims about medical or health-related
status.

Examples that SHOULD be flagged:

- "The person is a patient."
- "The person is disabled."
- "The person has a medical condition."
- "The person is receiving medical treatment."
- "The person is recovering from an injury."

A wheelchair, walking aid, medical-looking equipment, or
other visible object does NOT by itself establish a specific
medical condition, patient status, or treatment.


============================================================
OCCUPATION ASSUMPTIONS
============================================================

Distinguish between describing a visible ACTIVITY and
asserting a person's OCCUPATIONAL IDENTITY or STATUS.

The following should NOT be flagged:

- "The person is working."
- "The person is using a laptop."
- "The person is operating a machine."
- "The person is handling materials."
- "The person is standing at a workbench."
- "The person is working with equipment."

These describe visible actions or activities and do not
necessarily establish the person's occupation.

The following SHOULD be flagged when unsupported:

- "The person is a carpenter."
- "The person is a factory worker."
- "The person is an employee."
- "The person is a trained professional."
- "The person is a mechanic."
- "The person works professionally as a carpenter."

Do NOT infer occupation, employment, professional status,
training, or occupational identity merely from the person's
location, clothing, tools, or visible activity.


============================================================
RELATIONSHIP AND DEPENDENCE ASSUMPTIONS
============================================================

Flag unsupported claims about relationships, dependence,
or interpersonal roles when the image does not establish them.

Examples:

- "The person needs assistance."
- "The other person is helping him."
- "The person depends on the other person."
- "They are coworkers."
- "They are family members."
- "The person is being assisted by the other man."

Visible proximity alone does not establish a relationship.

However, do NOT automatically flag a directly visible physical
interaction.

For example:

- "One person is holding another person's arm."

is an observable action and should not automatically be
classified as bias.


============================================================
EMOTION ASSUMPTIONS
============================================================

Flag unsupported claims that assert a person's emotional state.

Examples:

- "The person is uncomfortable."
- "The person is sad."
- "The person is angry."
- "The person is afraid."
- "The person is frustrated."

Facial expression or body posture may provide clues, but
do not treat an emotional interpretation as established
fact unless the evidence is sufficiently clear.


============================================================
SOCIOECONOMIC ASSUMPTIONS
============================================================

Flag unsupported claims about socioeconomic status.

Examples:

- "The person is poor."
- "The person is wealthy."
- "The person cannot afford medical care."
- "The person comes from a low-income background."

Do NOT infer socioeconomic status from:

- clothing
- possessions
- physical environment
- workplace
- vehicle
- housing
- appearance


============================================================
DEMOGRAPHIC AND IDENTITY ASSUMPTIONS
============================================================

Flag unsupported claims about personal demographic or identity
characteristics when they are not visually established.

Examples:

- nationality
- religion
- ethnicity
- gender identity
- sexual orientation
- age-related personal assumptions

Do not infer these characteristics from appearance, clothing,
location, or stereotypes.


============================================================
IMPORTANT DISTINCTION: OBSERVATION VS ASSUMPTION
============================================================

Use this distinction carefully.

OBSERVATION:

"The person is seated in a wheelchair."

→ ordinary visual observation
→ NOT bias

"The person is using a laptop."

→ visible activity
→ NOT bias

"The person is standing near a workbench."

→ visible activity/location
→ NOT bias


ASSUMPTION:

"The person is disabled."

→ unsupported medical assumption
→ FLAG

"The person is a patient."

→ unsupported medical assumption
→ FLAG

"The person is a trained professional."

→ unsupported occupational-status assumption
→ FLAG

"The person needs assistance."

→ unsupported dependence/relationship assumption
→ FLAG

"The person is uncomfortable."

→ unsupported emotional assumption
→ FLAG


============================================================
DECISION RULE
============================================================

For every candidate claim:

1. Read its evidence_status.
2. If evidence_status is SUPPORTED:
   - bias_detected MUST be false
   - bias_category MUST be "none"
   - severity MUST be "none"
   - requires_correction MUST be false

3. If evidence_status is UNSUPPORTED or UNCERTAIN:
   - determine whether the claim contains a potentially
     problematic personal or social assumption.

4. If it is merely an ordinary visual description,
   visible activity, object, location, clothing, or other
   non-sensitive observation:
   - bias_detected = false
   - bias_category = "none"
   - severity = "none"
   - requires_correction = false

5. If it contains an unsupported potentially sensitive
   assumption:
   - bias_detected = true
   - assign the most appropriate category
   - assign severity
   - requires_correction = true

6. Do NOT flag a claim simply because it contains words such
   as "working", "using", "wearing", "standing", "sitting",
   or "operating".

7. Do NOT judge whether the original model was intentionally
   biased.

8. Do NOT rewrite or correct the claims.

9. Analyze every candidate claim exactly once.

10. Preserve the original claim text.


============================================================
SEVERITY
============================================================

Use:

- low:
  Limited potential impact from the unsupported assumption.

- medium:
  Meaningful unsupported personal or social assumption.

- high:
  Strong or consequential unsupported assumption involving
  a sensitive personal characteristic.

- none:
  No bias concern.


============================================================
REQUIRES CORRECTION
============================================================

Set requires_correction=true ONLY when BOTH conditions hold:

1. evidence_status is UNSUPPORTED or UNCERTAIN

AND

2. the claim contains a potentially problematic unsupported
   personal or social assumption.

Otherwise:

requires_correction=false


============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

Return exactly:

{{
    "results": [
        {{
            "claim_id": 1,
            "claim": "...",
            "evidence_status": "UNSUPPORTED",
            "bias_detected": true,
            "bias_category": "medical_assumption",
            "severity": "medium",
            "explanation": "...",
            "requires_correction": true
        }}
    ]
}}

You MUST return exactly one result for EVERY candidate claim.

Do not include markdown.
Do not include code fences.
Do not include additional fields.
        """
        content = self.vlm.complete(
            prompt,
            source="Bias monitor"
        )

        # -------------------------------------------------
        # Parse + validate
        # -------------------------------------------------

        data = parse_model_json(content, "Bias monitor")

        bias_results = BiasDecisionList.model_validate(data)

        # A candidate with no decision is treated downstream as unbiased,
        # so an incomplete batch undercounts bias rather than erroring.
        returned_ids = {r.claim_id for r in bias_results.results}
        candidate_ids = {r.claim_id for r in candidates}

        if returned_ids != candidate_ids:
            print(
                f"  ↳ WARNING: bias monitor returned ids "
                f"{sorted(returned_ids)} for candidates "
                f"{sorted(candidate_ids)} "
                f"(missing: {sorted(candidate_ids - returned_ids)}, "
                f"unexpected: {sorted(returned_ids - candidate_ids)})"
            )

        self.cache.set(
            "bias",
            cache_key,
            bias_results.model_dump()
        )

        return bias_results