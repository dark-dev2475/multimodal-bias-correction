from pydantic import BaseModel, field_validator
from typing import List, Literal
from utils.json_parse import parse_model_json
from models.vlm import OpenRouterVLM


VerificationStatus = Literal["SUPPORTED", "UNCERTAIN", "UNSUPPORTED"]


class VerificationResult(BaseModel):
    claim: str
    evidence_status: VerificationStatus
    problematic: bool
    explanation: str

    @field_validator("evidence_status", mode="before")
    @classmethod
    def _normalise_status(cls, value):
        if isinstance(value, str):
            return value.strip().upper()
        return value


class VerificationList(BaseModel):
    claims: List[VerificationResult]


class BiasVerifier:

    def __init__(self):
        self.vlm = OpenRouterVLM()

    def verify(self, image_path, corrected_response):

        prompt = f"""
You are the final verification component of a
multimodal bias-correction system.

Your task is to verify whether the corrected response
contains claims that are unsupported by the image.

CORRECTED RESPONSE:

{corrected_response}

Break the response into atomic claims.

For each claim determine:

1. Whether the claim is visually supported.
2. Whether the claim is problematic because it makes
   an unsupported personal, medical, demographic,
   occupational, socioeconomic, emotional, identity,
   or relationship assumption.

Evidence status must be exactly one of:

- SUPPORTED
- UNCERTAIN
- UNSUPPORTED

IMPORTANT:

- Use only visual evidence from the image.
- Do not infer medical conditions from wheelchair use.
- Do not infer occupation or professional status from
  clothing, location, or activity alone.
- Do not infer socioeconomic status from appearance
  or possessions.
- Do not infer emotions, relationships, dependence,
  or intentions without sufficient evidence.
- A directly observable physical fact should normally
  be marked SUPPORTED.
- If a claim cannot be established from the image,
  mark it UNSUPPORTED or UNCERTAIN.

A claim is problematic if it contains an unsupported
sensitive personal assumption.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "claims": [
        {{
            "claim": "...",
            "evidence_status": "SUPPORTED",
            "problematic": false,
            "explanation": "..."
        }}
    ]
}}
"""

        content = self.vlm.complete(
            prompt,
            image_path=image_path,
            source="Verifier"
        )

        data = parse_model_json(content, "Verifier")

        return VerificationList.model_validate(data)