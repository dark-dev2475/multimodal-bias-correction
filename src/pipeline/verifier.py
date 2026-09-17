import json
from pydantic import BaseModel
from typing import List
from models.vlm import OpenRouterVLM


class VerificationResult(BaseModel):
    claim: str
    evidence_status: str
    problematic: bool
    explanation: str


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

        result = self.vlm.client.chat.completions.create(
            model=self.vlm.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self.vlm._encode_image(image_path)
                            }
                        }
                    ]
                }
            ]
        )

        content = result.choices[0].message.content

        if not content:
            raise ValueError(
                "Verifier returned an empty response"
            )

        content = content.strip()

        if content.startswith("```"):
            content = content.replace("```json", "", 1)
            content = content.replace("```", "", 1)
            content = content.strip()

        try:
            data = json.loads(content)
            return VerificationList.model_validate(data)

        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"Verifier returned invalid JSON: {content}"
            ) from exc