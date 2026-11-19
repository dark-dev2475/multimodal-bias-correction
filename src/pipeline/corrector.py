from models.vlm import OpenRouterVLM
from pipeline.correction_policy import (
    CORRECTION_POLICY_VERSION,
    policy_text,
)


class SelfCorrector:

    PROMPT_VERSION = "self_corrector_v2"

    def __init__(self, use_structured_feedback=True, use_correction_policy=True):
        self.vlm = OpenRouterVLM()
        self.use_structured_feedback = use_structured_feedback
        self.use_correction_policy = use_correction_policy

    def correct(
        self,
        image_path,
        original_response,
        flagged_claims,
        supported_claims=None,
    ):
        supported_claims = supported_claims or []

        if self.use_structured_feedback:
            claims_text = "\n\n".join(
                [
                    "FLAGGED CLAIM:\n"
                    f"Claim: {item['claim']}\n"
                    f"Claim type: {item.get('claim_type', 'unknown')}\n"
                    f"Bias category: {item['bias_category']}\n"
                    f"Evidence status: {item['evidence_status']}\n"
                    f"Evidence confidence: {item['evidence_confidence']}\n"
                    f"Reason: {item['evidence_explanation']}\n"
                    f"Bias explanation: {item.get('bias_explanation', '')}\n"
                    f"Severity: {item['severity']}"
                    for item in flagged_claims
                ]
            )
        else:
            claims_text = "\n".join(
                f"- {item['claim']} [{item['bias_category']}]"
                for item in flagged_claims
            )

        supported_text = "\n".join(
            f"- {item['claim']}"
            for item in supported_claims
        ) or "None recorded."
        policy_section = (
            f"CORRECTION POLICY ({CORRECTION_POLICY_VERSION}):\n"
            f"{policy_text()}"
            if self.use_correction_policy
            else "Use the generic correction instructions below."
        )

        prompt = f"""
You are the self-correction component of a
multimodal AI bias-correction system.

Your task is to revise an AI-generated response using
visual evidence from the provided image.

ORIGINAL RESPONSE:

{original_response}

CLAIMS FLAGGED FOR CORRECTION:

{claims_text}

SUPPORTED CLAIMS TO PRESERVE:

{supported_text}

{policy_section}

Instructions:

1. Remove or rewrite claims that are unsupported by
   the visual evidence.

2. Preserve the supported claims listed above whenever they remain natural.

3. Do not introduce new facts.

4. Do not assume medical conditions, disability status,
   occupation, socioeconomic status, relationships,
   emotions, identity, or other personal attributes
   unless they are directly supported by the image.

5. Do not mention the bias-detection process in the
   corrected response.

6. Do not say that the original response was biased.

7. Produce a natural, factual response describing only
   what can reasonably be established from the image.

8. Do not make the response unnecessarily vague.
   Preserve useful visual details.

Return ONLY the corrected response.
"""


        result = self.vlm.complete(
            prompt,
            image_path=image_path,
            source="Self-corrector"
        )

        return result.strip()