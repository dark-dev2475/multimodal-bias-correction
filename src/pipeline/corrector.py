from models.vlm import OpenRouterVLM


class SelfCorrector:

    def __init__(self):
        self.vlm = OpenRouterVLM()

    def correct(self, image_path, original_response, flagged_claims):

        claims_text = "\n".join(
            [
                f"- {item['claim']} "
                f"[{item['bias_category']}]"
                for item in flagged_claims
            ]
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

Instructions:

1. Remove or rewrite claims that are unsupported by
   the visual evidence.

2. Preserve claims that are supported by the image.

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