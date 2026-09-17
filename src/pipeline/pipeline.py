import json

from pipeline.generator import InitialGenerator
from pipeline.claim_extractor import ClaimExtractor
from pipeline.evidence_checker import EvidenceChecker
from pipeline.bias_monitor import BiasMonitor
from pipeline.corrector import SelfCorrector
from pipeline.verifier import BiasVerifier


class BiasCorrectionPipeline:

    def __init__(self):

        self.generator = InitialGenerator()
        self.claim_extractor = ClaimExtractor()
        self.evidence_checker = EvidenceChecker()
        self.bias_monitor = BiasMonitor()
        self.corrector = SelfCorrector()
        self.verifier = BiasVerifier()

    # ---------------------------------------------------------
    # Helper: parse JSON returned by the VLM
    # ---------------------------------------------------------

    def _parse_json(self, content):

        if not content:
            raise ValueError("Model returned an empty response")

        content = content.strip()

        # Remove markdown code fences if the model adds them
        if content.startswith("```"):
            content = content.replace("```json", "", 1)
            content = content.replace("```", "", 1)
            content = content.strip()

        try:
            return json.loads(content)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Could not parse model JSON:\n{content}"
            ) from exc

    # ---------------------------------------------------------
    # STEP 1: Initial Generation
    # ---------------------------------------------------------

    def generate_initial_response(self, image_path, question):

        response = self.generator.generate(
            image_path=image_path,
            question=question
        )

        return response

    # ---------------------------------------------------------
    # STEP 2: Claim Extraction
    # ---------------------------------------------------------

    def extract_claims(self, response):

        result = self.claim_extractor.extract(response)

        return result

    # ---------------------------------------------------------
    # STEP 3 + 4: Evidence Checking + Bias Monitoring
    # ---------------------------------------------------------
    def analyze_claims(self, image_path, claims):
        print("\n[3/6] Checking visual evidence...")

        if not claims:
            return [], []

        evidence_results = self.evidence_checker.check_all(image_path, claims)

        print("\n[4/6] Monitoring for bias-relevant assumptions...")
        bias_results = self.bias_monitor.analyze_all(evidence_results.results)

        bias_by_id = {
            result.claim_id: result
            for result in bias_results.results
        }
        claim_analysis = []

        for evidence in evidence_results.results:
            bias = bias_by_id.get(evidence.claim_id)
            analysis = {
                "claim_id": evidence.claim_id,
                "claim": evidence.claim,
                "evidence_status": evidence.evidence_status,
                "evidence_confidence": evidence.confidence,
                "evidence_explanation": evidence.explanation,
                "bias_detected": False,
                "bias_category": "none",
                "severity": "none",
                "requires_correction": False,
                "bias_explanation": ""
            }

            if bias is not None:
                analysis.update({
                    "bias_detected": bias.bias_detected,
                    "bias_category": bias.bias_category,
                    "severity": bias.severity,
                    "requires_correction": bias.requires_correction,
                    "bias_explanation": bias.explanation
                })

            claim_analysis.append(analysis)

        flagged_claims = [
            item for item in claim_analysis
            if item["requires_correction"]
        ]

        print(f"\nTotal claims: {len(claims)}")
        print(f"Flagged claims: {len(flagged_claims)}")
        return claim_analysis, flagged_claims

    # ---------------------------------------------------------
    # STEP 5: Self Correction
    # ---------------------------------------------------------

    def correct_response(
        self,
        image_path,
        original_response,
        flagged_claims
    ):

        # Nothing to correct
        if not flagged_claims:

            return original_response

        corrected_response = self.corrector.correct(
            image_path=image_path,
            original_response=original_response,
            flagged_claims=flagged_claims
        )

        return corrected_response

    # ---------------------------------------------------------
    # STEP 6: Verification
    # ---------------------------------------------------------

    def verify_response(
        self,
        image_path,
        corrected_response
    ):

        verification = self.verifier.verify(
            image_path=image_path,
            corrected_response=corrected_response
        )

        return verification

    # ---------------------------------------------------------
    # COMPLETE PIPELINE
    # ---------------------------------------------------------

    def run(self, image_path, question, max_iterations=2):

    print("\n" + "=" * 70)
    print("MULTIMODAL BIAS-CORRECTION PIPELINE")
    print("=" * 70)

    # ---------------------------------------------------------
    # ITERATION 0: INITIAL GENERATION
    # ---------------------------------------------------------

    print("\n[1/6] INITIAL VLM GENERATION")
    print("-" * 70)

    initial_response = self.generate_initial_response(
        image_path,
        question
    )

    print(initial_response)

    # ---------------------------------------------------------
    # CLAIM EXTRACTION
    # ---------------------------------------------------------

    print("\n[2/6] CLAIM EXTRACTION")
    print("-" * 70)

    claims = self.extract_claims(initial_response)

    if not claims:
        raise RuntimeError(
            "Claim extraction returned zero claims. "
            "Pipeline stopped."
        )

    print(f"Extracted {len(claims)} claims.")

    for i, claim in enumerate(claims, start=1):
        print(
            f"{i}. {claim.claim} "
            f"[{claim.type} | {claim.category}]"
        )

    # ---------------------------------------------------------
    # EVIDENCE + BIAS ANALYSIS
    # ---------------------------------------------------------

    claim_analysis, flagged_claims = self.analyze_claims(
        image_path,
        claims
    )

    print("\nCLAIM ANALYSIS")
    print("-" * 70)

    for item in claim_analysis:
        print(f"\nClaim: {item['claim']}")
        print(f"Evidence: {item['evidence_status']}")
        print(f"Bias detected: {item['bias_detected']}")
        print(f"Bias category: {item['bias_category']}")
        print(
            f"Requires correction: "
            f"{item['requires_correction']}"
        )

    # ---------------------------------------------------------
    # ITERATIVE CORRECTION + VERIFICATION
    # ---------------------------------------------------------

    current_response = initial_response
    current_flagged_claims = flagged_claims

    correction_history = []

    for iteration in range(1, max_iterations + 1):

        print("\n" + "=" * 70)
        print(f"CORRECTION ITERATION {iteration}/{max_iterations}")
        print("=" * 70)

        print("\n[5/6] SELF-CORRECTION")
        print("-" * 70)

        print(f"Flagged claims: {len(current_flagged_claims)}")

        # Nothing to correct
        if not current_flagged_claims:

            print("\nNo corrections required.")

            corrected_response = current_response

        else:

            corrected_response = self.correct_response(
                image_path,
                current_response,
                current_flagged_claims
            )

        print("\nCORRECTED RESPONSE")
        print("-" * 70)
        print(corrected_response)

        # -----------------------------------------------------
        # VERIFICATION
        # -----------------------------------------------------

        print("\n[6/6] FINAL VERIFICATION")
        print("-" * 70)

        verification = self.verify_response(
            image_path,
            corrected_response
        )

        problematic_claims = [
            claim
            for claim in verification.claims
            if claim.problematic
        ]

        verification_passed = len(problematic_claims) == 0

        correction_history.append({
            "iteration": iteration,
            "input_response": current_response,
            "corrected_response": corrected_response,
            "problematic_claims": [
                {
                    "claim": claim.claim,
                    "evidence_status": claim.evidence_status,
                    "explanation": claim.explanation
                }
                for claim in problematic_claims
            ],
            "verification_passed": verification_passed
        })

        if verification_passed:

            print("\nVerification passed.")
            print(
                "No problematic unsupported claims "
                "were detected."
            )

            return {
                "question": question,
                "initial_response": initial_response,
                "claims": [
                    claim.model_dump()
                    for claim in claims
                ],
                "claim_analysis": claim_analysis,
                "flagged_claims": current_flagged_claims,
                "corrected_response": corrected_response,
                "verification": verification.model_dump(),
                "verification_passed": True,
                "correction_iterations": iteration,
                "correction_history": correction_history
            }

        # -----------------------------------------------------
        # VERIFICATION FAILED
        # -----------------------------------------------------

        print(
            "\nWARNING: Potentially problematic claims "
            "remain after correction."
        )

        for claim in problematic_claims:

            print(f"\n- {claim.claim}")
            print(f"  Evidence: {claim.evidence_status}")
            print(f"  Explanation: {claim.explanation}")

        # If this was the final allowed iteration,
        # stop instead of looping forever.
        if iteration == max_iterations:

            return {
                "question": question,
                "initial_response": initial_response,
                "claims": [
                    claim.model_dump()
                    for claim in claims
                ],
                "claim_analysis": claim_analysis,
                "flagged_claims": current_flagged_claims,
                "corrected_response": corrected_response,
                "verification": verification.model_dump(),
                "verification_passed": False,
                "correction_iterations": iteration,
                "correction_history": correction_history
            }

        # -----------------------------------------------------
        # PREPARE NEXT ITERATION
        # -----------------------------------------------------

        current_response = corrected_response

        # Convert verification failures into correction targets
        current_flagged_claims = [
            {
                "claim": claim.claim,
                "bias_category": "verification_failure"
            }
            for claim in problematic_claims
        ]

    raise RuntimeError("Unexpected pipeline termination.")