from pipeline.generator import InitialGenerator
from pipeline.claim_extractor import ClaimExtractor
from pipeline.evidence_checker import EvidenceChecker
from pipeline.bias_monitor import BiasMonitor
from pipeline.corrector import SelfCorrector
from pipeline.verifier import BiasVerifier


class BiasCorrectionPipeline:

    def __init__(self, generation_mode="real"):

        self.generator = InitialGenerator(mode=generation_mode)
        self.claim_extractor = ClaimExtractor()
        self.evidence_checker = EvidenceChecker()
        self.bias_monitor = BiasMonitor()
        self.corrector = SelfCorrector()
        self.verifier = BiasVerifier()

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

        return result.claims

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

            # Only claims the Bias Monitor was asked to review can be
            # missing a decision. A SUPPORTED claim legitimately has none.
            was_reviewed = evidence.evidence_status in (
                "UNSUPPORTED",
                "UNCERTAIN"
            )

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
                "bias_explanation": "",
                "bias_analysis_missing": was_reviewed and bias is None
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

        missing = [
            item for item in claim_analysis
            if item["bias_analysis_missing"]
        ]

        if missing:
            print(
                f"\nWARNING: {len(missing)} claim(s) were sent for bias "
                f"review but came back without a decision. They are "
                f"recorded as unbiased and flagged with "
                f"bias_analysis_missing=True:"
            )
            for item in missing:
                print(f"  - [{item['claim_id']}] {item['claim']}")

        print(f"\nTotal claims: {len(claims)}")
        print(f"Analysed claims: {len(claim_analysis)}")
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

        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

        print("\n" + "=" * 70)
        print("MULTIMODAL BIAS-CORRECTION PIPELINE")
        print("=" * 70)

        # ---------------------------------------------------------
        # INITIAL GENERATION (runs once)
        # ---------------------------------------------------------

        print("\n[1/6] INITIAL VLM GENERATION")
        print("-" * 70)

        initial_response = self.generate_initial_response(
            image_path,
            question
        )

        print(initial_response)

        current_response = initial_response

        correction_history = []

        previous_claim_texts = None
        completed_iterations = 0
        stop_reason = "max_iterations"

        # ---------------------------------------------------------
        # EACH ITERATION RE-ANALYSES THE CURRENT RESPONSE FROM
        # SCRATCH: EXTRACTION -> EVIDENCE -> BIAS -> CORRECT -> VERIFY
        # ---------------------------------------------------------

        for iteration in range(1, max_iterations + 1):

            print("\n" + "=" * 70)
            print(
                f"ANALYSIS + CORRECTION ITERATION "
                f"{iteration}/{max_iterations}"
            )
            print("=" * 70)

            # -----------------------------------------------------
            # CLAIM EXTRACTION
            # -----------------------------------------------------

            print("\n[2/6] CLAIM EXTRACTION")
            print("-" * 70)

            claims = self.extract_claims(current_response)

            # Zero claims from the original response means the
            # pipeline has nothing to work with at all. Zero claims
            # from an already-corrected response just means there is
            # nothing left to analyse.
            if not claims and iteration == 1:
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

            # Identical claims mean the previous correction changed nothing
            # of substance. Another round would re-flag the same claims and
            # spend corrector and verifier calls for no new information.
            claim_texts = [claim.claim for claim in claims]

            if claim_texts == previous_claim_texts:
                print(
                    "\nClaims are unchanged from the previous iteration. "
                    "Further correction cannot make progress. Stopping."
                )
                stop_reason = "no_progress"
                break

            previous_claim_texts = claim_texts

            # -----------------------------------------------------
            # EVIDENCE + BIAS ANALYSIS
            # -----------------------------------------------------

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

            # -----------------------------------------------------
            # SELF-CORRECTION
            # -----------------------------------------------------

            print("\n[5/6] SELF-CORRECTION")
            print("-" * 70)

            print(f"Flagged claims: {len(flagged_claims)}")

            if not flagged_claims:

                print("\nNo corrections required.")

                corrected_response = current_response

            else:

                corrected_response = self.correct_response(
                    image_path,
                    current_response,
                    flagged_claims
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
                "claims": [
                    claim.model_dump()
                    for claim in claims
                ],
                "claim_analysis": claim_analysis,
                "flagged_claims": flagged_claims,
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

            completed_iterations = iteration

            if verification_passed:

                print("\nVerification passed.")
                print(
                    "No problematic unsupported claims "
                    "were detected."
                )

                stop_reason = "verified"

                break

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

            if iteration == max_iterations:

                print(
                    "\nMaximum iterations reached. "
                    "Stopping with unresolved claims."
                )

                break

            # The corrected response becomes the input to the next
            # iteration, which re-extracts and re-checks it.
            current_response = corrected_response

        # ---------------------------------------------------------
        # RESULT
        #
        # Top-level fields describe the FINAL iteration.
        # correction_history holds the per-iteration detail.
        # ---------------------------------------------------------

        return {
            "question": question,
            "generation_mode": self.generator.mode,
            "initial_response": initial_response,
            "claims": [
                claim.model_dump()
                for claim in claims
            ],
            "claim_analysis": claim_analysis,
            "flagged_claims": flagged_claims,
            "corrected_response": corrected_response,
            "verification": verification.model_dump(),
            "verification_passed": verification_passed,
            "stop_reason": stop_reason,
            "correction_iterations": completed_iterations,
            "correction_history": correction_history
        }
