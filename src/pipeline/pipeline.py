from collections import Counter
from inspect import signature

from pipeline.generator import InitialGenerator
from pipeline.claim_extractor import Claim, ClaimExtractor
from pipeline.evidence_checker import EvidenceChecker, EvidenceResult, EvidenceResultList
from pipeline.bias_monitor import BiasMonitor
from pipeline.corrector import SelfCorrector
from pipeline.correction_policy import CORRECTION_POLICY_VERSION
from pipeline.verifier import BiasVerifier, VerificationList
from pipeline.audit_gate import (
    AUDIT_GATE_VERSION,
    issue_dicts,
    validate_bias_coverage,
    validate_evidence_coverage,
)
from utils.schemas import (
    GenerationMode, PipelineResult, StageMetrics, StopReason,
)


# complete() tags every call with a human-readable source; this maps those
# onto the stage names used in reported metrics.
STAGE_BY_SOURCE = {
    "Initial generator": "generation",
    "Claim extractor": "extraction",
    "Evidence checker": "evidence",
    "Bias monitor": "bias_monitor",
    "Self-corrector": "correction",
    "Verifier": "verification",
}

STAGE_BY_CLASS = {
    "InitialGenerator": "generation",
    "ClaimExtractor": "extraction",
    "EvidenceChecker": "evidence",
    "BiasMonitor": "bias_monitor",
    "SelfCorrector": "correction",
    "BiasVerifier": "verification"
}

MAX_CORRECTION_ITERATIONS = 2


def claim_fingerprint(claims):
    """Order- and whitespace-insensitive signature of a set of claims.

    Two iterations that produce the same claims in a different order, or with
    different spacing or casing, have made no substantive progress.
    """

    return tuple(sorted(
        " ".join(claim.claim.split()).casefold()
        for claim in claims
    ))


class BiasCorrectionPipeline:

    def __init__(
        self,
        generation_mode=GenerationMode.real,
        enable_claim_extraction=True,
        enable_evidence_checking=True,
        enable_bias_monitor=True,
        enable_verification=True,
        use_structured_feedback=True,
        use_correction_policy=True,
        enable_audit_gate=True,
    ):

        self.generator = InitialGenerator(mode=generation_mode)
        self.claim_extractor = ClaimExtractor()
        self.evidence_checker = EvidenceChecker()
        self.bias_monitor = BiasMonitor()
        corrector_parameters = signature(SelfCorrector).parameters
        if {
            "use_structured_feedback",
            "use_correction_policy"
        }.issubset(corrector_parameters):
            self.corrector = SelfCorrector(
                use_structured_feedback=use_structured_feedback,
                use_correction_policy=use_correction_policy
            )
        else:
            self.corrector = SelfCorrector()
        self.verifier = BiasVerifier()
        self.ablation_config = {
            "claim_extraction": enable_claim_extraction,
            "evidence_checking": enable_evidence_checking,
            "bias_monitor": enable_bias_monitor,
            "verification": enable_verification,
            "structured_feedback": use_structured_feedback,
            "correction_policy": use_correction_policy,
            "audit_gate": enable_audit_gate,
        }

    # ---------------------------------------------------------
    # Execution metadata
    # ---------------------------------------------------------

    def _stages(self):

        return (
            self.generator,
            self.claim_extractor,
            self.evidence_checker,
            self.bias_monitor,
            self.corrector,
            self.verifier
        )

    def _reset_call_logs(self):

        for stage in self._stages():
            stage.vlm.calls.clear()
            cache = getattr(stage, "cache", None)
            if cache is not None:
                cache.hits = 0
                cache.misses = 0

    def _collect_stage_metrics(self):
        """Per-stage latency, retries and token usage for the run just done.

        A stage served entirely from cache makes no calls and so does not
        appear here, which is the intended signal rather than a gap.
        """

        metrics = {}

        for stage in self._stages():
            stage_name = STAGE_BY_CLASS.get(
                stage.__class__.__name__, stage.__class__.__name__
            )
            cache = getattr(stage, "cache", None)
            if cache is not None:
                metrics.setdefault(stage_name, StageMetrics())
                metrics[stage_name].cache_hits = cache.hits
                metrics[stage_name].cache_misses = cache.misses

            for call in stage.vlm.calls:

                name = STAGE_BY_SOURCE.get(call["source"], call["source"])
                entry = metrics.setdefault(name, StageMetrics())

                entry.calls += 1
                entry.retries += call["attempts"] - 1
                entry.latency_seconds = round(
                    entry.latency_seconds + call["latency_seconds"], 6
                )

                usage = call.get("usage")

                if usage:
                    entry.usage_reported_calls += 1

                    for field in ("prompt_tokens", "completion_tokens",
                                  "total_tokens"):
                        value = usage.get(field)

                        if value is not None:
                            current = getattr(entry, field) or 0
                            setattr(entry, field, current + value)

        return metrics

    def _prompt_versions(self):

        return {
            "extraction": self.claim_extractor.PROMPT_VERSION,
            "evidence": self.evidence_checker.PROMPT_VERSION,
            "bias_monitor": self.bias_monitor.PROMPT_VERSION,
            "correction": self.corrector.PROMPT_VERSION,
            "verification": self.verifier.PROMPT_VERSION,
            "correction_policy": CORRECTION_POLICY_VERSION,
            "audit_gate": AUDIT_GATE_VERSION,
        }

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

        if not self.ablation_config["claim_extraction"]:
            return [Claim(
                claim=response,
                type="observation",
                category="unsegmented_response",
                requires_visual_evidence=True
            )]

        result = self.claim_extractor.extract(response)

        return result.claims

    # ---------------------------------------------------------
    # STEP 3 + 4: Evidence Checking + Bias Monitoring
    # ---------------------------------------------------------
    def analyze_claims(self, image_path, claims):
        print("\n[3/6] Checking visual evidence...")

        if not claims:
            return [], []

        if self.ablation_config["evidence_checking"]:
            evidence_results = self.evidence_checker.check_all(
                image_path, claims
            )
        else:
            evidence_results = EvidenceResultList(results=[
                EvidenceResult(
                    claim_id=index,
                    claim=claim.claim,
                    evidence_status="UNCERTAIN",
                    confidence=0.0,
                    explanation="Evidence checking disabled for ablation."
                )
                for index, claim in enumerate(claims, start=1)
            ])

        print("\n[4/6] Monitoring for bias-relevant assumptions...")
        if self.ablation_config["bias_monitor"]:
            bias_results = self.bias_monitor.analyze_all(
                evidence_results.results
            )
            bias_by_id = {
                result.claim_id: result
                for result in bias_results.results
            }
        else:
            bias_by_id = {}

        audit_issues = []
        if self.ablation_config.get("audit_gate", True):
            audit_issues = validate_evidence_coverage(
                claims, evidence_results.results
            )
            # An intentionally removed bias monitor is a valid ablation, not
            # a malformed monitor response.  Only enforce its contract when
            # that stage actually ran.
            if self.ablation_config["bias_monitor"]:
                audit_issues += validate_bias_coverage(
                    evidence_results.results, bias_results.results
                )

        claim_types = {
            claim.claim_id if hasattr(claim, "claim_id") else index: claim.type
            for index, claim in enumerate(claims, start=1)
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
                "claim_type": claim_types.get(evidence.claim_id, "unknown"),
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

        # Fail closed: a malformed upstream response cannot become a silent
        # "no bias" verdict.  The corrector receives a targeted instruction
        # to remove/neutralise the affected claim.  This is a reliability
        # intervention, not a prediction that the claim is socially biased.
        by_id = {index: claim for index, claim in enumerate(claims, start=1)}
        failed_ids = {
            issue.claim_id for issue in audit_issues
            if issue.claim_id in by_id
            and issue.code in {"missing_evidence", "evidence_claim_mismatch", "missing_bias_decision", "bias_claim_mismatch"}
        }
        seen_ids = {item["claim_id"] for item in claim_analysis}
        for claim_id in sorted(failed_ids - seen_ids):
            claim = by_id[claim_id]
            claim_analysis.append({
                "claim_id": claim_id,
                "claim": claim.claim,
                "claim_type": claim.type,
                "evidence_status": "UNCERTAIN",
                "evidence_confidence": 0.0,
                "evidence_explanation": "Audit gate could not validate evidence-stage coverage.",
                "bias_detected": False,
                "bias_category": "audit_failure",
                "severity": "medium",
                "requires_correction": True,
                "bias_explanation": "Fail-closed correction: a required structured decision was missing or misaligned.",
                "bias_analysis_missing": True,
                "audit_failure": True,
            })
        for item in claim_analysis:
            if item["claim_id"] in failed_ids:
                item["audit_failure"] = True
                item["requires_correction"] = True
                if item["bias_category"] == "none":
                    item["bias_category"] = "audit_failure"
                    item["severity"] = "medium"
                    item["bias_explanation"] = "Fail-closed correction: a required structured decision was missing or misaligned."

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
        return claim_analysis, flagged_claims, issue_dicts(audit_issues)

    # ---------------------------------------------------------
    # STEP 5: Self Correction
    # ---------------------------------------------------------

    def correct_response(
        self,
        image_path,
        original_response,
        flagged_claims,
        supported_claims=None
    ):

        # Nothing to correct
        if not flagged_claims:

            return original_response

        correction_args = {
            "image_path": image_path,
            "original_response": original_response,
            "flagged_claims": flagged_claims,
        }
        if "supported_claims" in signature(
            self.corrector.correct
        ).parameters:
            correction_args["supported_claims"] = supported_claims or []
        corrected_response = self.corrector.correct(**correction_args)

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

    def run(
        self,
        image_path,
        question,
        max_iterations=MAX_CORRECTION_ITERATIONS
    ):

        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

        self._reset_call_logs()

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

        previous_fingerprint = None
        completed_iterations = 0
        stop_reason = StopReason.max_iterations

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
            fingerprint = claim_fingerprint(claims)

            if previous_fingerprint is not None \
                    and fingerprint == previous_fingerprint:
                print(
                    "\nClaims are unchanged from the previous iteration. "
                    "Further correction cannot make progress. Stopping."
                )
                stop_reason = StopReason.no_progress
                break

            previous_fingerprint = fingerprint

            # -----------------------------------------------------
            # EVIDENCE + BIAS ANALYSIS
            # -----------------------------------------------------

            claim_analysis, flagged_claims, audit_issues = self.analyze_claims(
                image_path,
                claims
            )

            supported_claims = [
                item for item in claim_analysis
                if item["evidence_status"] == "SUPPORTED"
            ]

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

            if audit_issues:
                print(f"Audit-gate issues: {len(audit_issues)} (fail-closed correction enabled)")

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
                    flagged_claims,
                    supported_claims
                )

            print("\nCORRECTED RESPONSE")
            print("-" * 70)
            print(corrected_response)

            # -----------------------------------------------------
            # VERIFICATION
            # -----------------------------------------------------

            print("\n[6/6] FINAL VERIFICATION")
            print("-" * 70)

            if self.ablation_config["verification"]:
                verification = self.verify_response(
                    image_path,
                    corrected_response
                )
            else:
                verification = VerificationList(claims=[])

            problematic_claims = [
                claim
                for claim in verification.claims
                if claim.problematic
            ]

            verification_passed = (
                self.ablation_config["verification"]
                and len(problematic_claims) == 0
            )

            supported_before = [item["claim"] for item in supported_claims]
            corrected_lower = corrected_response.casefold()
            supported_preserved = [
                claim for claim in supported_before
                if claim.casefold() in corrected_lower
            ]
            categories = sorted({
                item["bias_category"] for item in flagged_claims
                if item["bias_category"] != "none"
            })
            severity_distribution = dict(Counter(
                item["severity"] for item in flagged_claims
            ))
            structured_feedback = [
                {
                    "claim": item["claim"],
                    "claim_type": item.get("claim_type", "unknown"),
                    "bias_category": item["bias_category"],
                    "evidence_status": item["evidence_status"],
                    "evidence_confidence": item["evidence_confidence"],
                    "evidence_explanation": item["evidence_explanation"],
                    "severity": item["severity"],
                }
                for item in flagged_claims
            ]
            corrected_claim_count = sum(
                item["claim"].casefold() not in corrected_lower
                for item in flagged_claims
            )
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
                "verification_passed": verification_passed,
                "triggered": bool(flagged_claims),
                "flagged_claim_count": len(flagged_claims),
                "corrected_claim_count": corrected_claim_count,
                "remaining_problematic_claims": len(problematic_claims),
                "correction_categories": categories,
                "severity_distribution": severity_distribution,
                "policy_version": CORRECTION_POLICY_VERSION,
                "supported_claims_before": supported_before,
                "supported_claims_preserved": supported_preserved,
                "structured_feedback": structured_feedback,
                "audit_issues": audit_issues,
            })

            completed_iterations = iteration

            if verification_passed:

                print("\nVerification passed.")
                print(
                    "No problematic unsupported claims "
                    "were detected."
                )

                stop_reason = StopReason.verified

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

        # Validated on the way out: a shape change here fails loudly rather
        # than reaching the saved results as a silently missing field.
        return PipelineResult.model_validate({
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
            "correction_history": correction_history,
            "stage_metrics": self._collect_stage_metrics(),
            "prompt_versions": self._prompt_versions(),
            "configuration": {
                "max_iterations": max_iterations,
                "generation_mode": self.generator.mode.value,
                "correction_policy_version": CORRECTION_POLICY_VERSION,
                **self.ablation_config
            }
        }).model_dump(mode="json")
