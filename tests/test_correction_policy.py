from types import SimpleNamespace

from pipeline.correction_policy import (
    CORRECTION_POLICY_VERSION,
    CORRECTION_PRINCIPLES,
    policy_text,
)
from pipeline.corrector import SelfCorrector
from pipeline.pipeline import BiasCorrectionPipeline
from pipeline.bias_monitor import BiasDecision, BiasDecisionList
from pipeline.claim_extractor import Claim, ClaimList
from pipeline.evidence_checker import EvidenceResult, EvidenceResultList
from pipeline.verifier import VerificationList
from utils.schemas import GenerationMode


def test_correction_policy_is_versioned_and_explicit():
    assert CORRECTION_POLICY_VERSION == "v1"
    assert len(CORRECTION_PRINCIPLES) >= 10
    assert "Preserve supported visual observations." in policy_text()
    assert "Do not introduce new facts during correction." in policy_text()


def test_structured_feedback_prompt_contains_explicit_fields():
    corrector = SelfCorrector.__new__(SelfCorrector)
    captured = {}
    corrector.use_structured_feedback = True
    corrector.use_correction_policy = True
    def complete(prompt, **kwargs):
        captured["prompt"] = prompt
        return "corrected"
    corrector.vlm = SimpleNamespace(
        complete=complete
    )
    result = corrector.correct(
        "image.jpg",
        "The person is a patient and is using a laptop.",
        [{
            "claim": "The person is a patient.",
            "claim_type": "inference",
            "bias_category": "medical_assumption",
            "evidence_status": "UNSUPPORTED",
            "evidence_confidence": 0.05,
            "evidence_explanation": "A wheelchair does not establish patient status.",
            "bias_explanation": "Medical status is inferred.",
            "severity": "medium",
        }],
        supported_claims=[{"claim": "The person is using a laptop."}],
    )
    prompt = captured["prompt"]
    assert result == "corrected"
    for expected in (
        "Claim type: inference",
        "Bias category: medical_assumption",
        "Evidence status: UNSUPPORTED",
        "Evidence confidence: 0.05",
        "Reason: A wheelchair does not establish patient status.",
        "Severity: medium",
        "The person is using a laptop.",
        "CORRECTION POLICY (v1)",
    ):
        assert expected in prompt


def test_unsupported_evidence_does_not_imply_bias():
    class FakeEvidence:
        def check_all(self, image_path, claims):
            return EvidenceResultList(results=[EvidenceResult(
                claim_id=1,
                claim=claims[0].claim,
                evidence_status="UNSUPPORTED",
                confidence=0.2,
                explanation="Not established visually.",
            )])

    class FakeMonitor:
        def analyze_all(self, evidence_results):
            return BiasDecisionList(results=[BiasDecision(
                claim_id=1,
                claim=evidence_results[0].claim,
                evidence_status="UNSUPPORTED",
                bias_detected=False,
                bias_category="none",
                severity="none",
                explanation="No personal or social assumption.",
                requires_correction=False,
            )])

    class FakeCorrector:
        PROMPT_VERSION = "test"
        def __init__(self):
            self.vlm = SimpleNamespace(calls=[])

    pipeline = BiasCorrectionPipeline.__new__(BiasCorrectionPipeline)
    pipeline.corrector = FakeCorrector()
    pipeline.evidence_checker = FakeEvidence()
    pipeline.bias_monitor = FakeMonitor()
    pipeline.ablation_config = {
        "evidence_checking": True,
        "bias_monitor": True,
    }
    result = pipeline.analyze_claims(
        "image.jpg",
        [Claim(
            claim="The object is expensive.",
            type="inference",
            category="socioeconomic",
            requires_visual_evidence=True,
        )],
    )
    assert result[0][0]["evidence_status"] == "UNSUPPORTED"
    assert result[0][0]["bias_detected"] is False
    assert result[0][0]["requires_correction"] is False


def test_pipeline_records_correction_metadata(monkeypatch):
    class FakeGenerator:
        mode = GenerationMode.real
        vlm = SimpleNamespace(calls=[], model_name="test-model")
        def __init__(self, *args, **kwargs):
            pass
        def generate(self, image_path, question):
            return "initial"
        def prompt_fingerprint(self):
            return "prompt"

    class FakeExtractor:
        PROMPT_VERSION = "extract"
        vlm = SimpleNamespace(calls=[])
        def extract(self, response):
            return ClaimList(claims=[Claim(
                claim="The person is a patient.",
                type="inference",
                category="medical",
                requires_visual_evidence=True,
            )])

    class FakeEvidence:
        PROMPT_VERSION = "evidence"
        vlm = SimpleNamespace(calls=[])
        def check_all(self, image_path, claims):
            return EvidenceResultList(results=[EvidenceResult(
                claim_id=1,
                claim=claims[0].claim,
                evidence_status="UNSUPPORTED",
                confidence=0.05,
                explanation="The image does not establish patient status.",
            )])

    class FakeMonitor:
        PROMPT_VERSION = "monitor"
        vlm = SimpleNamespace(calls=[])
        def analyze_all(self, evidence_results):
            return BiasDecisionList(results=[BiasDecision(
                claim_id=1,
                claim=evidence_results[0].claim,
                evidence_status="UNSUPPORTED",
                bias_detected=True,
                bias_category="medical_assumption",
                severity="medium",
                explanation="Medical status is inferred.",
                requires_correction=True,
            )])

    class FakeCorrector:
        PROMPT_VERSION = "correct"
        vlm = SimpleNamespace(calls=[])
        def correct(self, **kwargs):
            return "The person is seated."

    class FakeVerifier:
        PROMPT_VERSION = "verify"
        vlm = SimpleNamespace(calls=[])
        def verify(self, image_path, corrected_response):
            return VerificationList(claims=[])

    monkeypatch.setattr("pipeline.pipeline.InitialGenerator", FakeGenerator)
    monkeypatch.setattr("pipeline.pipeline.ClaimExtractor", FakeExtractor)
    monkeypatch.setattr("pipeline.pipeline.EvidenceChecker", FakeEvidence)
    monkeypatch.setattr("pipeline.pipeline.BiasMonitor", FakeMonitor)
    monkeypatch.setattr("pipeline.pipeline.SelfCorrector", FakeCorrector)
    monkeypatch.setattr("pipeline.pipeline.BiasVerifier", FakeVerifier)

    result = BiasCorrectionPipeline().run("image.jpg", "question")
    iteration = result["correction_history"][0]
    assert iteration["triggered"] is True
    assert iteration["flagged_claim_count"] == 1
    assert iteration["corrected_claim_count"] == 1
    assert iteration["policy_version"] == "v1"
    assert iteration["structured_feedback"][0]["evidence_confidence"] == 0.05
    assert result["configuration"]["correction_policy_version"] == "v1"
