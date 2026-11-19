from types import SimpleNamespace

import pytest

from models import vlm as vlm_module
from pipeline import pipeline as pipeline_module
from pipeline.bias_monitor import BiasDecision, BiasDecisionList
from pipeline.claim_extractor import Claim, ClaimList
from pipeline.evidence_checker import EvidenceResult, EvidenceResultList
from pipeline.verifier import VerificationList, VerificationResult
from utils.chache import Cache
from utils.json_parse import parse_model_json
from utils.schemas import GenerationMode, StopReason


def _fake_vlm():
    return SimpleNamespace(model_name="test-model", calls=[])


class FakeGenerator:
    def __init__(self, *args, **kwargs):
        self.mode = GenerationMode.real
        self.vlm = _fake_vlm()

    def generate(self, image_path, question):
        return "initial"

    def prompt_fingerprint(self):
        return "prompt-hash"


class FakeExtractor:
    PROMPT_VERSION = "extract-test"

    def __init__(self):
        self.vlm = _fake_vlm()

    def extract(self, response):
        claim = "bad claim" if response == "initial" else response
        return ClaimList(claims=[Claim(
            claim=claim,
            type="inference",
            category="medical",
            requires_visual_evidence=True
        )])


class FakeEvidence:
    PROMPT_VERSION = "evidence-test"

    def __init__(self):
        self.vlm = _fake_vlm()

    def check_all(self, image_path, claims):
        return EvidenceResultList(results=[EvidenceResult(
            claim_id=1,
            claim=claims[0].claim,
            evidence_status="UNSUPPORTED",
            confidence=0.9,
            explanation="test evidence"
        )])


class FakeMonitor:
    PROMPT_VERSION = "monitor-test"

    def __init__(self):
        self.vlm = _fake_vlm()

    def analyze_all(self, evidence_results):
        return BiasDecisionList(results=[BiasDecision(
            claim_id=1,
            claim=evidence_results[0].claim,
            evidence_status="UNSUPPORTED",
            bias_detected=True,
            bias_category="medical_assumption",
            severity="high",
            explanation="test bias",
            requires_correction=True
        )])


class FakeCorrector:
    PROMPT_VERSION = "correct-test"

    def __init__(self):
        self.vlm = _fake_vlm()

    def correct(self, image_path, original_response, flagged_claims):
        return "corrected" if original_response == "initial" else "corrected-again"


class FakeVerifier:
    PROMPT_VERSION = "verify-test"

    def __init__(self):
        self.vlm = _fake_vlm()

    def verify(self, image_path, corrected_response):
        problematic = corrected_response == "corrected"
        claims = [] if not problematic else [VerificationResult(
            claim="bad claim",
            evidence_status="UNSUPPORTED",
            problematic=True,
            explanation="still problematic"
        )]
        return VerificationList(claims=claims)


def _mock_pipeline(monkeypatch):
    monkeypatch.setattr(pipeline_module, "InitialGenerator", FakeGenerator)
    monkeypatch.setattr(pipeline_module, "ClaimExtractor", FakeExtractor)
    monkeypatch.setattr(pipeline_module, "EvidenceChecker", FakeEvidence)
    monkeypatch.setattr(pipeline_module, "BiasMonitor", FakeMonitor)
    monkeypatch.setattr(pipeline_module, "SelfCorrector", FakeCorrector)
    monkeypatch.setattr(pipeline_module, "BiasVerifier", FakeVerifier)


def test_malformed_json_is_rejected():
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_model_json("{not json}", "test")


def test_api_failure_retries_and_records_attempts(monkeypatch):
    monkeypatch.setattr(vlm_module, "before_hosted_request", lambda *a: None)
    responses = iter([
        RuntimeError("temporary"),
        SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="ok")
            )],
            usage=None
        )
    ])

    class Completions:
        def create(self, **kwargs):
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    vlm = object.__new__(vlm_module.OpenRouterVLM)
    vlm.client = client
    vlm.api_key = "unit-test"
    vlm.model_name = "test-model"
    vlm.calls = []
    monkeypatch.setattr(vlm_module, "RETRYABLE_ERRORS", (RuntimeError,))
    monkeypatch.setattr(vlm_module.time, "sleep", lambda _: None)

    assert vlm.complete("prompt", max_attempts=2, base_delay=0) == "ok"
    assert vlm.calls[0]["attempts"] == 2


def test_empty_model_output_fails_instead_of_meaning_no_bias(monkeypatch):
    monkeypatch.setattr(vlm_module, "before_hosted_request", lambda *a: None)
    class Completions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="")
                )],
                usage=None
            )

    vlm = object.__new__(vlm_module.OpenRouterVLM)
    vlm.api_key = "unit-test"
    vlm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    vlm.model_name = "test-model"
    vlm.calls = []
    monkeypatch.setattr(vlm_module.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="after 2 attempts"):
        vlm.complete("prompt", max_attempts=2, base_delay=0)
    assert vlm.calls == []


def test_cache_hit_and_miss_are_recorded(tmp_path):
    cache = Cache(tmp_path)
    assert cache.get("test", {"key": 1}) is None
    cache.set("test", {"key": 1}, {"value": 2})
    assert cache.get("test", {"key": 1}) == {"value": 2}
    assert cache.misses == 1
    assert cache.hits == 1


def test_pipeline_runs_second_iteration(monkeypatch):
    _mock_pipeline(monkeypatch)
    runner = pipeline_module.BiasCorrectionPipeline()
    result = runner.run("image.jpg", "question", max_iterations=2)

    assert result["correction_iterations"] == 2
    assert result["stop_reason"] == StopReason.verified.value


def test_pipeline_stops_at_maximum_iterations(monkeypatch):
    _mock_pipeline(monkeypatch)
    monkeypatch.setattr(
        FakeVerifier,
        "verify",
        lambda self, image_path, corrected_response: VerificationList(
            claims=[VerificationResult(
                claim="bad claim",
                evidence_status="UNSUPPORTED",
                problematic=True,
                explanation="still problematic"
            )]
        )
    )
    runner = pipeline_module.BiasCorrectionPipeline()
    result = runner.run("image.jpg", "question", max_iterations=2)

    assert result["correction_iterations"] == 2
    assert result["stop_reason"] == StopReason.max_iterations.value


def test_correction_and_verifier_failures_propagate():
    runner = pipeline_module.BiasCorrectionPipeline.__new__(
        pipeline_module.BiasCorrectionPipeline
    )
    runner.corrector = SimpleNamespace(
        correct=lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("correction failed")
        )
    )
    with pytest.raises(RuntimeError, match="correction failed"):
        runner.correct_response("image.jpg", "response", [{"claim": "x"}])

    runner.verifier = SimpleNamespace(
        verify=lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("verifier failed")
        )
    )
    with pytest.raises(RuntimeError, match="verifier failed"):
        runner.verify_response("image.jpg", "response")
