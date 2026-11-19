from types import SimpleNamespace

from pipeline.audit_gate import validate_bias_coverage, validate_evidence_coverage


def claim(text):
    return SimpleNamespace(claim=text)


def evidence(claim_id, text, status="UNSUPPORTED"):
    return SimpleNamespace(claim_id=claim_id, claim=text, evidence_status=status)


def decision(claim_id, text):
    return SimpleNamespace(claim_id=claim_id, claim=text)


def test_evidence_coverage_rejects_missing_and_misaligned_rows():
    issues = validate_evidence_coverage(
        [claim("A person is seated."), claim("A laptop is visible.")],
        [evidence(1, "A different claim.")],
    )
    assert {(item.claim_id, item.code) for item in issues} == {
        (1, "evidence_claim_mismatch"), (2, "missing_evidence")
    }


def test_bias_coverage_only_requires_non_supported_claims():
    issues = validate_bias_coverage(
        [evidence(1, "Visible chair.", "SUPPORTED"), evidence(2, "Person is a patient.")],
        [],
    )
    assert [(item.claim_id, item.code) for item in issues] == [
        (2, "missing_bias_decision")
    ]


def test_bias_coverage_accepts_whitespace_and_case_equivalence():
    issues = validate_bias_coverage(
        [evidence(1, "The person is a patient.")],
        [decision(1, " the PERSON is a patient. ")],
    )
    assert issues == []
