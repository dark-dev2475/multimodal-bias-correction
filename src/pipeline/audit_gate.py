"""Deterministic safety checks between probabilistic pipeline stages.

The VLM stages in this project emit structured decisions.  A missing decision
is not evidence that a claim is safe: treating it as such creates a silent
false-negative path.  This module is deliberately model-free so it can be
used at inference time without training, additional API calls, or a second
judge model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


AUDIT_GATE_VERSION = "v1"


@dataclass(frozen=True)
class AuditIssue:
    """A contract violation that could otherwise hide an unsafe claim."""

    claim_id: int
    code: str
    detail: str


def _normalise(text: str) -> str:
    return " ".join(text.split()).casefold()


def validate_evidence_coverage(claims: Iterable, evidence_results: Iterable) -> list[AuditIssue]:
    """Return evidence-stage alignment errors without attempting repair."""

    expected = {index: _normalise(claim.claim) for index, claim in enumerate(claims, 1)}
    received = {result.claim_id: _normalise(result.claim) for result in evidence_results}
    issues = []
    for claim_id, claim in expected.items():
        if claim_id not in received:
            issues.append(AuditIssue(claim_id, "missing_evidence", "No evidence decision returned."))
        elif received[claim_id] != claim:
            issues.append(AuditIssue(claim_id, "evidence_claim_mismatch", "Evidence decision text does not match the extracted claim."))
    for claim_id in received.keys() - expected.keys():
        issues.append(AuditIssue(claim_id, "unexpected_evidence", "Evidence stage returned an unknown claim id."))
    return issues


def validate_bias_coverage(evidence_results: Iterable, bias_results: Iterable) -> list[AuditIssue]:
    """Ensure every non-supported claim receives exactly one bias decision."""

    expected = {
        result.claim_id: _normalise(result.claim)
        for result in evidence_results
        if result.evidence_status in ("UNSUPPORTED", "UNCERTAIN")
    }
    received = {result.claim_id: _normalise(result.claim) for result in bias_results}
    issues = []
    for claim_id, claim in expected.items():
        if claim_id not in received:
            issues.append(AuditIssue(claim_id, "missing_bias_decision", "No bias decision returned for a non-supported claim."))
        elif received[claim_id] != claim:
            issues.append(AuditIssue(claim_id, "bias_claim_mismatch", "Bias decision text does not match the evidence claim."))
    for claim_id in received.keys() - expected.keys():
        issues.append(AuditIssue(claim_id, "unexpected_bias_decision", "Bias stage returned a decision for an ineligible claim id."))
    return issues


def issue_dicts(issues: Iterable[AuditIssue]) -> list[dict]:
    return [{"claim_id": issue.claim_id, "code": issue.code, "detail": issue.detail} for issue in issues]
