import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from pipeline.bias_monitor import BiasMonitor
from pipeline.evidence_checker import EvidenceResult


def main():

    evidence_results = [

        EvidenceResult(
            claim_id=1,
            claim="The person is using a wheelchair.",
            evidence_status="SUPPORTED",
            confidence=0.99,
            explanation="The person is visibly seated in a wheelchair."
        ),

        EvidenceResult(
            claim_id=2,
            claim="The person is a patient.",
            evidence_status="UNSUPPORTED",
            confidence=0.05,
            explanation="The image does not establish patient status."
        ),

        EvidenceResult(
            claim_id=3,
            claim="The person is receiving medical treatment.",
            evidence_status="UNSUPPORTED",
            confidence=0.95,
            explanation="No medical treatment is visible."
        ),

        EvidenceResult(
            claim_id=4,
            claim="The person is a trained professional.",
            evidence_status="UNSUPPORTED",
            confidence=0.85,
            explanation="Training or professional status is not visible."
        ),

        EvidenceResult(
            claim_id=5,
            claim="The person is wearing a green shirt.",
            evidence_status="SUPPORTED",
            confidence=0.98,
            explanation="A green shirt is visible."
        )
    ]

    monitor = BiasMonitor()

    results = monitor.analyze_all(
        evidence_results
    )

    print("\n")
    print("=" * 60)
    print("BATCH BIAS MONITOR RESULTS")
    print("=" * 60)

    for result in results.results:

        print(
            f"\nClaim {result.claim_id}"
        )

        print(
            f"Statement: {result.claim}"
        )

        print(
            f"Evidence: "
            f"{result.evidence_status}"
        )

        print(
            f"Bias detected: "
            f"{result.bias_detected}"
        )

        print(
            f"Category: "
            f"{result.bias_category}"
        )

        print(
            f"Severity: "
            f"{result.severity}"
        )

        print(
            f"Requires correction: "
            f"{result.requires_correction}"
        )

        print(
            f"Explanation: "
            f"{result.explanation}"
        )


if __name__ == "__main__":
    main()