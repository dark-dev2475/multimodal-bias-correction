import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from pipeline.claim_extractor import ClaimExtractor
from pipeline.evidence_checker import EvidenceChecker


def main():

    image_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "images"
        / "wheelchair.jpg"
    )

    # -----------------------------------------
    # Same response we previously tested
    # -----------------------------------------

    response = """
The image shows two men inside what appears to be
a woodworking workshop or factory setting.

In the foreground, a man with a beard and short dark
hair is seated in a manual wheelchair. He is wearing
a light green t-shirt and tan work pants. He has a
silver laptop open on his lap.

In the background, another man is standing near
woodworking machinery wearing a blue plaid shirt
and jeans.
"""

    # -----------------------------------------
    # Claim extraction
    # -----------------------------------------

    extractor = ClaimExtractor()

    claim_result = extractor.extract(
        response
    )

    claims = claim_result.claims

    print("\n")
    print("=" * 60)
    print("CLAIMS")
    print("=" * 60)

    for i, claim in enumerate(
        claims,
        start=1
    ):

        print(
            f"{i}. {claim.claim}"
        )

    # -----------------------------------------
    # BATCH EVIDENCE CHECK
    # -----------------------------------------

    checker = EvidenceChecker()

    results = checker.check_all(
        image_path=image_path,
        claims=claims
    )

    print("\n")
    print("=" * 60)
    print("BATCH EVIDENCE RESULTS")
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
            f"Confidence: "
            f"{result.confidence}"
        )

        print(
            f"Explanation: "
            f"{result.explanation}"
        )


if __name__ == "__main__":
    main()