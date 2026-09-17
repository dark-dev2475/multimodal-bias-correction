import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT / "src")
)

from pipeline.claim_extractor import ClaimExtractor


def main():

    response = """
    In the image, two men are inside a woodworking workshop.

    A man with a beard sits in a manual wheelchair facing
    toward the right. He is wearing a light green t-shirt,
    work pants, and has red earmuffs resting around his neck.
    He is actively using a laptop resting on his lap.

    Another man with yellow earmuffs over his ears, wearing
    a blue plaid shirt and jeans, stands in front of a large
    industrial woodworking machine and works with wooden
    materials.
    """

    extractor = ClaimExtractor()

    result = extractor.extract(response)

    print("\nCLAIMS")
    print("=" * 60)

    for i, claim in enumerate(result.claims, start=1):

        print(f"\nClaim {i}")
        print(f"Statement: {claim.claim}")
        print(f"Type: {claim.type}")
        print(f"Category: {claim.category}")
        print(
            f"Requires visual evidence: "
            f"{claim.requires_visual_evidence}"
        )


if __name__ == "__main__":
    main()