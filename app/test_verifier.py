import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from pipeline.verifier import BiasVerifier


def main():

    image_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "images"
        / "wheelchair.jpg"
    )

    corrected_response = """
A man in a wheelchair is using a laptop in a woodworking
workshop, with red ear protection resting around his neck.
In the background, another man works at a table saw wearing
yellow ear protection and a plaid shirt. The workshop
contains woodworking machinery, lumber stored on a cart,
and various tools along the walls.
"""

    verifier = BiasVerifier()

    result = verifier.verify(
        image_path=image_path,
        corrected_response=corrected_response
    )

    print("\n" + "=" * 60)
    print("VERIFICATION RESULT")
    print("=" * 60)

    for i, claim in enumerate(result.claims, start=1):

        print(f"\nClaim {i}")
        print(f"Statement: {claim.claim}")
        print(f"Evidence: {claim.evidence_status}")
        print(f"Problematic: {claim.problematic}")
        print(f"Explanation: {claim.explanation}")


if __name__ == "__main__":
    main()