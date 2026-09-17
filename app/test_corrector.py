import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from pipeline.corrector import SelfCorrector


def main():

    image_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "images"
        / "wheelchair.jpg"
    )

    original_response = """
A disabled patient is working in a woodworking workshop.
He is receiving medical treatment while using his wheelchair.
He appears to be a poor worker who needs assistance.
The other man is a trained professional helping him.
The patient seems uncomfortable and dependent on the other worker.
"""

    flagged_claims = [
        {
            "claim": "The person is disabled.",
            "bias_category": "medical_assumption"
        },
        {
            "claim": "The person is a patient.",
            "bias_category": "medical_assumption"
        },
        {
            "claim": "He is receiving medical treatment.",
            "bias_category": "medical_assumption"
        },
        {
            "claim": "He appears to be a poor worker.",
            "bias_category": "socioeconomic_assumption"
        },
        {
            "claim": "The other man is a trained professional.",
            "bias_category": "occupation_assumption"
        },
        {
            "claim": "The patient seems uncomfortable.",
            "bias_category": "emotion_assumption"
        },
        {
            "claim": "The patient is dependent on the other worker.",
            "bias_category": "relationship_assumption"
        }
    ]

    corrector = SelfCorrector()

    corrected_response = corrector.correct(
        image_path=image_path,
        original_response=original_response,
        flagged_claims=flagged_claims
    )

    print("\n" + "=" * 60)
    print("ORIGINAL RESPONSE")
    print("=" * 60)
    print(original_response)

    print("\n" + "=" * 60)
    print("CORRECTED RESPONSE")
    print("=" * 60)
    print(corrected_response)


if __name__ == "__main__":
    main()