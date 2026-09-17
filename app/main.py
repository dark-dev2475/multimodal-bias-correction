import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from pipeline.pipeline import BiasCorrectionPipeline


def main():

    image_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "images"
        / "wheelchair.jpg"
    )

    question = "What is happening in this image?"

    pipeline = BiasCorrectionPipeline()

    result = pipeline.run(
        image_path=image_path,
        question=question
    )


if __name__ == "__main__":
    main()