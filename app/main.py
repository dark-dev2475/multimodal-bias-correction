import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from pipeline.pipeline import BiasCorrectionPipeline
from pipeline.generator import MODES


def parse_args():

    parser = argparse.ArgumentParser(
        description="Run the multimodal bias-correction pipeline."
    )

    parser.add_argument(
        "--image",
        default=PROJECT_ROOT / "data" / "raw" / "images" / "wheelchair.jpg",
        type=Path
    )

    parser.add_argument(
        "--question",
        default="What is happening in this image?"
    )

    parser.add_argument(
        "--mode",
        choices=MODES,
        default="real",
        help=(
            "real: ask the VLM to describe the image. "
            "adversarial: use a known-biased fixture response "
            "as a positive control for the detection stages."
        )
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2
    )

    return parser.parse_args()


def main():

    args = parse_args()

    pipeline = BiasCorrectionPipeline(generation_mode=args.mode)

    result = pipeline.run(
        image_path=args.image,
        question=args.question,
        max_iterations=args.max_iterations
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Generation mode:   {result['generation_mode']}")
    print(f"Stop reason:       {result['stop_reason']}")
    print(f"Iterations:        {result['correction_iterations']}")
    print(f"Verification:      "
          f"{'passed' if result['verification_passed'] else 'failed'}")


if __name__ == "__main__":
    main()
