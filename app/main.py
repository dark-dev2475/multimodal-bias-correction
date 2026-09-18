import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from pipeline.pipeline import BiasCorrectionPipeline
from pipeline.generator import MODES
from utils.logger import ExperimentLogger


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

    parser.add_argument(
        "--case-id",
        default=None,
        help="Identifier for the saved run. Defaults to the image stem."
    )

    parser.add_argument(
        "--experiment-id",
        default="adhoc",
        help="Groups this run under <output-dir>/<experiment-id>/."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "outputs"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run without writing a result file."
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

    if args.no_save:
        return

    case_id = args.case_id or Path(args.image).stem

    output_file = ExperimentLogger(
        output_dir=args.output_dir
    ).save_run(
        case_id=case_id,
        image_path=args.image,
        model=pipeline.generator.vlm.model_name,
        result=result,
        experiment_id=args.experiment_id,
        max_iterations=args.max_iterations,
        prompt_hash=pipeline.generator.prompt_fingerprint()
    )

    print(f"Saved:             {output_file}")


if __name__ == "__main__":
    main()
