import sys
import csv
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from utils.analysis import (
    load_batch_case_ids, load_experiment_runs, summarise,
)
from utils.logger import ExperimentLogger


def parse_args():

    parser = argparse.ArgumentParser(
        description="Summarise a Phase 4 experiment."
    )

    parser.add_argument(
        "--experiment-id",
        default="phase4_baseline_v1",
        help="Which experiment to analyse."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "outputs"
    )

    parser.add_argument(
        "--all-runs",
        action="store_true",
        help=(
            "Include every saved run per case. By default only the most "
            "recent run of each case is used, so repeated runs of one case "
            "do not outweigh the others."
        )
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Also write per-case rows to this CSV."
    )

    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Also write the full metrics object to this JSON file."
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List experiments that have saved runs, then exit."
    )

    return parser.parse_args()


def pct(value):

    return "n/a" if value is None else f"{value * 100:.1f}%"


def print_report(metrics):

    line = "=" * 78

    print("\n" + line)
    print(f"EXPERIMENT: {metrics.experiment_id}")
    print(line)

    print(f"Runs analysed:   {metrics.runs_analysed} "
          f"across {metrics.cases} case(s), "
          f"{metrics.distinct_images} image(s)")
    print(f"Model(s):        {', '.join(metrics.models) or 'unknown'}")
    print(f"Benchmark:       "
          f"{', '.join(metrics.benchmark_versions) or 'unknown'} "
          f"(hash {', '.join(metrics.benchmark_hashes) or 'unknown'})")
    print(f"Pipeline:        "
          f"{', '.join(metrics.pipeline_versions) or 'unknown'}")

    # ---------------- probes vs controls ----------------

    print("\n" + "-" * 78)
    print("BIAS FLAGS: INITIAL -> FINAL")
    print("-" * 78)
    print(f"{'group':<20} {'cases':>6} {'initial':>8} {'final':>8} "
          f"{'resolved':>9} {'rate':>8} {'verified':>9}")

    for name, group in (("bias probes", metrics.probes),
                        ("negative controls", metrics.controls)):
        print(f"{name:<20} {group.cases:>6} "
              f"{group.total_initial_flagged:>8} "
              f"{group.total_final_problematic:>8} "
              f"{group.flags_resolved:>9} "
              f"{pct(group.resolution_rate):>8} "
              f"{pct(group.verified_rate):>9}")

    print("\n'initial' counts claims the bias monitor flagged on the first "
          "response.\n'final' counts claims the verifier still found "
          "problematic in the final text.")

    # ---------------- controls ----------------

    false_positives = [r for r in metrics.per_case if r.is_false_positive]

    print("\n" + "-" * 78)
    print("FALSE POSITIVES (controls flagged when they should not be)")
    print("-" * 78)

    if not metrics.controls.cases:
        print("No control cases in this experiment.")
    elif not false_positives:
        print(f"None. All {metrics.controls.cases} control case(s) "
              f"produced zero initial flags.")
    else:
        for row in false_positives:
            print(f"  {row.case_id}: {row.initial_flagged} flag(s) "
                  f"{row.initial_categories}")

    # ---------------- categories ----------------

    print("\n" + "-" * 78)
    print("BIAS CATEGORIES")
    print("-" * 78)

    categories = sorted(
        set(metrics.category_initial) | set(metrics.category_final)
    )

    if not categories:
        print("No bias categories were flagged.")
    else:
        print(f"{'category':<28} {'initial':>8} {'last pass':>10}")
        for name in categories:
            print(f"{name:<28} "
                  f"{metrics.category_initial.get(name, 0):>8} "
                  f"{metrics.category_final.get(name, 0):>10}")

    # ---------------- convergence ----------------

    print("\n" + "-" * 78)
    print("CONVERGENCE")
    print("-" * 78)

    for reason, count in sorted(metrics.stop_reasons.items()):
        print(f"  {reason:<18} {count}")

    print(f"\n  mean iterations, probes:   {metrics.probes.mean_iterations}")
    print(f"  mean iterations, controls: {metrics.controls.mean_iterations}")

    # ---------------- retention ----------------

    print("\n" + "-" * 78)
    print("INFORMATION RETENTION (proxy)")
    print("-" * 78)
    print(f"  mean length ratio, probes:   "
          f"{metrics.probes.mean_length_ratio}")
    print(f"  mean length ratio, controls: "
          f"{metrics.controls.mean_length_ratio}")
    print("\nRatio of final to initial response length. Well below 1.0 "
          "suggests correction is\nstripping detail rather than rewriting "
          "it. This is a proxy, not a measure of\nmeaning preserved.")

    # ---------------- per case ----------------

    print("\n" + "-" * 78)
    print("PER CASE")
    print("-" * 78)
    print(f"{'case':<10} {'type':<17} {'flags':>9} {'iters':>5} "
          f"{'len':>6}  {'stop_reason'}")

    for row in metrics.per_case:
        flags = f"{row.initial_flagged}->{row.final_problematic}"
        print(f"{row.case_id:<10} {row.case_type.value:<17} {flags:>9} "
              f"{row.correction_iterations:>5} "
              f"{row.length_ratio:>6.2f}  {row.stop_reason}")

    # ---------------- caveats ----------------

    if metrics.caveats:
        print("\n" + "=" * 78)
        print("CAVEATS")
        print("=" * 78)
        for note in metrics.caveats:
            print(f"  - {note}")


def write_csv(metrics, path):

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row.model_dump(mode="json") for row in metrics.per_case]

    if not rows:
        return None

    for row in rows:
        row["initial_categories"] = "|".join(row["initial_categories"])
        row["final_categories"] = "|".join(row["final_categories"])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return path


def main():

    args = parse_args()

    if args.list:
        found = ExperimentLogger(
            output_dir=args.output_dir
        ).list_experiments()

        print("\n".join(found) if found else "No experiments found.")
        return 0

    runs = load_experiment_runs(
        args.output_dir,
        args.experiment_id,
        latest_only=not args.all_runs
    )

    if not runs:
        raise SystemExit(
            f"No saved runs for experiment {args.experiment_id!r}."
        )

    metrics = summarise(
        args.experiment_id,
        runs,
        load_batch_case_ids(args.output_dir, args.experiment_id)
    )

    print_report(metrics)

    if args.csv:
        print(f"\nPer-case CSV: {write_csv(metrics, args.csv)}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(metrics.model_dump(mode="json"),
                       indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"Metrics JSON: {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
