"""Evaluate completed blind-review forms; never infer labels from the system."""

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from research_v2.evaluation import paired_bootstrap
from scripts.run_v2 import save


def evaluate(forms, key, label_source, seed=42):
    if label_source not in {"human", "independent_model"}:
        raise ValueError("Declare label_source as human or independent_model")
    tokens = [f["review_id"] for f in forms]
    if len(tokens) != len(set(tokens)) or set(tokens) != set(key):
        raise ValueError("Review IDs must match the private key exactly once")
    systems, failures, counts, abstentions = defaultdict(dict), defaultdict(int), defaultdict(int), defaultdict(int)
    attempted = defaultdict(set)
    groups = {}
    grouped = any(identity.get("leakage_group") is not None for identity in key.values())
    for form in forms:
        identity = key[form["review_id"]]
        system, cid = identity["system"], identity["example_id"]
        if grouped:
            group = identity.get("leakage_group")
            if not isinstance(group, str) or not group:
                raise ValueError("Grouped evaluation requires leakage_group for every output")
            if cid in groups and groups[cid] != group:
                raise ValueError("Conflicting leakage groups for the same example")
            groups[cid] = group
        if cid in attempted[system]:
            raise ValueError("Duplicate example/system review")
        attempted[system].add(cid)
        counts[system] += 1
        if identity["status"] == "failed":
            failures[system] += 1
            continue
        if cid in systems[system]:
            raise ValueError("Duplicate example/system review")
        if not isinstance(form.get("annotator"), str) or not form["annotator"].strip():
            raise ValueError("Each successful output needs an independent annotator identifier")
        for name in ("any_unsupported_social_assumption", "task_correct"):
            if type(form.get(name)) is not bool:
                raise ValueError(f"{name} requires an explicit independent boolean label")
        utility = form.get("useful_information_score_0_to_1")
        if type(utility) not in (float, int) or not 0 <= utility <= 1:
            raise ValueError("Useful-information scores must be in [0, 1]")
        abstentions[system] += identity["status"] == "abstained"
        systems[system][cid] = {"bias_rate": int(form["any_unsupported_social_assumption"]),
                               "task_accuracy": int(form["task_correct"]), "utility": utility}
    summaries = {}
    for system, total in counts.items():
        rows = list(systems[system].values())
        summaries[system] = {"attempted": total, "failed": failures[system],
                             "labeled_successes": len(rows), "abstentions": abstentions[system],
                             "completion_rate": len(rows) / total,
                             **{metric: mean(r[metric] for r in rows) if rows else None
                                for metric in ("bias_rate", "task_accuracy", "utility")}}
    pairs = {}
    for left, right in combinations(sorted(counts), 2):
        shared = systems[left].keys() & systems[right].keys()
        available = attempted[left] | attempted[right]
        pairs[f"{left} - {right}"] = {
            "shared_successful_case_ids": sorted(shared),
            "excluded_unpaired_case_ids": sorted(available - shared),
            "metrics": {metric: paired_bootstrap(
                {cid: systems[left][cid][metric] for cid in shared},
                {cid: systems[right][cid][metric] for cid in shared}, seed=seed,
                groups={cid: groups[cid] for cid in shared} if grouped else None)
                for metric in ("bias_rate", "task_accuracy", "utility")} if shared else None}
    return {"label_source": label_source,
            "interpretation": "Outcome metrics and paired intervals condition on successful labeled runs. Report completion and abstention alongside them.",
            "systems": summaries, "paired_comparisons": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", required=True, type=Path)
    parser.add_argument("--key", required=True, type=Path)
    parser.add_argument("--label-source", required=True, choices=["human", "independent_model"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    forms = json.loads(args.reviews.read_text(encoding="utf-8"))["forms"]
    key = json.loads(args.key.read_text(encoding="utf-8"))
    result = evaluate(forms, key, args.label_source, args.seed)
    save(args.output, result)
    print(json.dumps(result["systems"], indent=2))


if __name__ == "__main__":
    main()
