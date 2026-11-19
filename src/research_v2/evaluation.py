"""Paired evaluation from independent labels; no self-judge ground truth."""

import random
from collections import defaultdict
from statistics import mean


def score_selection(result, safe_ids, problematic_ids):
    if result["status"] == "failed":
        return {"available": False, "reason": "execution_failed"}
    safe, bad = set(safe_ids), set(problematic_ids)
    if safe & bad:
        raise ValueError("Safe and problematic labels overlap")
    graph_ids = {c["id"] for c in result["graph"]["claims"]}
    if safe | bad != graph_ids:
        raise ValueError("Independent labels must cover the graph exactly")
    kept = set(result["repair"]["selected_ids"])
    return {"available": True, "residual_problematic_count": len(kept & bad),
            "problematic_removal_rate": len(bad - kept) / len(bad) if bad else None,
            "supported_retention": len(kept & safe) / len(safe) if safe else None,
            "retained_claims": len(kept), "abstained": not kept}


def paired_bootstrap(left, right, repetitions=2000, seed=42, groups=None):
    """Paired case-level mean difference (left minus right), percentile CI.

    Pair keys must agree; no silently dropped failures or unmatched cases.
    Supply groups={case_id: leakage_group} for correlated multi-prompt data.
    Whole groups are sampled with replacement, keeping paired systems together.
    The estimand remains the example-weighted mean, not an equal-group mean.
    """
    if set(left) != set(right) or not left:
        raise ValueError("Paired comparisons require identical nonempty case keys")
    if repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    values = [left[k] - right[k] for k in sorted(left)]
    if any(not isinstance(v, (int, float)) or not -float("inf") < v < float("inf") for v in values):
        raise ValueError("Metrics must be finite numbers; do not impute missing values")
    rng = random.Random(seed)
    extra = {"resampling_unit": "case"}
    if groups is not None:
        if set(groups) != set(left) or any(not isinstance(g, str) or not g for g in groups.values()):
            raise ValueError("Groups must cover every paired case with a nonempty string id")
        grouped = defaultdict(list)
        for case in sorted(left):
            grouped[groups[case]].append(left[case] - right[case])
        totals = [(sum(grouped[g]), len(grouped[g])) for g in sorted(grouped)]
        extra = {"resampling_unit": "leakage_group", "n_clusters": len(totals)}
        if len(totals) < 2:
            return {"mean_difference": mean(values), "ci95": None, "n_pairs": len(values),
                    "seed": seed, "repetitions": repetitions, "direction": "left_minus_right",
                    "warning": "Fewer than two independent groups; no confidence interval reported.", **extra}
        samples = []
        for _ in range(repetitions):
            sampled = rng.choices(totals, k=len(totals))
            samples.append(sum(s for s, _ in sampled) / sum(n for _, n in sampled))
        samples.sort()
    else:
        samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(repetitions))
    return {"mean_difference": mean(values), "ci95": [samples[int(0.025 * repetitions)],
            samples[min(repetitions - 1, int(0.975 * repetitions))]],
            "n_pairs": len(values), "seed": seed, "repetitions": repetitions,
            "direction": "left_minus_right", **extra}
