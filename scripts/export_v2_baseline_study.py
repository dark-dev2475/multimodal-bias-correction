"""Export a completed frozen study, independent-label forms and audit summary."""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from research_v2.backend import digest
from scripts.run_v2_baseline_study import summarise, report_markdown, save


def export(study, output, allow_incomplete=False):
    protocol = json.loads((study / "protocol.json").read_text(encoding="utf-8"))
    cases = []
    missing = []
    for cid in protocol["example_ids"]:
        path = study / "cases" / (cid + ".json")
        if not path.exists() and allow_incomplete:
            missing.append(cid)
            continue
        case = json.loads(path.read_text(encoding="utf-8"))
        if case["protocol_hash"] != protocol["protocol_hash"] or case["example_id"] != cid:
            raise ValueError("Case identity/protocol mismatch")
        if case["leakage_group"] != protocol["selected_groups"][cid]:
            raise ValueError("Case leakage-group mismatch")
        if set(case["systems"]) != set(protocol["systems"]) or set(case["reviews"]) != set(protocol["systems"]):
            raise ValueError("Incomplete system or review coverage")
        cases.append(case)
    if len({c["example_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case ids")
    source_match = all(digest((ROOT / name).read_text(encoding="utf-8")) == expected
                       for name, expected in protocol["source_hashes"].items())
    if not source_match:
        raise ValueError("Implementation differs from the frozen source hashes")
    if not cases and not allow_incomplete:
        raise ValueError("No completed case records to export")
    summary = (summarise(cases) if not missing else {
        "evaluation_type": "incomplete_collection_no_performance_conclusion",
        "planned_examples": len(protocol["example_ids"]), "completed_examples": len(cases),
        "missing_example_ids": missing, "published_paper_scores_reproduced": False,
        "performance_match_established": False,
        "model_failure_counts_in_completed_cases": dict(Counter(s for c in cases for s, r in c["systems"].items() if r["status"] == "failed")),
        "scored_outputs_in_completed_cases": sum(r["status"] == "completed" for c in cases for r in c["reviews"].values()),
        "category_counts_in_completed_cases": dict(Counter(c["category"] for c in cases)),
        "warning": "Stopped by provider access/rate limits. The completed prefix is not the planned category-balanced sample. No performance table is released from this incomplete run."})
    key, forms, automated = {}, [], []
    problems = []
    for case in cases:
        for system, result in case["systems"].items():
            token = digest([case["example_id"], system, result.get("final_response")])[:16]
            key[token] = {"example_id": case["example_id"], "system": system,
                          "status": result["status"], "leakage_group": case["leakage_group"]}
            form = {"review_id": token, "image_path": case["image_path"], "question": case["question"],
                    "context": case["context"], "response": result.get("final_response"),
                    "any_unsupported_social_assumption": None, "task_correct": None,
                    "useful_information_score_0_to_1": None, "annotator": None}
            forms.append(form)
            review = case["reviews"][system]
            if review["status"] == "completed":
                automated.append({**form, **review["labels"], "annotator": protocol["judge_model"]})
            else:
                automated.append(form)
            if result["status"] == "failed" or review["status"] == "failed":
                problems.append({"example_id": case["example_id"], "system": system,
                                 "model_error": result.get("error"), "judge_error": review.get("error")})
    random.Random(protocol["seed"]).shuffle(forms)
    random.Random(protocol["seed"]).shuffle(automated)
    response_models = Counter(call.get("response_model", "not_reported_or_cache")
                             for case in cases for call in case["physical_call_log"])
    audit = {"all_selected_cases_present": not missing, "all_systems_present_for_completed_cases": True,
             "source_hashes_match_frozen_protocol": source_match, "independent_human_labels": False,
             "response_models": dict(response_models), "failed_records": problems,
             "dataset": "custom BBQ-V held-out subset", "protocol_hash": protocol["protocol_hash"]}
    for name, value in (("protocol.json", protocol), ("summary.json", summary), ("audit.json", audit),
                        ("blind_review.json", {"forms": forms}),
                        ("automated_review.json", {"annotation_type": "external_model_estimates", "forms": automated}),
                        ("review_key_private.json", key)):
        save(output / name, value)
    if missing:
        blocked_path = study / "BLOCKED.json"
        reason = json.loads(blocked_path.read_text(encoding="utf-8"))["reason"] if blocked_path.exists() else "Incomplete collection"
        report = ("# Version 2 baseline comparison — incomplete\n\n"
            f"**{len(cases)} of {len(protocol['example_ids'])} selected cases completed. No baseline match has been established.**\n\n"
            f"Stop reason: {reason}.\n\n"
            f"Generator: `{protocol['model']}`. External evaluator: `{protocol['judge_model']}`.\n\n"
            "Self-Refine and Intent-Aware are documented VQA adaptations, not reproductions of published numerical scores. "
            "The completed prefix is not category-balanced, so no aggregate performance table is released. "
            "Do not reinterpret unprocessed examples as model errors or zero-accuracy observations.\n\n"
            f"Saved outputs: {len(forms)}. Successfully scored outputs: {summary['scored_outputs_in_completed_cases']}. "
            "Model failures remain in the records; missing human labels have not been invented.\n\n"
            "The free-only study is separate from historical paid-model results. Resume using exactly the "
            "same arguments once quota/provider availability permits. Completed stages and failures are "
            "preserved. No paid fallback, model switching or test-outcome tuning is permitted. "
            "See README_VERSION_2.md, section 14, for the free-only workflow.\n")
    else:
        report = report_markdown(protocol, summary)
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Exported {len(cases)} cases / {len(forms)} blinded outputs to {output}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "data/outputs/v2/free_baseline_study_36")
    parser.add_argument("--output", type=Path, default=ROOT / "output/version2/free_baseline_comparison")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    export(args.study, args.output, args.allow_incomplete)
