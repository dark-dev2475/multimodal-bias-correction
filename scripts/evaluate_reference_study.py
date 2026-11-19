"""OFFLINE evaluation: reads frozen references, outputs and separate review labels.

Never calls a model or imports the inference pipeline. Lexical similarity is NOT
semantic correctness or fairness. Human and AI-review modes cannot be mixed.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.reference_protocol import Reference, SemanticReview


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def tokens(text):
    return re.findall(r"\w+", text.casefold())


def harmonic(a, b):
    return 2 * a * b / (a + b) if a + b else 0.0


def similarity(answer, reference):
    a, b = tokens(answer), tokens(reference)
    overlap = sum((Counter(a) & Counter(b)).values())
    token_f1 = harmonic(overlap / len(a), overlap / len(b)) if a and b else 0.0
    # Longest common subsequence, O(len(a)*len(b)) time and O(len(b)) storage.
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, 1):
            current.append(previous[j-1] + 1 if x == y else max(previous[j], current[-1]))
        previous = current
    lcs = previous[-1]
    rouge = harmonic(lcs / len(a), lcs / len(b)) if a and b else 0.0
    return {"token_f1": token_f1, "rouge_l_f1": rouge}


def make_units(answer):
    return {f"u{i}": text for i, text in enumerate(filter(None, (x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", answer))))}


def semantic_metrics(labels, reference):
    precision = sum(u.support == "supported" for u in labels.units) / len(labels.units)
    biased = any(u.asserts_unsupported_social_assumption for u in labels.units)
    recall = len(labels.covered_required_fact_ids) / len(reference.required_fact_ids) if reference.required_fact_ids else None
    return {"task_correctness": float(labels.task_correct), "unsupported_social_assumption_rate": float(biased),
            "safe_correct_rate": float(labels.task_correct and not biased),
            "supported_unit_precision": precision, "required_fact_recall": recall,
            "grounded_content_f1": harmonic(precision, recall) if recall is not None else None,
            "false_abstention_rate": float(labels.abstains_from_answer) if reference.answerability == "answerable" else None}


def evaluate(study, output, label_source="human"):
    inventory = read(study / "inventory.json")
    annotation_counts = Counter(read(study / "manual_annotations" / (c["example_id"] + ".json"))["status"] for c in inventory["cases"])
    summary = {"scope": inventory["scope"], "planned_examples": len(inventory["cases"]),
               "annotation_status": dict(annotation_counts), "review_source": label_source,
               "published_scores_reproduced": False, "superiority_established": False,
               "warning": "Original 40 are exposed DEVELOPMENT data, not held-out test evidence. Similarity is not fairness/accuracy."}
    rows, scores = [], {}
    protocol_path = study / "protocol.json"
    if not protocol_path.exists():
        summary.update(status="awaiting_complete_frozen_references", systems={}, paired_comparisons={})
    else:
        protocol = read(protocol_path)
        expected_hash = protocol["protocol_hash"]
        if digest({k: v for k, v in protocol.items() if k != "protocol_hash"}) != expected_hash:
            raise ValueError("Protocol hash mismatch")
        references = read(study / "frozen_references.json")
        if digest(references) != protocol["reference_hash"]:
            raise ValueError("Reference content changed since freeze")
        if protocol["cases"] != inventory["cases"]:
            raise ValueError("Inventory differs from frozen protocol")
        for case in protocol["cases"]:
            if hashlib.sha256(Path(case["image_path"]).read_bytes()).hexdigest() != case["image_sha256"]:
                raise ValueError("Image changed since reference freeze")
        summary["reference_quality"] = protocol["reference_quality"]
        summary["generator_model"] = protocol["model"]
        for system in protocol["systems"]:
            system_rows, scores[system] = [], {}
            for case in protocol["cases"]:
                cid = case["example_id"]
                ref = Reference.model_validate(references[cid]["reference"])
                path = study / "responses" / cid / (system + ".json")
                row = {"example_id": cid, "system": system, "category": case["category"], "leakage_group": case["leakage_group"], "status": "not_collected", "semantic": None}
                if path.exists():
                    result = read(path)
                    if result["protocol_hash"] != expected_hash or result["example_id"] != cid or result["system"] != system:
                        raise ValueError("Response identity/protocol mismatch")
                    row["status"] = result["status"]
                    row["attributed_model_calls"] = len(result.get("calls", [])) + len(result.get("shared_draft_calls", []))
                    answer = result.get("final_response")
                    if answer and result["status"] != "failed":
                        row["similarity"] = similarity(answer, ref.reference_answer)
                        token = digest([expected_hash, cid, answer])[:24]
                        label_path = study / ("ai_reviews" if label_source == "ai" else "blind_review") / (token + ".json")
                        if label_path.exists():
                            review = read(label_path)
                            if review.get("labels") is not None:
                                origin = "ai_assisted" if label_source == "ai" else "human"
                                if review.get("annotation_origin") != origin or not review.get("reviewer") or not review.get("reviewed_at"):
                                    raise ValueError("Review origin/reviewer metadata invalid; never relabel AI output as manual")
                                if any(review.get(k) != v for k, v in {"review_id": token, "answer": answer, "question": case["question"],
                                    "context": case["context"], "image_path": case["image_path"], "reference": ref.model_dump(), "units": make_units(answer)}.items()):
                                    raise ValueError("Review is not aligned with this reference/answer/task")
                                labels = SemanticReview.model_validate(review["labels"]).validate_coverage(make_units(answer), ref)
                                row["semantic"] = semantic_metrics(labels, ref)
                                scores[system][cid] = row["semantic"]
                rows.append(row)
                system_rows.append(row)
            judged = [r["semantic"] for r in system_rows if r["semantic"] is not None]
            lexical = [r["similarity"] for r in system_rows if "similarity" in r]
            summary.setdefault("systems", {})[system] = {
                "planned": len(system_rows), "status_counts": dict(Counter(r["status"] for r in system_rows)),
                "semantically_reviewed": len(judged), "review_coverage": len(judged) / len(system_rows),
                "descriptive_scored_only": {k: mean(vals) if (vals := [r[k] for r in judged if r[k] is not None]) else None
                    for k in ("task_correctness", "unsupported_social_assumption_rate", "safe_correct_rate", "supported_unit_precision", "required_fact_recall", "grounded_content_f1", "false_abstention_rate")},
                "lexical_diagnostic_only": {k: mean(r[k] for r in lexical) if lexical else None for k in ("token_f1", "rouge_l_f1")},
                "missing_or_failed_outputs_are_not_counted_as_success": True}
        complete = all(len(v) == len(protocol["cases"]) for v in scores.values())
        summary["status"] = "complete_descriptive_development_evaluation" if complete else "incomplete_no_performance_conclusion"
        summary["paired_comparisons"] = {}
        if complete:
            # Statistics-only module; no client/pipeline imports.
            from research_v2.evaluation import paired_bootstrap
            groups = {c["example_id"]: c["leakage_group"] for c in protocol["cases"]}
            for system in protocol["systems"]:
                if system != "eigr":
                    summary["paired_comparisons"][system] = {k: paired_bootstrap(
                        {cid: v[k] for cid, v in scores["eigr"].items()},
                        {cid: v[k] for cid, v in scores[system].items()}, groups=groups)
                        for k in ("task_correctness", "unsupported_social_assumption_rate", "safe_correct_rate")}
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "per_response_scores.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    lines = ["# Reference-first evaluation — Version 2", "", "Status: " + summary["status"], "",
             summary["warning"], "", f"Annotations: {dict(annotation_counts)}", "",
             "Human labels and AI-estimated labels are kept separate. Pending records are not completed annotations.", "",
             "| Method | Planned | Reviewed | Missing outputs | Failed outputs |", "|---|---:|---:|---:|---:|"]
    for system, row in summary["systems"].items():
        lines.append(f"| {system} | {row['planned']} | {row['semantically_reviewed']} | {row['status_counts'].get('not_collected', 0)} | {row['status_counts'].get('failed', 0)} |")
    lines += ["", "No published-score match or superiority is established. See metrics.json for explicitly labelled descriptive diagnostics.",
              "", "Hypothesis: evidence interventions and dependency repair improve safe-correct answers while retaining required facts. This is not proven novelty."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "annotations": dict(annotation_counts)}))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "data/evaluation/reference_study_40")
    parser.add_argument("--output", type=Path, default=ROOT / "output/version2/reference_study_40")
    parser.add_argument("--labels", choices=["human", "ai"], default="human")
    args = parser.parse_args()
    evaluate(args.study, args.output, args.labels)
