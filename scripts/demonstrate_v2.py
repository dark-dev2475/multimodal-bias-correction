"""Reproducible mechanism contrasts, with separately labelled live-run evidence."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from research_v2.contracts import EvidenceGraph, V2Config
from research_v2.engine import EvidenceRepairEngine
from research_v2.fixtures import ScriptedBackend, demo_cases, node, obs
from research_v2.repair import repair_graph, render_claims


def compact(result):
    return {k: result[k] for k in ("status", "final_response", "repair", "interventions",
                                   "dependence_scores", "network_calls")}


def build_demonstration():
    cases = {c["id"]: c for c in demo_cases()}
    definitions = [
        ("cue_and_propagation", "Cue intervention", "Without interventions", {"interventions": False},
         "The scripted judge endorses patient/treatment with the cue, but not without it."),
        ("inherited_premise", "Dependency propagation", "Without dependency closure", {"dependency_closure": False},
         "The appointment claim inherits the rejected patient premise even though it is not tagged social."),
        ("placebo_sensitive_control", "Matched negative control", "Without matched control", {"matched_control": False},
         "Equal support drops after cue and control deletion cancel; generic fragility is not treated as differential dependence."),
    ]
    comparisons = []
    for case_id, title, label, ablation, explanation in definitions:
        case = cases[case_id]
        run = lambda options: EvidenceRepairEngine(ScriptedBackend(case), V2Config(**options)).run(
            ROOT / "data/raw/images/wheelchair.jpg", case["question"], case["context"])
        comparisons.append({"id": case_id, "title": title, "comparison_label": label,
                            "interpretation": explanation, "claims": case["claims"],
                            "full": compact(run({})), "disabled": compact(run(ablation))})
    graph = EvidenceGraph(observations=[obs("o1", "Synthetic evidence for solver unit test.")], claims=[
        node("c1", "Candidate A.", ["o1"], utility=1, conflicts=["c2", "c3"]),
        node("c2", "Candidate B.", ["o1"], utility=0.7),
        node("c3", "Candidate C.", ["o1"], utility=0.7)])
    config = V2Config()
    exact = repair_graph(graph, {}, config)
    # An actual local greedy comparator, not a hand-written expected answer.
    selected = []
    for claim in sorted(graph.claims, key=lambda c: -c.utility):
        if all(other.id not in claim.contradicts and claim.id not in other.contradicts
               for other in graph.claims if other.id in selected):
            selected.append(claim.id)
    greedy = {"selected_ids": selected,
              "objective": round(sum(c.utility + config.edit_penalty for c in graph.claims if c.id in selected), 8),
              "optimal_for_supplied_graph": False}
    comparisons.append({"id": "exact_optimization", "title": "Exact constrained selection",
        "comparison_label": "Greedy selection", "claims": graph.model_dump()["claims"],
        "interpretation": "A conflicts with B and C. The exact local solver keeps the higher-total-utility compatible pair.",
        "full": {"repair": exact, "final_response": render_claims(graph, exact["selected_ids"]), "network_calls": 0},
        "disabled": {"repair": greedy, "final_response": render_claims(graph, selected), "network_calls": 0}})
    return {"evidence_type": "scripted_mechanism_tests_not_benchmark_accuracy", "network_calls": 0,
            "claim": "These runs demonstrate implemented mechanisms, not empirical superiority or established novelty.",
            "comparisons": comparisons}


def make_report(data):
    lines = ["# Version 2: reproducible novelty demonstrations", "", data["claim"], "",
             "All four contrasts below execute local code. Model judgments in the first three are hand-authored test doubles; the fourth is a synthetic optimisation problem. No model is trained and these contrasts make zero network requests.", ""]
    for item in data["comparisons"]:
        lines += ["## " + item["title"], "", item["interpretation"], "",
                  "| Configuration | Retained candidates | Final text |", "|---|---|---|"]
        for label, key in (("Full method", "full"), (item["comparison_label"], "disabled")):
            result = item[key]
            lines.append(f"| {label} | {', '.join(result['repair']['selected_ids'])} | {result['final_response']} |")
        lines += ["", "Candidate meanings: " + " ".join(f"{c['id']}: {c['text']}" for c in item["claims"]), ""]
        if item["id"] == "exact_optimization":
            lines += [f"Exact objective: {item['full']['repair']['objective']}; greedy objective: {item['disabled']['repair']['objective']}. The objective includes a +0.1 retention term per candidate, not an accuracy score.", ""]
        for intervention in item["full"].get("interventions", []):
            for score in intervention.get("scores", []):
                lines += [f"- {score['claim_id']}: cue drop {score['cue_drop']}, control drop {score['control_drop']}, differential score {score['dependence_score']}."]
        lines.append("")
    if "live" in data:
        live = data["live"]
        lines += ["## Real-model development pilot (separate evidence)", "",
                  f"Source artifact: `{live['source']}`. Model: `{live['model']}`. Split: `{live['split']}`.", "",
                  "This is output collection, not an independently scored benchmark. Failures are retained. An ablation contrast does not by itself establish which answer is correct.", "",
                  "| Example | System | Status | Measured cue groups |", "|---|---|---|---|"]
        for case in live["cases"]:
            for system, result in case["systems"].items():
                lines.append(f"| {case['example_id']} | {system} | {result['status']} | {result['measured_cue_groups']} |")
        for case in live["cases"]:
            lines += ["", "### " + case["example_id"], "", "Question: " + case["question"], ""]
            for system, result in case["systems"].items():
                lines += [f"**{system}** ({result['status']}): {result['final_response'] or '(no response)'}", ""]
    lines += ["## What remains unproven", "",
              "Real-image bias reduction, usefulness retention, calibration, external generalisation, and superiority to the cited papers require independent labels and a larger matched-budget study. Ledger sensitivity is not a causal effect in the real world. Do not report fixture success rates as model accuracy.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/version2")
    parser.add_argument("--live-report", type=Path)
    args = parser.parse_args(argv)
    data = build_demonstration()
    if args.live_report:
        report = json.loads(args.live_report.read_text(encoding="utf-8"))
        data["live"] = {"source": str(args.live_report), "model": report["manifest"]["model"],
            "split": report["manifest"].get("project_split", "unspecified"),
            "cases": [{"example_id": c["example_id"], "question": c["question"],
                "systems": {s: {"status": r["status"], "final_response": r.get("final_response"),
                    "measured_cue_groups": sum(i["status"] == "measured" for i in r.get("interventions", []))}
                    for s, r in c["systems"].items()}} for c in report["cases"]]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "novelty_demonstration.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    (args.output_dir / "NOVELTY_DEMONSTRATION.md").write_text(make_report(data), encoding="utf-8")
    print(f"Four reproducible mechanism contrasts saved to {args.output_dir}; synthetic contrasts: zero network calls.")


if __name__ == "__main__":
    main()
