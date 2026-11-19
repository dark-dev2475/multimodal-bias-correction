"""OFFLINE local report: read saved outputs/reviews, never call a model.

Similarity diagnostics, model-estimated semantic labels and operational failures
are distinct. Do not interpret this small development experiment as paper parity.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_reference_study import evaluate, read


def export(study, output):
    summary = evaluate(study, output, "ai")
    protocol = read(study / "protocol.json")
    refs = read(study / "frozen_references.json")
    rows = read(output / "per_response_scores.json")
    summary["review_warning"] = protocol["review_warning"]
    summary["judge_independence"] = False
    summary["reference_answerability"] = dict(Counter(r['reference']['answerability'] for r in refs.values()))
    summary["review_errors"] = len(list((study / "review_errors").glob("*.json")))
    report = ["# Version 2 — local 40-image development study", "",
        "**Provisional: AI-drafted references and same-model AI judgments, not human ground truth.**", "",
        "Generator/judge: Qwen3-VL-2B-Instruct, Q4_K_M weights, Q8_0 vision projector, local llama.cpp Vulkan. No paid API calls.", "",
        "EIGR selection/intervention algorithms are unchanged. All methods use JSON-schema decoding; task-derived source/quotation and judgment-ID constraints are enforced where applicable. Context quotes are restricted to full context or exact sentence spans. This is a transport adaptation, not a published reproduction.", "",
        "## Coverage and failures", "",
        "| Method | Planned | Outputs | Failures | Reviewed | Safe & correct / all planned* |",
        "|---|---:|---:|---:|---:|---:|"]
    for system, info in summary["systems"].items():
        selected = [r for r in rows if r["system"] == system]
        observed = sum(r['semantic']['safe_correct_rate'] for r in selected if r['semantic'] is not None)
        count = info["planned"] - info["status_counts"].get("not_collected", 0)
        info['operational_safe_correct_lower_bound'] = observed / info['planned']
        info['operational_denominator'] = info['planned']
        report.append(f"| {system} | {info['planned']} | {count} | {info['status_counts'].get('failed', 0)} | {info['semantically_reviewed']} | {observed:.0f}/{info['planned']} |")
    report += ["", "*Observed safe-correct AI labels divided by **all 40 planned examples**, including failures and missing labels in the denominator. This is a conservative operational lower bound, not a validated accuracy measurement.", "",
        "## Descriptive AI estimates on reviewed responses only", "",
        "| Method | Task correct | Unsupported social assertion | Required-fact recall | Token F1 (lexical only) |",
        "|---|---:|---:|---:|---:|"]
    def pct(value):
        return 'not available' if value is None else f'{100*value:.1f}%'
    for system, info in summary['systems'].items():
        m = info['descriptive_scored_only']
        report.append(f"| {system} | {pct(m['task_correctness'])} | {pct(m['unsupported_social_assumption_rate'])} | {pct(m['required_fact_recall'])} | {pct(info['lexical_diagnostic_only']['token_f1'])} |")
    report += ["", "## Interpretation limits", "",
        "- The references were frozen before this model produced answers and were never supplied to inference.",
        "- References remain pending human review. Generator and semantic judge share weights: errors may be correlated.",
        f"- Reference answerability: {summary['reference_answerability']}. This heavily ambiguous subset cannot establish preservation of usefulness on answerable tasks.",
        "- All 40 images are previously exposed development data, not a new held-out test set.",
        "- Failed EIGR outputs are not credited as successful abstentions. Review failures remain missing, never silently fixed to improve scores.",
        "- Self-Refine, Intent-Aware, Reflexion and Constitutional critique are inference-only VQA adaptations under a maximum nine-call budget (including the shared draft where applicable). Constitutional AI's full training method is not reproduced.",
        "- No match with published scores, superiority, or statistically established novelty is claimed.", "",
        "## Files to show", "",
        "- `EXAMPLES.md`: all 40 reference/draft/EIGR answers with status and JSON links.",
        "- `all_outputs.json`: every collected method output, including EIGR evidence graphs and repair certificates.",
        "- `metrics.json`: coverage, failures, lexical diagnostics and explicitly labelled AI estimates.",
        "- `per_response_scores.json`: separate testing scores for each planned method/image pair.",
        "- Study `manual_annotations/` and `blind_review/`: reference drafts and blank human-review forms."]
    examples = ["# All 40 examples — reference versus EIGR", "", "AI references are drafts; statuses are part of the result. Human review is still required.", ""]
    bundle = {"protocol_hash": protocol['protocol_hash'], "warning": protocol['review_warning'], "examples": []}
    for case in protocol['cases']:
        cid = case['example_id']
        outputs = {}
        for system in protocol['systems']:
            path = study / 'responses' / cid / (system + '.json')
            if path.exists():
                outputs[system] = read(path)
        bundle['examples'].append({'case': case, 'reference': refs[cid], 'outputs': outputs})
        examples += [f"## {cid}", "", f"Question: {case['question']}", "", f"Context: {case['context'] or '(none)'}", "",
                     f"Reference draft: {refs[cid]['reference']['reference_answer']}", ""]
        for system in ['raw', 'eigr']:
            result = outputs.get(system, {})
            examples += [f"**{system} — {result.get('status', 'not_collected')}**", "",
                         result.get('final_response') or '(no answer)', ""]
            if result.get('error'):
                examples += [f"Failure: {result['error']}", ""]
        json_path = (study / 'responses' / cid / 'eigr.json').resolve().as_posix()
        examples += [f"[Image](<{Path(case['image_path']).resolve().as_posix()}>) · [EIGR JSON](<{json_path}>)", ""]
    (output / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    (output / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    (output / 'EXAMPLES.md').write_text('\n'.join(examples) + '\n', encoding='utf-8')
    (output / 'all_outputs.json').write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({s: v['status_counts'] for s, v in summary['systems'].items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=ROOT / 'data/evaluation/reference_study_40_local')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/version2/reference_study_40_local')
    args = parser.parse_args()
    export(args.study, args.output)
