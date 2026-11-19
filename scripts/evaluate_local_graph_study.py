"""Offline presentation wrapper disclosing development-informed structural fixes."""
import json
import sys
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_local_reference_study import export

STUDY = ROOT / 'data/evaluation/reference_study_40_local_graphids'
OUTPUT = ROOT / 'output/version2/reference_study_40_local_graphids'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=STUDY)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    STUDY, OUTPUT = args.study, args.output
    export(STUDY, OUTPUT)
    protocol = json.loads((STUDY / 'protocol.json').read_text(encoding='utf-8'))
    disclosure = protocol['development_adaptation_disclosure']
    metrics_path = OUTPUT / 'metrics.json'
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    rows = json.loads((OUTPUT / 'per_response_scores.json').read_text(encoding='utf-8'))
    metrics.update(engineering_followup=True, development_adaptation_disclosure=disclosure,
                   structural_schema_specialization=protocol['structural_schema_specialization'],
                   report_generated_at=datetime.now(timezone.utc).isoformat())
    metrics['collection_attempts_complete'] = all(r['status'] != 'not_collected' for r in rows)
    valid = [r for r in rows if r['status'] not in {'not_collected', 'failed'}]
    metrics['valid_outputs_reviewed_fully'] = all(r['semantic'] is not None for r in valid) and bool(valid)
    for system, summary in metrics['systems'].items():
        judged = [r['semantic'] for r in rows if r['system'] == system and r['semantic'] is not None]
        summary['semantic_metric_denominators'] = {key: sum(m[key] is not None for m in judged)
            for key in summary['descriptive_scored_only']}
    outputs = [json.loads(path.read_text(encoding='utf-8')) for path in (STUDY / 'responses').glob('*/eigr.json')]
    mechanism = {'attempted_cases': len(outputs), 'cases_with_saved_graph': 0,
        'cases_with_dependency_edges': 0, 'dependency_edges': 0, 'cases_with_measured_interventions': 0,
        'measured_intervention_contrasts': 0, 'intervention_status_counts': {},
        'verification_calls_with_any_rejected_claim': 0, 'answers_changed_from_draft': 0,
        'warning': 'Mechanism activation and structural certificates do not prove correctness, bias reduction, or novelty.'}
    statuses = Counter()
    for row in outputs:
        graph = row.get('graph')
        if graph:
            mechanism['cases_with_saved_graph'] += 1
            edges = sum(len(claim['depends_on']) for claim in graph['claims'])
            mechanism['dependency_edges'] += edges
            mechanism['cases_with_dependency_edges'] += int(edges > 0)
        plans = row.get('interventions', [])
        measured = sum(plan['status'] == 'measured' for plan in plans)
        mechanism['cases_with_measured_interventions'] += int(measured > 0)
        mechanism['measured_intervention_contrasts'] += measured
        statuses.update(plan['status'] for plan in plans)
        verification = row.get('verification') or {}
        mechanism['verification_calls_with_any_rejected_claim'] += int(any(
            j['support'] != 'supported' for j in verification.get('judgments', [])))
        if row['status'] != 'failed':
            mechanism['answers_changed_from_draft'] += int(row.get('final_response') != row.get('initial_response'))
    mechanism['intervention_status_counts'] = dict(statuses)
    metrics['mechanism_activation'] = mechanism
    (OUTPUT / 'mechanism_audit.json').write_text(json.dumps(mechanism, indent=2) + '\n', encoding='utf-8')
    references = json.loads((STUDY / 'frozen_references.json').read_text(encoding='utf-8'))
    by_case = {r['example_id']: r for r in rows if r['system'] == 'eigr'}
    answers, matched = [], []
    for case in protocol['cases']:
        cid = case['example_id']
        source = STUDY / 'responses' / cid / 'eigr.json'
        result = json.loads(source.read_text(encoding='utf-8')) if source.exists() else {}
        answer = {'example_id': cid, 'image_path': case['image_path'], 'question': case['question'],
            'context': case['context'], 'status': result.get('status', 'not_collected'),
            'final_response': result.get('final_response'), 'detail_json': str(source.relative_to(ROOT)),
            'model': protocol['model'], 'protocol_hash': protocol['protocol_hash']}
        answers.append(answer)
        matched.append({**answer, 'reference_answer': references[cid]['reference']['reference_answer'],
            'reference_status': references[cid]['status'], 'scores': by_case[cid]})
    for filename, payload in [('final_responses_40.json', {'warning': 'Completion/verification status is not proof of truth.', 'responses': answers}),
        ('annotations_40.json', {'warning': 'AI-assisted draft references, not approved human ground truth.', 'annotations': references}),
        ('reference_similarity_40.json', {'warning': 'Lexical similarity is not accuracy/fairness. Semantic labels, where present, are same-model AI estimates.', 'results': matched})]:
        (OUTPUT / filename).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    metrics_path.write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    report_path = OUTPUT / 'REPORT.md'
    report = report_path.read_text(encoding='utf-8')
    report = report.replace('# Version 2 — local 40-image development study',
        '# Version 2 — local graph-ID engineering study\n\n**Development-informed change:** ' + disclosure, 1)
    report += '\n## Change from the initial local study\n\n' + protocol['structural_schema_specialization'] + '\n\n'
    report += 'The initial local run is preserved separately, including all graph failures. No benchmark answers or reference labels were used to choose support labels or repair outputs.\n'
    report += '\n## Completion and mechanism audit\n\n'
    report += f"All 280 planned method/image attempts recorded: **{metrics['collection_attempts_complete']}**. "
    report += f"Every valid output has an AI review: **{metrics['valid_outputs_reviewed_fully']}**. "
    report += 'Failed inference attempts are not valid outputs; missing semantic labels remain missing. Human approval remains pending.\n\n'
    report += '| Mechanism | Cases |\n|---|---:|\n'
    report += f"| Saved evidence graph | {mechanism['cases_with_saved_graph']} |\n"
    report += f"| Non-empty dependency edges | {mechanism['cases_with_dependency_edges']} |\n"
    report += f"| Measured cue-removal/control contrast | {mechanism['cases_with_measured_interventions']} |\n"
    report += f"| Verification rejected at least one selected claim | {mechanism['verification_calls_with_any_rejected_claim']} |\n"
    report += f"| Final answer differs from draft (not necessarily an improvement) | {mechanism['answers_changed_from_draft']} |\n\n"
    report += 'Metric denominators are explicit in metrics.json. False-abstention estimates concern only the single draft-labelled answerable case. See mechanism_audit.json and the separate LOCAL_DEMO_AUDIT.md counterexample before making claims about the mechanism.\n'
    if mechanism['attempted_cases'] == len(protocol['cases']) and not mechanism['cases_with_measured_interventions']:
        report += '\n**Important negative result:** zero cue-removal/control contrasts were activated on these 40 model-produced graphs. This study therefore does not test or establish the proposed advantage of controlled evidence removal. Dependency repair and verification may run, but that is a different claim.\n'
    report += '\nCompact exports: annotations_40.json, final_responses_40.json, and reference_similarity_40.json. These are evaluation artifacts and are never passed into inference.\n'
    report_path.write_text(report, encoding='utf-8')
