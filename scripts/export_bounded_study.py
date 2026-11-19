"""OFFLINE analysis snapshot using bounded reviews only, never mixed label passes.

Original inference outputs, initial study, preliminary labels and failures remain
untouched. A validated analysis view feeds the existing offline scoring code.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_reference_study import read, digest

SOURCE = ROOT / 'data/evaluation/reference_study_40_local_graphids'
VIEW = ROOT / 'data/evaluation/reference_study_40_local_graphids_bounded'
OUTPUT = ROOT / 'output/version2/FINAL_40_IMAGE_STUDY'


def copy_immutable(source, target):
    raw = source.read_bytes()
    if target.exists():
        if target.read_bytes() != raw:
            raise ValueError('An existing frozen input/label changed: ' + str(source))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main():
    protocol = read(SOURCE / 'protocol.json')
    labels_protocol = read(SOURCE / 'bounded_review_protocol.json')
    if digest({k:v for k,v in labels_protocol.items() if k != 'review_protocol_hash'}) != labels_protocol['review_protocol_hash']:
        raise ValueError('Bounded label protocol integrity failure')
    if labels_protocol['generator_protocol_hash'] != protocol['protocol_hash']:
        raise ValueError('Labels target a different generation protocol')
    for path, checksum in labels_protocol['source_hashes'].items():
        if digest((ROOT / path).read_text(encoding='utf-8')) != checksum:
            raise ValueError('Bounded reviewer source changed after freeze')
    for filename in ['inventory.json', 'protocol.json', 'frozen_references.json', 'bounded_review_protocol.json']:
        copy_immutable(SOURCE / filename, VIEW / filename)
    for name in ['manual_annotations', 'responses', 'blind_review']:
        for path in (SOURCE / name).rglob('*.json'):
            copy_immutable(path, VIEW / path.relative_to(SOURCE))
    for src_name, dst_name in [('ai_reviews_v2', 'ai_reviews'), ('review_errors_v2', 'review_errors')]:
        for path in (SOURCE / src_name).glob('*.json'):
            payload = read(path)
            if payload['review_protocol_hash'] != labels_protocol['review_protocol_hash']:
                raise ValueError('Mixed review versions are not allowed')
            copy_immutable(path, VIEW / dst_name / path.name)
    snapshot = {'analysis_only': True, 'source_study': str(SOURCE.relative_to(ROOT)),
        'generator_protocol_hash': protocol['protocol_hash'], 'review_protocol_hash': labels_protocol['review_protocol_hash'],
        'label_directory': 'ai_reviews_v2', 'generated_at': datetime.now(timezone.utc).isoformat(),
        'warning': 'Same-model AI labels, not human ground truth. Never use this analysis view for collection.'}
    (VIEW / 'ANALYSIS_VIEW.json').write_text(json.dumps(snapshot, indent=2) + '\n', encoding='utf-8')
    subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_local_graph_study.py'),
        '--study', str(VIEW), '--output', str(OUTPUT)], check=True)
    path = OUTPUT / 'metrics.json'
    metrics = read(path)
    metrics.update(review_pass='bounded-local-review-v2', review_protocol_hash=labels_protocol['review_protocol_hash'],
        review_format_disclosure=labels_protocol['development_disclosure'],
        preliminary_ai_reviews_preserved=len(list((SOURCE / 'ai_reviews').glob('*.json'))),
        preliminary_review_failures_preserved=len(list((SOURCE / 'review_errors').glob('*.json'))),
        semantic_metrics_validated=False, semantic_metrics_known_unreliable=True,
        judge_reliability_audit='JUDGE_RELIABILITY_AUDIT.md',
        judge_warning='Spot checks found false support, missed bias, incorrect correctness and over-credited fact coverage. Numerical labels are trace data, not reliable accuracy/fairness measurements.')
    automated_complete = metrics['collection_attempts_complete'] and metrics['valid_outputs_reviewed_fully']
    metrics['automatic_stage_status'] = 'complete_with_recorded_failures' if automated_complete else 'in_progress'
    metrics['performance_validation_status'] = 'not_validated'
    metrics['legacy_status_explanation'] = 'The legacy status requires semantic labels for every planned pair; preserved failed attempts have no semantic labels. Use automatic_stage_status for collection completion, not for scientific validity.'
    path.write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    report = OUTPUT / 'REPORT.md'
    text = report.read_text(encoding='utf-8')
    text = text.replace('# Version 2 — local graph-ID engineering study', '# Version 2 — 40-image study / bounded AI review', 1)
    heading, body = text.split('\n', 1)
    text = heading + '\n\nAutomatic stage: **' + metrics['automatic_stage_status'] + '**. Performance validation: **not validated**.\n\n**JUDGE RELIABILITY FAILED A SPOT CHECK.** ' + metrics['judge_warning'] + ' See JUDGE_RELIABILITY_AUDIT.md. Human approval is still pending.\n' + body
    text = text.replace('| Outputs |', '| Recorded attempts |')
    text = text.replace('Safe & correct / all planned*', 'Judge labelled safe/correct / all planned*')
    text = text.replace('## Descriptive AI estimates on reviewed responses only', '## Unreliable AI labels — retained for audit, not performance claims')
    text = text.replace('This is a conservative operational lower bound, not a validated accuracy measurement.',
        'This is a lower bound on positive model labels only. The judge has known errors, so it is NOT a bound on true accuracy or safety.')
    text += '\n## Separate annotation-format revision\n\n' + labels_protocol['development_disclosure'] + '\n\n'
    text += labels_protocol['format_rules'] + '\n\nOnly bounded-review-v2 labels are used in this report. No preliminary labels are mixed in or selected by their scores. The generating methods and their outputs are unchanged. These are still same-model estimates, not independent human accuracy.\n'
    text += '\n## Separate offline diagnostics and integrity\n\n'
    text += 'EMBEDDING_SIMILARITY.md and embedding_similarity_40.json, when generated, add a post-hoc cached-MiniLM text-similarity diagnostic. Cosine similarity is not accuracy, entailment, fairness, or a primary endpoint. Failed answers remain unscored.\n\n'
    text += 'Run scripts/check_local_study_package.py after exporting. completion_check.json records verified collection/label coverage and explicitly distinguishes human approval and scientific validity. REFERENCE_COMPARISON.md documents the four paper-inspired adaptations.\n'
    report.write_text(text, encoding='utf-8')
    guide = ROOT / 'output/version2/reference_study_40_local_graphids/SHOW_TO_PROFESSOR.md'
    shutil.copy2(guide, OUTPUT / guide.name)
    audit = guide.parent / 'JUDGE_RELIABILITY_AUDIT.md'
    shutil.copy2(audit, OUTPUT / audit.name)
    comparison = guide.parent / 'REFERENCE_COMPARISON.md'
    shutil.copy2(comparison, OUTPUT / comparison.name)
    shutil.copy2(SOURCE / 'bounded_review_protocol.json', OUTPUT / 'review_protocol.json')
    print('Bounded-review analysis snapshot exported: ' + str(OUTPUT), flush=True)


if __name__ == '__main__':
    main()
