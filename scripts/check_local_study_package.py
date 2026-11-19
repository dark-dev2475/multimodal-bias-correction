"""Offline handoff checks: completion/integrity, not a scientific validity test."""
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_reference_study import read, digest


def check_bundle(bundle, protocol, study):
    expected = {c['example_id']: c for c in protocol['cases']}
    seen, statuses = set(), Counter()
    for example in bundle['examples']:
        cid = example['case']['example_id']
        if cid in seen or example['case'] != expected.get(cid):
            raise ValueError('Duplicate/unknown/mismatched bundle case')
        seen.add(cid)
        if set(example['outputs']) != set(protocol['systems']):
            raise ValueError('A planned method attempt is missing from the bundle')
        for system, result in example['outputs'].items():
            if result != read(study / 'responses' / cid / (system + '.json')):
                raise ValueError('Bundle differs from original saved output')
            if result['example_id'] != cid or result['system'] != system or result['protocol_hash'] != protocol['protocol_hash']:
                raise ValueError('Output identity/protocol mismatch')
            statuses[result['status']] += 1
    if seen != set(expected):
        raise ValueError('A planned image is missing from the bundle')
    return dict(statuses)


def main():
    from scripts import run_local_graph_study as generator
    from scripts.run_bounded_local_reviews import review_protocol
    generator.configure()
    protocol = generator.runner.frozen(generator.OUT)
    review = review_protocol(generator.OUT, protocol)
    output = ROOT / 'output/version2/FINAL_40_IMAGE_STUDY'
    bundle = read(output / 'all_outputs.json')
    references = read(generator.OUT / 'frozen_references.json')
    if digest(references) != protocol['reference_hash']:
        raise ValueError('Reference content changed')
    statuses = check_bundle(bundle, protocol, generator.OUT)
    if any(e['reference'] != references[e['case']['example_id']] for e in bundle['examples']):
        raise ValueError('Bundled references differ from frozen references')
    metrics = read(output / 'metrics.json')
    if not metrics['collection_attempts_complete'] or not metrics['valid_outputs_reviewed_fully']:
        raise ValueError('Collection or valid-output review coverage is incomplete')
    rows = read(output / 'per_response_scores.json')
    expected_pairs = {(c['example_id'], s) for c in protocol['cases'] for s in protocol['systems']}
    if len(rows) != len(expected_pairs) or {(r['example_id'], r['system']) for r in rows} != expected_pairs:
        raise ValueError('Missing/duplicate planned score rows')
    if any(r['semantic'] is None for r in rows if r['status'] != 'failed'):
        raise ValueError('A valid output lacks an evaluation label')
    if any(r['semantic'] is not None for r in rows if r['status'] == 'failed'):
        raise ValueError('A failed attempt was credited with a semantic score')
    approved = sum(r['status'] == 'human_approved' for r in references.values())
    check = {'checked_at': datetime.now(timezone.utc).isoformat(),
             'automatic_collection_and_label_coverage_complete': True,
             'cases': len(protocol['cases']), 'planned_method_image_pairs': len(expected_pairs),
             'status_counts': statuses, 'valid_outputs': sum(r['status'] != 'failed' for r in rows),
             'failed_attempts_preserved': sum(r['status'] == 'failed' for r in rows),
             'human_approved_references': approved, 'human_reference_review_complete': approved == len(references),
             'semantic_scores_validated': False, 'known_judge_errors': True,
             'published_scores_reproduced': False, 'superiority_established': False,
             'generator_protocol_hash': protocol['protocol_hash'],
             'review_protocol_hash': review['review_protocol_hash'],
             'source_and_image_freeze_verified': True, 'bundle_matches_saved_outputs': True,
             'warning': 'Technical completion is NOT human review, reliable semantic evaluation, or proof of bias correction.'}
    (output / 'completion_check.json').write_text(json.dumps(check, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(check, indent=2))


if __name__ == '__main__':
    main()
