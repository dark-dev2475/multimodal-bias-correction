"""Re-annotate EVERY valid output with a uniform, separately frozen format policy.

Preliminary AI reviews/errors stay untouched. Labels remain same-model AI estimates.
No method output is changed and no semantic label is selected for a preferred score.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts import run_local_graph_study as graph_study
from scripts.run_reference_study import read, save, now, digest, task, make_units, Reference, SemanticReview, JUDGE, parse_model_json
from local_review_v2.backend import BoundedReviewBackend, LocalInfrastructureError


def review_protocol(out, generator):
    files = [Path(__file__), ROOT / 'src/local_review_v2/backend.py', ROOT / 'scripts/run_reference_study.py',
             ROOT / 'src/local_experiment/backend.py', ROOT / 'src/evaluation/reference_protocol.py']
    protocol = {'version': 'bounded-local-review-v2', 'generator_protocol_hash': generator['protocol_hash'],
        'model': generator['model'], 'installation': generator['installation'], 'server': generator['server'],
        'prompt_hash': digest(JUDGE), 'source_hashes': {str(p.relative_to(ROOT)): digest(p.read_text(encoding='utf-8')) for p in files},
        'format_rules': 'Fixed unit IDs/order; explanations <=240 characters per unit and <=400 overall; required facts must be a unique subset of permitted IDs. Semantic decisions remain free.',
        'scope': 'Every valid output, all seven systems. Failed inference outputs remain failures. Identical case/answer strings share one blind review.',
        'development_disclosure': 'Separate format revision after preliminary annotation outputs exhausted 3000 tokens. All preliminary labels/errors retained; this pass rejudges all valid outputs, not only unfavorable labels.',
        'independent_judge': False, 'human_reviewed': False, 'temperature': 0, 'seed': 42, 'max_output_tokens': 3000}
    protocol['review_protocol_hash'] = digest(protocol)
    path = out / 'bounded_review_protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Bounded review protocol changed; create a new evaluation version')
    if not path.exists():
        save(path, protocol)
    return protocol


def run(out, limit):
    graph_study.configure()
    p = graph_study.runner.frozen(out)
    rp = review_protocol(out, p)
    references = read(out / 'frozen_references.json')
    if digest(references) != p['reference_hash']:
        raise ValueError('Reference integrity failure')
    backend = BoundedReviewBackend(ROOT / 'data/cache/bounded_local_review_v2')
    count = 0
    for case in p['cases']:
        ref = Reference.model_validate(references[case['example_id']]['reference'])
        for system in sorted(p['systems'], key=lambda s: digest([case['example_id'], s])):
            path = out / 'responses' / case['example_id'] / (system + '.json')
            if not path.exists():
                continue
            result = read(path)
            if result['protocol_hash'] != p['protocol_hash']:
                raise ValueError('Output protocol mismatch')
            if result['status'] == 'failed' or not result.get('final_response'):
                continue
            answer = result['final_response']
            token = digest([p['protocol_hash'], case['example_id'], answer])[:24]
            target = out / 'ai_reviews_v2' / (token + '.json')
            error = out / 'review_errors_v2' / (token + '.json')
            if target.exists() or error.exists() or count >= limit:
                continue
            units = make_units(answer)
            form = {'review_id': token, **task(case), 'image_path': case['image_path'], 'reference': ref.model_dump(),
                'answer': answer, 'units': units, 'labels': None, 'reviewer': None, 'reviewed_at': None,
                'annotation_origin': 'pending_human_review'}
            blind = out / 'blind_review' / (token + '.json')
            if not blind.exists():
                save(blind, form)
            prompt_data = {**task(case), 'reference': ref.model_dump(), 'answer': answer, 'units': units}
            prompt = JUDGE + '\nSCHEMA:\n' + json.dumps(SemanticReview.model_json_schema()) + '\nINPUT DATA:\n' + json.dumps(prompt_data)
            start = len(backend.calls)
            try:
                response = backend.complete(prompt, case['image_path'])
                labels = SemanticReview.model_validate(parse_model_json(response, 'bounded_review')).validate_coverage(units, ref)
                save(target, {**form, 'labels': labels.model_dump(), 'reviewer': p['model'], 'reviewed_at': now(),
                    'annotation_origin': 'ai_assisted', 'warning': 'Same-model AI judgment, not independent or human validation.',
                    'review_protocol_hash': rp['review_protocol_hash'], 'calls': backend.calls[start:]})
                print('Bounded AI review: ' + token, flush=True)
            except LocalInfrastructureError:
                raise
            except Exception as exc:
                save(error, {'review_id': token, 'review_protocol_hash': rp['review_protocol_hash'], 'error_type': type(exc).__name__,
                    'message': str(exc)[:500], 'at': now(), 'calls': backend.calls[start:]})
                print('Bounded review failed: ' + token, flush=True)
            count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=280)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('limit must be positive')
    try:
        run(graph_study.OUT, args.limit)
    except LocalInfrastructureError as exc:
        save(graph_study.OUT / 'BOUNDED_REVIEW_PAUSED.json', {'at': now(), 'reason': str(exc)})
        print(str(exc), flush=True)
        raise SystemExit(2)
