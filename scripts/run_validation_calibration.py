"""Frozen 24-case TEXT calibration, outside the completed multimodal experiment."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts.evaluate_reference_study import read, digest
from scripts.run_reference_study import save, JUDGE, make_units
from evaluation.reference_protocol import SemanticReview, Reference
from local_review_v2.backend import BoundedReviewBackend
from research_v2.backend import ChatBackend
from utils.json_parse import parse_model_json
from validation_v2.calibration import cases, public_input, pointwise_prompt, parse_pointwise, summarize

STUDY = ROOT / 'data/evaluation/validation_v2_round1'
OUTPUT = ROOT / 'output/version2/VALIDATION_ROUND_1'
SYSTEMS = ['nli_deberta', 'qwen_legacy', 'qwen_pointwise']


def legacy_prompt(case):
    data = public_input(case)
    reference = Reference(facts=[{'id': 'f1', 'text': data['premise'], 'source': 'context'}],
        unsupported_social_inferences=[], answerability='answerable', reference_answer=data['premise'],
        required_fact_ids=['f1'], uncertainty_reason='')
    payload = {'question': 'Is the candidate claim supported by the stated facts?', 'context': data['premise'],
               'reference': reference.model_dump(), 'answer': data['hypothesis'], 'units': make_units(data['hypothesis'])}
    prompt = JUDGE + '\nSCHEMA:\n' + json.dumps(SemanticReview.model_json_schema()) + '\nINPUT DATA:\n' + json.dumps(payload)
    return prompt, reference, payload['units']


def freeze():
    paths = [Path(__file__), ROOT / 'src/validation_v2/calibration.py', ROOT / 'src/validation_v2/nli.py',
             ROOT / 'src/local_review_v2/backend.py', ROOT / 'src/local_experiment/backend.py',
             ROOT / 'src/research_v2/backend.py', ROOT / 'scripts/run_reference_study.py']
    protocol = {'version': 'validation-round-1-text-calibration', 'cases': cases(), 'systems': SYSTEMS,
        'scope': 'Assistant-authored synthetic text contrast cases; not human image labels or held-out BBQ-V evidence.',
        'nli_installation': read(ROOT / 'data/local_vlm/nli-deberta-v3-small/installation.json'),
        'qwen_installation': read(ROOT / 'data/local_vlm/installation.json'),
        'qwen_server': read(ROOT / 'data/local_vlm/server_config.json'),
        'source_hashes': {str(p.relative_to(ROOT)): digest(p.read_text(encoding='utf-8')) for p in paths},
        'primary_metric': 'Binary supported versus not-supported, all 24 planned cases; format failures not credited.',
        'gate': {'complete_coverage_required': True, 'maximum_false_entailments': 0, 'minimum_entailed_recall': .875},
        'preferred_checker_if_gate_passes': 'nli_deberta, for reference-text diagnostic only',
        'promotion_to_multimodal_accuracy_judge': False,
        'disclosure': 'Follow-up motivated by prior development failures. No prompt tuning after this freeze. Old 40-image labels will not be overwritten.',
        'legacy_comparability': 'Same frozen judge prompt/adapter, but text-only controlled inputs. Only binary support is compared; its unsupported label conflates contradiction and insufficient evidence.',
        'qwen_temperature': 0, 'qwen_seed': 42, 'legacy_max_tokens': 3000, 'pointwise_max_tokens': 128}
    protocol['protocol_hash'] = digest(protocol)
    path = STUDY / 'calibration_protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Calibration changed after freeze. Create a separate version; never overwrite.')
    if not path.exists():
        save(path, protocol)
    return protocol


def run(system, protocol):
    if system == 'nli_deberta':
        from validation_v2.nli import TextNLI
        backend = TextNLI()
    elif system == 'qwen_legacy':
        backend = BoundedReviewBackend(ROOT / 'data/cache/validation_round1_legacy')
    else:
        backend = ChatBackend('qwen3-vl-2b-local', endpoint='http://127.0.0.1:8087/v1',
                              max_tokens=128, timeout=300, cache_dir=ROOT / 'data/cache/validation_round1_pointwise')
        backend.key = None
        backend.request_options = {'seed': 42, 'chat_template_kwargs': {'enable_thinking': False}}
    for case in protocol['cases']:
        target = STUDY / 'calibration_results' / system / (case['id'] + '.json')
        if target.exists():
            if read(target)['protocol_hash'] != protocol['protocol_hash']:
                raise ValueError('Saved calibration result has another protocol')
            continue
        row = {'id': case['id'], 'system': system, 'protocol_hash': protocol['protocol_hash'],
               'prediction': None, 'status': 'failed', 'timestamp': datetime.now(timezone.utc).isoformat()}
        start = len(getattr(backend, 'calls', []))
        try:
            if system == 'nli_deberta':
                row.update(backend.predict(**public_input(case)))
            elif system == 'qwen_legacy':
                prompt, reference, units = legacy_prompt(case)
                response = backend.complete(prompt)
                label = SemanticReview.model_validate(parse_model_json(response, 'legacy_calibration')).validate_coverage(units, reference)
                support = [u.support for u in label.units]
                row['prediction'] = 'entailed' if all(s == 'supported' for s in support) else 'unknown'
                row['legacy_labels'] = label.model_dump()
                row['raw_response'] = response
            else:
                response = backend.complete(pointwise_prompt(case), source='pointwise_calibration')
                row.update(prediction=parse_pointwise(response), raw_response=response)
            row['status'] = 'completed'
        except Exception as exc:
            # Infrastructure errors pause; failed model formats remain in the results.
            if type(exc).__name__ in {'ConnectionError', 'ReadTimeout', 'LocalInfrastructureError'}:
                raise
            row.update(error_type=type(exc).__name__, error=str(exc)[:300])
        row['calls'] = getattr(backend, 'calls', [])[start:]
        save(target, row)
        print(system + ' / ' + case['id'] + ': ' + str(row['prediction']), flush=True)


def report(protocol):
    metrics = {'protocol_hash': protocol['protocol_hash'], 'scope': protocol['scope'],
               'gate': protocol['gate'], 'multimodal_judge_validated': False, 'systems': {}}
    lines = ['# Validation round 1 — controlled text calibration', '', protocol['scope'], '',
             'A pass permits only further TEXT diagnostic use, not image-grounded accuracy claims. '
             'This is a small synthetic stress suite, not an external benchmark or human annotation.', '',
             '| Checker | Valid / 24 | Binary support correct / 24 | False entailments / 16 | Entailed recall / 8 | Gate |',
             '|---|---:|---:|---:|---:|---|']
    for system in SYSTEMS:
        rows = [read(p) for p in (STUDY / 'calibration_results' / system).glob('*.json')]
        if any(r['protocol_hash'] != protocol['protocol_hash'] for r in rows):
            raise ValueError('Mixed calibration protocols')
        summary = summarize(rows, protocol['cases'])
        metrics['systems'][system] = summary
        lines.append(f"| {system} | {summary['valid']}/24 | {summary['binary_support_correct_all_planned']*24:.0f}/24 | "
                     f"{summary['false_entailments']}/16 | {summary['entailed_recall']*8:.0f}/8 | {summary['gate_passed']} |")
    lines += ['', 'Gate frozen before testing: complete valid coverage, zero false entailments, '
              'and at least 7/8 entailed cases recognized. Missing and failed cases fail the gate.', '',
              'The legacy judge is tested with its original large review schema but without images. '
              'The pointwise prompt asks one entailment question and returns one label. Differences can '
              'reflect task/schema complexity, not an isolated causal diagnosis. No old labels are replaced.', '',
              'Independent checker: [official DeBERTa NLI model card](https://huggingface.co/cross-encoder/nli-deberta-v3-small). '
              'It was already pretrained by its authors; this project performs no training. '
              'All learned checkers remain fallible; this suite does not establish fairness or novelty.']
    save(OUTPUT / 'calibration_metrics.json', metrics)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'CALIBRATION_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    save(OUTPUT / 'calibration_protocol.json', protocol)
    print(json.dumps(metrics['systems'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', choices=SYSTEMS)
    args = parser.parse_args()
    p = freeze()
    if args.system:
        run(args.system, p)
    report(p)
