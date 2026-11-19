"""Post-hoc independent TEXT diagnostic for all saved outputs; never inference.

References remain unapproved. Non-entailment is not necessarily bias or error,
especially for uncertainty answers. This script does not inspect the images.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts.evaluate_reference_study import read, digest, make_units
from scripts.run_reference_study import save
from scripts.run_validation_calibration import freeze, STUDY, OUTPUT
from validation_v2.calibration import summarize
from validation_v2.nli import TextNLI

WARNING = ('Independent-model TEXT entailment against AI-drafted reference facts. '
           'Not image-grounded accuracy, fairness, or a replacement for human labels. '
           'Non-entailment can reflect incomplete references or justified uncertainty.')


def tasks_for(answer, reference):
    premise = '\n'.join(f['text'] for f in reference['facts'])
    units = [{'unit_id': uid, 'text': text, 'premise': premise, 'hypothesis': text}
             for uid, text in make_units(answer).items()]
    required = set(reference['required_fact_ids'])
    facts = [{'fact_id': f['id'], 'text': f['text'], 'premise': answer, 'hypothesis': f['text']}
             for f in reference['facts'] if f['id'] in required]
    return units, facts


def complete_fraction(judgments):
    if not judgments or any(j['status'] != 'completed' for j in judgments):
        return None
    return sum(j['prediction'] == 'entailed' for j in judgments) / len(judgments)


def main():
    calibration = freeze()
    nli_rows = [read(p) for p in (STUDY / 'calibration_results/nli_deberta').glob('*.json')]
    if any(r['protocol_hash'] != calibration['protocol_hash'] for r in nli_rows):
        raise ValueError('Calibration result protocol mismatch')
    if not summarize(nli_rows, calibration['cases'])['gate_passed']:
        raise ValueError('Independent checker has not passed the frozen text gate')
    bundle_path = ROOT / 'output/version2/FINAL_40_IMAGE_STUDY/all_outputs.json'
    bundle = read(bundle_path)
    sources = [Path(__file__), ROOT / 'src/validation_v2/nli.py']
    protocol = {'version': 'reference-text-diagnostic-round1', 'warning': WARNING,
                'calibration_protocol_hash': calibration['protocol_hash'], 'source_bundle_hash': digest(bundle),
                'model_installation': calibration['nli_installation'], 'post_hoc': True,
                'uses_images': False, 'reference_answer_used_as_evidence': False,
                'directions': {'unit_support': 'reference facts -> answer sentence',
                               'required_fact_coverage': 'whole answer -> required reference fact'},
                'source_hashes': {str(p.relative_to(ROOT)): digest(p.read_text(encoding='utf-8')) for p in sources}}
    protocol['protocol_hash'] = digest(protocol)
    protocol_path = STUDY / 'reference_text_protocol.json'
    if protocol_path.exists() and read(protocol_path) != protocol:
        raise ValueError('Text diagnostic changed after freeze; create a new version')
    if not protocol_path.exists():
        save(protocol_path, protocol)
    backend = None

    def judge(task):
        nonlocal backend
        pair = {k: task[k] for k in ['premise', 'hypothesis']}
        key = digest([protocol['protocol_hash'], pair])
        cache_path = ROOT / 'data/cache/validation_round1_nli_pairs' / (key + '.json')
        if cache_path.exists():
            result = read(cache_path)
            if result['pair_hash'] != key:
                raise ValueError('NLI pair-cache identity mismatch')
        else:
            if backend is None:
                backend = TextNLI()
            try:
                result = {'status': 'completed', **backend.predict(**pair)}
            except ValueError as exc:
                result = {'status': 'failed', 'prediction': None, 'error': str(exc)}
            result.update(pair_hash=key)
            save(cache_path, result)
        return {k:v for k,v in task.items() if k not in {'premise', 'hypothesis'}} | result

    rows = []
    for example in bundle['examples']:
        cid = example['case']['example_id']
        reference = example['reference']['reference']
        for system, result in example['outputs'].items():
            row = {'example_id': cid, 'system': system, 'inference_status': result['status'],
                   'reference_status': example['reference']['status'],
                   'answer': result.get('final_response'), 'unit_judgments': [], 'fact_judgments': [],
                   'reference_entailed_unit_fraction': None, 'answer_entailed_required_fact_fraction': None}
            if result['status'] != 'failed' and result.get('final_response'):
                unit_tasks, fact_tasks = tasks_for(result['final_response'], reference)
                row['unit_judgments'] = [judge(t) for t in unit_tasks]
                row['fact_judgments'] = [judge(t) for t in fact_tasks]
                row['reference_entailed_unit_fraction'] = complete_fraction(row['unit_judgments'])
                row['answer_entailed_required_fact_fraction'] = complete_fraction(row['fact_judgments'])
            rows.append(row)
        print('Reference-text diagnostic: ' + cid, flush=True)
    by_system = defaultdict(list)
    for row in rows:
        by_system[row['system']].append(row)
    summary = {}
    for system, selected in by_system.items():
        summary[system] = {'planned': len(selected), 'valid_inference_outputs': sum(r['inference_status'] != 'failed' for r in selected),
                           'failed_inference_attempts': sum(r['inference_status'] == 'failed' for r in selected)}
        for metric in ['reference_entailed_unit_fraction', 'answer_entailed_required_fact_fraction']:
            values = [r[metric] for r in selected if r[metric] is not None]
            summary[system][metric] = {'mean_available_only': mean(values) if values else None, 'denominator': len(values)}
    payload = {'protocol': protocol, 'systems': dict(summary), 'results': rows}
    save(OUTPUT / 'independent_reference_text_all_methods.json', payload)
    save(OUTPUT / 'independent_reference_text_40.json', {**payload, 'results': [r for r in rows if r['system'] == 'eigr']})
    lines = ['# Independent reference-text diagnostic', '', WARNING, '',
             'The NLI model passed the frozen binary support gate on 24 synthetic text cases. '
             'This does not validate it on images or real social-bias decisions. The original model labels '
             'and scores are unchanged. This report is a separate post-hoc diagnostic.', '',
             '| Method | Valid outputs / 40 | Reference-entailing units | Answer-entailing required facts |',
             '|---|---:|---:|---:|']
    def fmt(value):
        return 'N/A' if value['mean_available_only'] is None else f"{value['mean_available_only']:.3f} (n={value['denominator']})"
    for system, info in summary.items():
        lines.append(f"| {system} | {info['valid_inference_outputs']}/40 | {fmt(info['reference_entailed_unit_fraction'])} | {fmt(info['answer_entailed_required_fact_fraction'])} |")
    lines += ['', 'These are fractions of text-entailment judgments, NOT task accuracy or bias rates. '
              'An uncertainty answer can legitimately be non-entailed by a short fact list. Required facts '
              'may be over-inclusive in the unapproved reference drafts. Missing or overlength pairs are '
              'unscored, never silently truncated. Different denominators must be considered.', '',
              'Useful-fact coverage is directional: the answer must express the fact; merely placing a '
              'fact in the reference is not evidence that the answer contains it. The inverse direction '
              'checks support for an answer sentence. Neither direction proves visual truth.']
    (OUTPUT / 'INDEPENDENT_TEXT_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
