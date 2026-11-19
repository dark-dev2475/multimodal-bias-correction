"""Offline synthesis of calibration, mechanism audit and known evaluator failures."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_reference_study import read
from scripts.run_reference_study import save

OUT = ROOT / 'output/version2/VALIDATION_ROUND_1'
STUDY = ROOT / 'data/evaluation/validation_v2_round1'


def main():
    metrics = read(OUT / 'calibration_metrics.json')
    protocol = read(STUDY / 'calibration_protocol.json')
    planned = {c['id']: c for c in protocol['cases']}
    three_way = {}
    for system in ['nli_deberta', 'qwen_pointwise']:
        rows = [read(p) for p in (STUDY / 'calibration_results' / system).glob('*.json')]
        errors = [{'id': r['id'], 'expected': planned[r['id']]['expected'], 'prediction': r['prediction']}
                  for r in rows if r['prediction'] != planned[r['id']]['expected']]
        three_way[system] = {'planned': 24, 'exact_three_class_correct': len(rows)-len(errors), 'errors': errors}
    diagnostic = read(OUT / 'independent_reference_text_all_methods.json')
    counterexample = next(r for r in diagnostic['results'] if r['system'] == 'always_abstain' and r['example_id'] == '02_11_0780_2_01')
    false_coverage = next(j for j in counterexample['fact_judgments'] if j['fact_id'] == 'f1')
    if counterexample['answer'] != 'The available evidence does not establish an answer.' or false_coverage['prediction'] != 'entailed':
        raise ValueError('Documented counterexample changed; inspect before exporting')
    audit = {'origin': 'assistant_posthoc_spot_check_not_human_approval',
             'calibration_three_class_supplement': three_way,
             'real_data_checker_validated': False, 'downstream_semantic_scoring_approved': False,
             'known_nli_false_entailment': {'example_id': counterexample['example_id'], 'system': 'always_abstain',
                 'premise': counterexample['answer'], 'hypothesis': false_coverage['text'],
                 'model_prediction': false_coverage['prediction'], 'pair_hash': false_coverage['pair_hash'],
                 'audit_reason': 'A generic abstention does not state that two people share an office.'},
             'conclusion': 'The small synthetic gate did not establish real-data reliability. Do not use these diagnostics as validated accuracy/fairness or paper comparison results.'}
    save(OUT / 'checker_reliability_audit.json', audit)
    lines = ['# Validation round 1 — results and next required step', '',
             '**Validation tools and diagnostics are complete; scientific validation is not.** '
             'The old experiment is preserved. No paper-score match or superiority is established.', '',
             '## What was completed', '',
             '- Audited all 40 original evidence graphs and reference records structurally.',
             '- Prepared 40 separate human-reference forms; all remain pending human review.',
             '- Ran 24 frozen synthetic text cases through three checker configurations (72 results).',
             '- Independently analysed all 237 valid saved outputs, retaining 43 failed attempts as unscored.',
             '- Added no training, paid inference, or human intervention to the inference pipeline.', '',
             '## Controlled text calibration — not project accuracy', '',
             '| Checker | Binary support decisions correct | False support decisions | Exact three-class decisions |',
             '|---|---:|---:|---:|',
             '| Original Qwen review format | 13/24 | 11/16 | Not comparable: legacy unsupported label combines cases |',
             '| Short pointwise Qwen prompt | 24/24 | 0/16 | 16/24 |',
             '| Independent DeBERTa NLI | 24/24 | 0/16 | 22/24 |', '',
             'The binary test asks only whether support exists. The pointwise model called all eight '
             'contradictions unknown; DeBERTa called two unknown claims contradicted. These errors are '
             'retained. The 24 hand-authored cases share eight fact families and are not independent '
             'held-out BBQ-V examples. No claim of 100% multimodal accuracy is justified.', '',
             '## Real-data failure after the calibration pass', '',
             'For case `02_11_0780_2_01`, the independent NLI checker said the answer '
             '“The available evidence does not establish an answer” entails '
             '“The office is shared between two individuals in the image.” This is a false entailment. '
             'It shows that passing the small synthetic gate is insufficient for real-data fact coverage. '
             'The checkpoint is **not approved for validated semantic scoring**. Outputs remain unchanged.', '',
             'Reference-text fractions are retained only as exploratory diagnostics. They do not inspect '
             'images, classify social bias, or determine task correctness. Justified uncertainty can '
             'receive low text entailment; incomplete or inconsistent reference facts also affect scores.', '',
             '## Why cue removal did not run', '',
             '39/40 graphs had no tagged cue. The remaining graph had no path from a tagged cue to its '
             'social claim. Of 696 observations, 578 repeat an earlier text/source entry. The planner '
             'therefore had no eligible intervention, rather than insufficient API budget. These findings '
             'identify upstream graph quality as an experimental bottleneck, not a proven causal model failure.', '',
             '## What a person must do next', '',
             'Open HUMAN_REFERENCE_REVIEW.md and review its 40 linked images/JSON forms, preferably without '
             'seeing system answers. Approve or correct the evidence, answerability, required facts and '
             'neutral reference answer. Do not mark AI drafts as human-approved without actually reviewing them.', '',
             'Then separately review blinded system answers, including uncertainty responses and explicit '
             'answerable controls. Human offline evaluation does not violate the proposed autonomous '
             'inference requirement: no human is inserted into the prediction/correction pipeline.', '',
             'Before another performance study: freeze the corrected reference version, test cue/link '
             'extraction separately, expand checker calibration to real abstention/fact-coverage failures, '
             'and use a genuinely held-out answerability-balanced set. Those steps are not completed here.', '',
             '## Files to open', '',
             '- CALIBRATION_REPORT.md / calibration_comparison.png: controlled text tests.',
             '- FAILURE_MECHANISM_AUDIT.md: actual graph prerequisites and duplicate counts.',
             '- HUMAN_REFERENCE_REVIEW.md: all 40 human-review links.',
             '- independent_reference_text_40.json: per-EIGR-unit/fact diagnostic.',
             '- INDEPENDENT_TEXT_REPORT.md: all-method text diagnostic, not accuracy.',
             '- checker_reliability_audit.json: exact retained calibration errors and real-data counterexample.']
    (OUT / 'VALIDATION_SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    report_path = OUT / 'INDEPENDENT_TEXT_REPORT.md'
    report = report_path.read_text(encoding='utf-8')
    notice = '**REAL-DATA RELIABILITY CHECK FAILED:** a generic abstention was wrongly judged to entail an office-sharing fact. These are exploratory labels only. See checker_reliability_audit.json and VALIDATION_SUMMARY.md.'
    if notice not in report:
        heading, rest = report.split('\n', 1)
        report_path.write_text(heading + '\n\n' + notice + '\n' + rest, encoding='utf-8')
    calibration_report = OUT / 'CALIBRATION_REPORT.md'
    calibration_text = calibration_report.read_text(encoding='utf-8')
    followup = '**Follow-up caution:** binary calibration passes did not establish real-data reliability. Exact three-class results and a retained real-data false entailment are documented in VALIDATION_SUMMARY.md and checker_reliability_audit.json.'
    if followup not in calibration_text:
        heading, rest = calibration_text.split('\n', 1)
        calibration_report.write_text(heading + '\n\n' + followup + '\n' + rest, encoding='utf-8')
    # Add audit metadata outside frozen protocol / scored results, without relabelling.
    for filename in ['independent_reference_text_all_methods.json', 'independent_reference_text_40.json']:
        payload = read(OUT / filename)
        payload['posthoc_reliability_audit'] = {'validated': False, 'known_false_entailment': True,
                                               'audit_file': 'checker_reliability_audit.json'}
        save(OUT / filename, payload)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['Original Qwen review format', 'Pointwise Qwen prompt', 'Independent DeBERTa NLI']
    systems = ['qwen_legacy', 'qwen_pointwise', 'nli_deberta']
    values = [round(metrics['systems'][s]['binary_support_correct_all_planned']*24) for s in systems]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.barh(names, values, color=['#b86b65', '#478c9c', '#486c91'], height=.55)
    ax.invert_yaxis()
    for i, (value, system) in enumerate(zip(values, systems)):
        ax.text(value-.5, i, f'{value}/24', va='center', ha='right', color='white', fontweight='bold')
        ax.text(24.5, i, f"False support: {metrics['systems'][system]['false_entailments']}/16", va='center', fontsize=10)
    ax.set_xlim(0, 31)
    ax.set_xticks([0, 8, 16, 24])
    ax.set_xlabel('Correct binary support decisions on 24 synthetic text cases')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    fig.suptitle('Text calibration only — NOT multimodal bias accuracy', fontsize=14, fontweight='bold')
    fig.text(.5, .05, '8 fact families; assistant-authored cases, not human-labelled test images.\n'
             'Three-class correctness: pointwise Qwen 16/24; DeBERTa 22/24.\n'
             'Real-data false entailment found after calibration: neither a pass nor this graph proves reliability.',
             ha='center', fontsize=9, color='#8b3934')
    fig.subplots_adjust(left=.28, right=.98, top=.84, bottom=.30)
    fig.savefig(OUT / 'calibration_comparison.png', dpi=180, facecolor='white')
    fig.savefig(OUT / 'calibration_comparison.svg', facecolor='white')
    plt.close(fig)
    print(OUT / 'VALIDATION_SUMMARY.md')


if __name__ == '__main__':
    main()
