"""Static scientific figure from saved metrics only; no inference or rejudging."""
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/version2/reference_study_40_local'


def main(output=OUT):
    global OUT
    OUT = output
    metrics = json.loads((OUT / 'metrics.json').read_text(encoding='utf-8'))
    rows = json.loads((OUT / 'per_response_scores.json').read_text(encoding='utf-8'))
    names = {'raw': 'Raw model', 'eigr': 'EIGR V2 (local)', 'self_refine_vqa': 'Self-Refine adaptation',
             'intent_aware_vqa': 'Intent-Aware adaptation', 'reflexion_vqa': 'Reflexion adaptation',
             'constitutional_critique_vqa': 'Constitutional critique', 'always_abstain': 'Always abstain'}
    if metrics.get('engineering_followup'):
        names['eigr'] = 'EIGR V2 (ordered DAG)'
    systems = list(metrics['systems'])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), sharey=True)
    for i, system in enumerate(systems):
        summary = metrics['systems'][system]
        planned = summary['planned']
        missing = summary['status_counts'].get('not_collected', 0)
        failed = summary['status_counts'].get('failed', 0)
        valid = planned - missing - failed
        segments = [(valid, '#348c9c'), (failed, '#c36560'), (missing, '#d6d9df')]
        selected = [r for r in rows if r['system'] == system and r['semantic'] is not None]
        safe = sum(r['semantic']['safe_correct_rate'] for r in selected)
        reviewed = len(selected)
        review_segments = [(safe, '#538865'), (reviewed-safe, '#c8a064'), (planned-reviewed, '#d6d9df')]
        for axis, values in zip(axes, [segments, review_segments]):
            left = 0
            for count, color in values:
                axis.barh(i, count, left=left, color=color, height=.65)
                if count >= 3:
                    axis.text(left+count/2, i, str(int(count)), ha='center', va='center', fontsize=10)
                left += count
    axes[0].set_yticks(range(len(systems)), [names[s] for s in systems])
    axes[0].invert_yaxis()
    axes[0].set_title('Execution status — not accuracy (all 40 planned)')
    axes[1].set_title('Same-model AI judgments (not human validation)')
    if metrics.get('semantic_metrics_known_unreliable'):
        axes[1].set_title('Unreliable judge labels — spot-check failed', color='#9a3934')
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=c, label=l) for c, l in [('#348c9c', 'Valid output'), ('#c36560', 'Failed'), ('#d6d9df', 'Not collected')]],
                   loc='upper center', bbox_to_anchor=(.5, -.13), ncol=3, frameon=False, fontsize=8)
    axes[1].legend(handles=[Patch(color=c, label=l) for c, l in [('#538865', 'Labelled safe/correct'), ('#c8a064', 'Other reviewed'), ('#d6d9df', 'Unreviewed / failed')]],
                   loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2, frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xlim(0, metrics['planned_examples'])
        axis.set_xlabel('Number of image responses')
        axis.spines[['top', 'right', 'left']].set_visible(False)
        axis.tick_params(axis='y', length=0)
        axis.set_axisbelow(True)
        axis.grid(axis='x', alpha=.16)
    fig.suptitle('EIGR V2 — free local 40-image development study', fontsize=15, fontweight='bold', y=.98)
    if metrics.get('engineering_followup'):
        fig.text(.5, .925, 'Ordered-DAG engineering follow-up: development-informed changes, not held-out validation',
                 ha='center', fontsize=10, color='#8a562b')
    warning = 'Judge errors documented: right panel is NOT accuracy. See JUDGE_RELIABILITY_AUDIT.md.' if metrics.get('semantic_metrics_known_unreliable') else 'Failures and missing labels are retained.'
    fig.text(.5, .015, 'AI-drafted references: human review pending. 39 ambiguous / 1 answerable draft labels.\n'
             'Paper-inspired controls, not published-score reproductions.\n' + warning,
             ha='center', fontsize=9, color='#444444')
    fig.subplots_adjust(left=.20, right=.98, bottom=.25, top=.86, wspace=.16)
    fig.savefig(OUT / 'comparison_coverage.png', dpi=180, facecolor='white')
    fig.savefig(OUT / 'comparison_coverage.svg', facecolor='white')
    plt.close(fig)
    print(OUT / 'comparison_coverage.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    main(parser.parse_args().output)
