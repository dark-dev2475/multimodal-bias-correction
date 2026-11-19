"""Versioned engineering follow-up after graph-link failures on DEVELOPMENT data.

Not a held-out hypothesis test; original local failures remain unchanged.
Same model, references, methods, budgets, metrics; deterministic claim IDs.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts import run_local_reference_study as runner
from local_graph_experiment.backend import GraphIDBackend

OUT = ROOT / 'data/evaluation/reference_study_40_local_graphids'
original_sources = runner.local_sources


def graph_sources():
    values = original_sources()
    paths = [Path(__file__)] + list((ROOT / 'src/local_graph_experiment').glob('*.py'))
    values.update({str(p.relative_to(ROOT)): runner.digest(p.read_text(encoding='utf-8')) for p in paths})
    return values


def configure():
    runner.LocalBackend = GraphIDBackend
    runner.local_sources = graph_sources


def initialize(out):
    if (out / 'protocol.json').exists():
        runner.frozen(out)
        print('Existing graph-ID protocol verified; nothing overwritten.')
        return
    runner.initialize(out)
    protocol = runner.read(out / 'protocol.json')
    protocol.pop('protocol_hash')
    protocol.update(version='reference-study-local-graphids-v1',
        engineering_followup=True,
        development_adaptation_disclosure='Claim identity/link constraints added after observing graph failures on these original development cases. Dependencies restricted to earlier draft units after a separate demo produced a cycle. This ordered-DAG representation cannot encode forward-premise dependencies. Not held-out validation or evidence of generalization.',
        structural_schema_specialization=protocol['structural_schema_specialization'] +
            ' Claim IDs/texts assigned deterministically in input unit order; dependencies limited to earlier claims; contradiction IDs limited to other known claims; evidence references must have observation-ID syntax. Core DAG/existence checks retained. Ordered-DAG constraint can miss forward-premise links.',
        prior_local_study='data/evaluation/reference_study_40_local',
        protocol_created_at=runner.now())
    protocol['protocol_hash'] = runner.digest(protocol)
    runner.save(out / 'protocol.json', protocol)
    print('Frozen explicit DEVELOPMENT engineering follow-up.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['init', 'collect', 'review'])
    parser.add_argument('--study', type=Path, default=OUT)
    parser.add_argument('--limit', type=int, default=280)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('limit must be positive')
    configure()
    try:
        if args.stage == 'init':
            initialize(args.study)
        else:
            getattr(runner, args.stage)(args.study, args.limit)
    except runner.LocalInfrastructureError as exc:
        runner.save(args.study / 'PAUSED.json', {'at': runner.now(), 'stage': args.stage, 'reason': str(exc)})
        print(str(exc), flush=True)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
