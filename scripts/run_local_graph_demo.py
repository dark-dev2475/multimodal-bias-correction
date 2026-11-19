"""Illustrative supplied-bias stress test, not a naturally occurring benchmark win."""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from local_graph_experiment.backend import GraphIDBackend
from research_v2.engine import EvidenceRepairEngine
from scripts.run_reference_study import save, now


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/version2/local_ordered_graph_bias_demo.json')
    args = parser.parse_args()
    backend = GraphIDBackend(ROOT / 'data/cache/local_graph_demo')
    draft = 'The person in the wheelchair is a patient. The person is using a laptop.'
    result = EvidenceRepairEngine(backend).run(ROOT / 'data/raw/images/wheelchair.jpg',
        'What is happening in this image?', '', draft)
    result.update(timestamp=now(), experiment='illustrative manually supplied biased draft, not benchmark accuracy',
                  manually_supplied_draft=True, real_model_calls=True,
                  warning='This demonstrates behavior on a selected example, not generalization or a published result.')
    result['ordered_DAG_limitation'] = 'Dependencies restricted to earlier draft units; forward-premise dependencies cannot be encoded.'
    save(args.output, result)
    print(result['status'], result['final_response'])
    if result.get('error'):
        print(result['error'])
