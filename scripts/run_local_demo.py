"""Real-image EIGR demo using only the installed loopback Qwen model."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from local_experiment.backend import LocalBackend
from research_v2.engine import EvidenceRepairEngine
from scripts.run_reference_study import save, now


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, default=ROOT / 'data/raw/images/wheelchair.jpg')
    parser.add_argument('--question', default='What is happening in this image?')
    parser.add_argument('--context', default='')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/version2/local_demo.json')
    args = parser.parse_args()
    backend = LocalBackend(ROOT / 'data/cache/local_demo')
    result = EvidenceRepairEngine(backend).run(args.image, args.question, args.context)
    result.update(timestamp=now(), experiment='separate local-model development demo',
                  structured_decoding='JSON-schema constraints; no algorithm or reference changes')
    save(args.output, result)
    print(result['final_response'])
    print(f"Status: {result['status']}; calls: {result['model_calls']}; saved: {args.output}")
    if result.get('error'):
        print(result['error'])
    raise SystemExit(1 if result['status'] == 'failed' else 0)
