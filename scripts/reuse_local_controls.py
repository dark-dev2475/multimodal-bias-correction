"""Reuse IDENTICAL frozen control results, including failures; never overwrite.

The engineering follow-up changes only graph decoding. These controls do not
use graphs, so rerunning their completed failures would waste time and risk
selective resampling. Provenance identifies every reused artifact explicitly.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_reference_study import read, digest

SOURCE = ROOT / 'data/evaluation/reference_study_40_local'
TARGET = ROOT / 'data/evaluation/reference_study_40_local_graphids'


def reuse(source=SOURCE, target=TARGET):
    before, after = read(source / 'protocol.json'), read(target / 'protocol.json')
    for protocol in [before, after]:
        if digest({k:v for k,v in protocol.items() if k != 'protocol_hash'}) != protocol['protocol_hash']:
            raise ValueError('Protocol integrity failure')
    for name in ['model', 'installation', 'server', 'cases', 'reference_hash', 'max_calls_including_draft',
                 'max_output_tokens_per_call', 'temperature', 'seed']:
        if before[name] != after[name]:
            raise ValueError('Cannot reuse controls with differing ' + name)
    for name, checksum in before['source_hashes'].items():
        if after['source_hashes'].get(name) != checksum:
            raise ValueError('Original implementation differs; cannot reuse controls')
    allowed = ['raw', 'self_refine_vqa', 'intent_aware_vqa', 'reflexion_vqa', 'constitutional_critique_vqa', 'always_abstain']
    count = 0
    for case in before['cases']:
        cid = case['example_id']
        for system in allowed:
            src, dst = source / 'responses' / cid / (system + '.json'), target / 'responses' / cid / (system + '.json')
            if not src.exists() or dst.exists():
                continue
            row = read(src)
            if (row['example_id'], row['system'], row['protocol_hash']) != (cid, system, before['protocol_hash']):
                raise ValueError('Source result identity mismatch')
            old_hash = digest(row)
            row['protocol_hash'] = after['protocol_hash']
            row['reused_from'] = {'path': str(src.relative_to(ROOT)), 'result_sha256': old_hash,
                'protocol_hash': before['protocol_hash'], 'reason': 'Identical non-graph control; includes failures unchanged',
                'new_model_requests_for_this_copy': 0}
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                with dst.open('x', encoding='utf-8') as handle:
                    json.dump(row, handle, indent=2, ensure_ascii=False)
                    handle.write('\n')
                count += 1
            except FileExistsError:
                pass
    print(f'Reused {count} identical frozen controls; no outputs overwritten or failures resampled.')
    return count


if __name__ == '__main__':
    reuse()
