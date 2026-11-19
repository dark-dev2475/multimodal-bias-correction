"""Validate reviewer-supplied forms; never generate approvals or change frozen data."""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts.evaluate_reference_study import read, digest
from evaluation.reference_protocol import Reference


def validate_form(form, original):
    if form['source_reference_hash'] != digest(original):
        raise ValueError('Source reference mismatch')
    for key in ['example_id', 'image_path', 'image_sha256', 'question', 'context']:
        if form[key] != original[key]:
            raise ValueError('Review form changed task identity')
    if form['reference_draft'] != original['reference']:
        raise ValueError('Do not edit reference_draft; use revised_reference')
    if form['decision'] == 'pending':
        return False
    if form['decision'] not in {'approve_as_is', 'revise'}:
        raise ValueError('Unknown review decision')
    if not isinstance(form.get('reviewer'), str) or not form['reviewer'].strip():
        raise ValueError('Missing actual reviewer')
    datetime.fromisoformat(form['reviewed_at'])
    if form.get('attestation') != 'personally_reviewed_image_and_context':
        raise ValueError('Missing personal-review attestation')
    chosen = original['reference'] if form['decision'] == 'approve_as_is' else form['revised_reference']
    ref = Reference.model_validate(chosen)
    if any(f.source == 'context' and f.text not in original['context'] for f in ref.facts):
        raise ValueError('Context facts must quote supplied text')
    return True


def main(packet):
    source = ROOT / 'data/evaluation/reference_study_40_local_graphids'
    originals = read(source / 'frozen_references.json')
    protocol = read(source / 'protocol.json')
    valid, pending = [], []
    for cid, original in originals.items():
        form = read(packet / (cid + '.json'))
        if form['source_protocol_hash'] != protocol['protocol_hash']:
            raise ValueError('Review protocol identity changed')
        if hashlib.sha256(Path(original['image_path']).read_bytes()).hexdigest() != original['image_sha256']:
            raise ValueError('Image changed')
        (valid if validate_form(form, original) else pending).append(cid)
    print(json.dumps({'reviewer_supplied_forms_validated': len(valid), 'pending': len(pending),
                      'complete': not pending, 'old_references_modified': False,
                      'warning': 'This checks form integrity and supplied attestations, not reviewer identity or semantic truth.'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, default=ROOT / 'data/evaluation/validation_v2_round1/human_references')
    main(parser.parse_args().packet)
