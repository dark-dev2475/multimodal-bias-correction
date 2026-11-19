import json

import pytest

from scripts.check_local_study_package import check_bundle


def setup_bundle(tmp_path):
    case = {'example_id': 'x'}
    protocol = {'cases': [case], 'systems': ['raw', 'eigr'], 'protocol_hash': 'p'}
    outputs = {s: {'example_id': 'x', 'system': s, 'protocol_hash': 'p', 'status': 'failed'} for s in protocol['systems']}
    directory = tmp_path / 'responses/x'
    directory.mkdir(parents=True)
    for system, result in outputs.items():
        (directory / (system + '.json')).write_text(json.dumps(result))
    return {'examples': [{'case': case, 'outputs': outputs}]}, protocol


def test_preserved_failure_is_recorded_not_missing(tmp_path):
    bundle, protocol = setup_bundle(tmp_path)
    assert check_bundle(bundle, protocol, tmp_path) == {'failed': 2}


def test_missing_or_modified_output_fails_handoff_check(tmp_path):
    bundle, protocol = setup_bundle(tmp_path)
    bundle['examples'][0]['outputs']['raw']['status'] = 'completed'
    with pytest.raises(ValueError, match='differs'):
        check_bundle(bundle, protocol, tmp_path)
    del bundle['examples'][0]['outputs']['raw']
    with pytest.raises(ValueError, match='missing'):
        check_bundle(bundle, protocol, tmp_path)
