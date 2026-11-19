import json
import pytest
from scripts.reuse_local_controls import reuse
from scripts.run_reference_study import save, digest


def setup_pair(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'target'
    common = dict(model='local', installation={'revision': 'fixed'}, server={'ctx': 8192},
        cases=[{'example_id': 'x'}], reference_hash='r', max_calls_including_draft=9,
        max_output_tokens_per_call=3000, temperature=0, seed=42, source_hashes={'baseline.py': 'abc'})
    before, after = {**common, 'version': 'before'}, {**common, 'version': 'after'}
    before['protocol_hash'], after['protocol_hash'] = digest(before), digest(after)
    save(source / 'protocol.json', before)
    save(target / 'protocol.json', after)
    return source, target, before, after


def test_reuse_preserves_failures_and_excludes_changed_method(tmp_path, monkeypatch):
    import scripts.reuse_local_controls as module
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    source, target, before, after = setup_pair(tmp_path)
    row = dict(example_id='x', system='self_refine_vqa', protocol_hash=before['protocol_hash'],
               status='failed', final_response=None, error='invalid structured output')
    save(source / 'responses/x/self_refine_vqa.json', row)
    save(source / 'responses/x/eigr.json', {**row, 'system': 'eigr'})
    assert reuse(source, target) == 1
    result = json.loads((target / 'responses/x/self_refine_vqa.json').read_text())
    assert result['status'] == 'failed' and result['error'] == row['error']
    assert result['reused_from']['result_sha256'] == digest(row)
    assert result['protocol_hash'] == after['protocol_hash']
    assert not (target / 'responses/x/eigr.json').exists()
    assert reuse(source, target) == 0


def test_reuse_rejects_changed_model(tmp_path):
    source, target, before, after = setup_pair(tmp_path)
    after['model'] = 'different'
    after.pop('protocol_hash')
    after['protocol_hash'] = digest(after)
    save(target / 'protocol.json', after)
    with pytest.raises(ValueError, match='model'):
        reuse(source, target)
