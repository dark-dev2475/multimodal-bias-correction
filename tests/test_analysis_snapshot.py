import pytest
from scripts.export_bounded_study import copy_immutable


def test_analysis_copy_is_exact_and_refuses_changed_inputs(tmp_path):
    source, target = tmp_path / 'source.json', tmp_path / 'view/result.json'
    source.write_bytes(b'{"status":"failed"}\n')
    copy_immutable(source, target)
    assert source.read_bytes() == target.read_bytes()
    copy_immutable(source, target)
    source.write_bytes(b'{"status":"completed"}\n')
    with pytest.raises(ValueError, match='changed'):
        copy_immutable(source, target)
    assert target.read_bytes() == b'{"status":"failed"}\n'
