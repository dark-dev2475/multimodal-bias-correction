from collections import Counter
import pytest

from validation_v2.calibration import cases, public_input, pointwise_prompt, parse_pointwise, summarize
from scripts.run_validation_calibration import legacy_prompt


def test_balanced_contrasts_and_no_expected_label_leakage():
    rows = cases()
    assert len(rows) == 24
    assert Counter(r['expected'] for r in rows) == {'entailed': 8, 'contradicted': 8, 'unknown': 8}
    row = {**rows[0], 'expected': 'SECRET_EXPECTED_CANARY', 'id': 'SECRET_ID_CANARY'}
    assert set(public_input(row)) == {'premise', 'hypothesis'}
    for prompt in [pointwise_prompt(row), legacy_prompt(row)[0]]:
        assert 'SECRET_EXPECTED_CANARY' not in prompt
        assert 'SECRET_ID_CANARY' not in prompt


def test_gate_rejects_missing_results_and_false_entailment():
    planned = cases()
    good = [{'id': r['id'], 'prediction': r['expected']} for r in planned]
    assert summarize(good, planned)['gate_passed']
    assert not summarize(good[:-1], planned)['gate_passed']
    bad = [{**r} for r in good]
    bad[1]['prediction'] = 'entailed'
    result = summarize(bad, planned)
    assert result['false_entailments'] == 1 and not result['gate_passed']
    with pytest.raises(ValueError, match='Duplicate'):
        summarize(good + good[:1], planned)


def test_pointwise_parser_does_not_cherry_pick_a_verdict_from_prose():
    assert parse_pointwise(' UNKNOWN. ') == 'unknown'
    with pytest.raises(ValueError):
        parse_pointwise('ENTAILED but possibly UNKNOWN')
