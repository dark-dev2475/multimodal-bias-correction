import pytest

from scripts.evaluate_embedding_similarity import comparison_rows, cosine, summarize


def example():
    return {'case': {'example_id': 'case1'},
            'reference': {'reference': {'reference_answer': 'Insufficient evidence.'}},
            'outputs': {'eigr': {'status': 'abstained', 'final_response': 'Cannot determine.'},
                        'raw': {'status': 'failed', 'final_response': 'A partial, invalid draft.'}}}


def test_failures_and_missing_outputs_have_no_similarity():
    rows = comparison_rows({'examples': [example()]})
    assert len(rows) == 7
    assert [r['system'] for r in rows if r['scorable']] == ['eigr']
    assert all(r['cosine_similarity'] is None for r in rows)
    rows[1]['cosine_similarity'] = .75
    results = summarize(rows)
    assert results['raw']['mean_cosine_scored_only'] is None
    assert results['raw']['unscored'] == 1
    assert results['eigr']['mean_cosine_scored_only'] == .75


def test_cosine_not_an_accuracy_or_nonnegative_score():
    assert cosine([1, 0], [1, 0]) == 1
    assert cosine([1, 0], [-1, 0]) == -1
    assert cosine([2, 0], [0, 3]) == 0
    with pytest.raises(ValueError, match='Zero'):
        cosine([0, 0], [1, 0])
    with pytest.raises(ValueError, match='dimensions'):
        cosine([1], [1, 0])


def test_duplicate_cases_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        comparison_rows({'examples': [example(), example()]})
