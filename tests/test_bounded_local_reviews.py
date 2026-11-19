import json
from evaluation.reference_protocol import SemanticReview
from local_review_v2.backend import bounded_review_schema


def test_review_schema_limits_format_but_leaves_semantic_decisions_open():
    original = SemanticReview.model_json_schema()
    data = {'units': {'u0': 'Patient.', 'u1': 'Laptop.'}, 'reference': {'required_fact_ids': ['f1', 'f2']}}
    schema = bounded_review_schema('Instructions\nSCHEMA:\n' + json.dumps(original) + '\nINPUT DATA:\n' + json.dumps(data))
    units = schema['properties']['units']['prefixItems']
    assert [u['properties']['unit_id']['const'] for u in units] == ['u0', 'u1']
    assert units[0]['properties']['explanation']['maxLength'] == 240
    assert schema['properties']['explanation']['maxLength'] == 400
    assert schema['properties']['covered_required_fact_ids']['enum'] == [[], ['f1'], ['f2'], ['f1', 'f2']]
    assert schema['properties']['task_correct'] == original['properties']['task_correct']
    assert units[0]['properties']['support'] == original['$defs']['SentenceReview']['properties']['support']
