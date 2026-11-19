import json
from local_graph_experiment.backend import fixed_claim_schema
from research_v2.contracts import EvidenceGraph


def test_fixed_claim_ids_do_not_supply_support_or_reference_answers():
    original = EvidenceGraph.model_json_schema()
    prompt = 'Instructions\nINPUT DATA:\n' + json.dumps({'draft_units': ['A patient.', 'Needs help.']})
    schema = fixed_claim_schema(original, prompt)
    claims = schema['properties']['claims']['prefixItems']
    assert claims[0]['properties']['id']['const'] == 'c1'
    assert claims[1]['properties']['text']['const'] == 'Needs help.'
    assert claims[0]['properties']['depends_on']['maxItems'] == 0
    assert claims[1]['properties']['depends_on']['items']['enum'] == ['c1']
    assert claims[0]['properties']['contradicts']['items']['enum'] == ['c2']
    assert claims[0]['properties']['support'] == original['$defs']['ClaimNode']['properties']['support']
    assert 'prefixItems' not in original['properties']['claims']


def test_single_claim_cannot_depend_on_itself_or_invent_another_claim():
    schema = fixed_claim_schema(EvidenceGraph.model_json_schema(),
        'Instructions\nINPUT DATA:\n' + json.dumps({'draft_units': ['A person.']}))
    fields = schema['properties']['claims']['prefixItems'][0]['properties']
    assert fields['depends_on']['maxItems'] == fields['contradicts']['maxItems'] == 0


def test_non_graph_schemas_unchanged():
    schema = {'type': 'object', 'properties': {}}
    assert fixed_claim_schema(schema, 'instruction') == schema
