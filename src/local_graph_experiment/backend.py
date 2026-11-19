"""Input-derived claim identities and foreign keys, without reference answers."""
import copy
import json
import requests

from local_experiment.backend import LocalBackend, LocalInfrastructureError, prompt_schema, task_constraints
from research_v2.backend import ChatBackend


def fixed_claim_schema(schema, prompt):
    if not schema or 'ClaimNode' not in schema.get('$defs', {}):
        return schema
    schema = copy.deepcopy(schema)
    data = json.loads(prompt.split('\nINPUT DATA:\n', 1)[1])
    units = data['draft_units']
    ids = [f'c{i+1}' for i in range(len(units))]
    variants = []
    for index, (cid, text) in enumerate(zip(ids, units)):
        node = copy.deepcopy(schema['$defs']['ClaimNode'])
        node['properties']['id'] = {'type': 'string', 'const': cid}
        node['properties']['text'] = {'type': 'string', 'const': text}
        node['properties']['evidence_ids']['items'] = {'type': 'string', 'pattern': r'^o[1-9][0-9]*$'}
        for name in ['depends_on', 'contradicts']:
            # Explicit ORDERED-DAG adaptation: dependencies can only point to
            # earlier draft units. Guarantees acyclicity, but cannot represent
            # forward-premise dependencies. Report this limitation, not parity.
            others = ids[:index] if name == 'depends_on' else [other for other in ids if other != cid]
            node['properties'][name]['maxItems'] = len(others)
            if others:
                node['properties'][name]['items'] = {'type': 'string', 'enum': others}
        variants.append(node)
    # Prefix items fix coverage/order/identity, but leave social labels, evidence,
    # utility and permitted dependency choices to the model. Duplicate references
    # and unknown observation IDs are STILL rejected by core validators.
    schema['properties']['claims'] = {'type': 'array', 'prefixItems': variants,
                                      'minItems': len(units), 'maxItems': len(units)}
    return schema


class GraphIDBackend(LocalBackend):
    def complete(self, prompt, image_path=None, source='model'):
        schema = fixed_claim_schema(task_constraints(prompt_schema(prompt), prompt), prompt)
        self.request_options = {'seed': 42, 'chat_template_kwargs': {'enable_thinking': False}}
        if schema:
            self.request_options['response_format'] = {'type': 'json_object', 'schema': schema}
        try:
            return ChatBackend.complete(self, prompt, image_path, source)
        except requests.RequestException as exc:
            self.calls[-1]['error_type'] = 'LocalInfrastructureError'
            raise LocalInfrastructureError(type(exc).__name__ + ': local server request interrupted') from None
        except RuntimeError as exc:
            if str(exc).startswith('Model endpoint returned HTTP 5'):
                self.calls[-1]['error_type'] = 'LocalInfrastructureError'
                raise LocalInfrastructureError('Local server unavailable; inspect server logs') from None
            raise
