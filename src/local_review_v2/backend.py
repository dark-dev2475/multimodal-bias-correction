"""Constrain annotation syntax/length, not the correctness/support decisions."""
import copy
import itertools
import json
import requests

from research_v2.backend import ChatBackend
from local_experiment.backend import LocalInfrastructureError, prompt_schema, task_constraints


def bounded_review_schema(prompt):
    schema = task_constraints(prompt_schema(prompt), prompt)
    if not schema or 'SentenceReview' not in schema.get('$defs', {}):
        raise ValueError('This adapter is restricted to semantic review annotations')
    data = json.loads(prompt.split('\nINPUT DATA:\n', 1)[1])
    variants = []
    for uid in data['units']:
        unit = copy.deepcopy(schema['$defs']['SentenceReview'])
        unit['properties']['unit_id'] = {'type': 'string', 'const': uid}
        unit['properties']['explanation']['maxLength'] = 240
        variants.append(unit)
    schema['properties']['units'] = {'type': 'array', 'prefixItems': variants,
                                    'minItems': len(variants), 'maxItems': len(variants)}
    schema['properties']['explanation']['maxLength'] = 400
    ids = data['reference']['required_fact_ids']
    if len(ids) > 12:
        raise ValueError('Reference schema supports at most 12 facts')
    subsets = [list(s) for n in range(len(ids)+1) for s in itertools.combinations(ids, n)]
    schema['properties']['covered_required_fact_ids'] = {'type': 'array', 'enum': subsets}
    return schema


class BoundedReviewBackend(ChatBackend):
    def __init__(self, cache_dir=None):
        super().__init__('qwen3-vl-2b-local', endpoint='http://127.0.0.1:8087/v1',
                         cache_dir=cache_dir, max_tokens=3000, timeout=300)
        self.key = None

    def complete(self, prompt, image_path=None, source='bounded_blind_review'):
        self.request_options = {'seed': 42, 'chat_template_kwargs': {'enable_thinking': False},
            'response_format': {'type': 'json_object', 'schema': bounded_review_schema(prompt)}}
        try:
            return super().complete(prompt, image_path, source)
        except requests.RequestException as exc:
            self.calls[-1]['error_type'] = 'LocalInfrastructureError'
            raise LocalInfrastructureError(type(exc).__name__ + ': review interrupted') from None
        except RuntimeError as exc:
            if str(exc).startswith('Model endpoint returned HTTP 5'):
                self.calls[-1]['error_type'] = 'LocalInfrastructureError'
                raise LocalInfrastructureError('Local server unavailable during review') from None
            raise
