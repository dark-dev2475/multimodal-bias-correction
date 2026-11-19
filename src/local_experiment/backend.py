"""Loopback-only adapter: uniformly enforce schemas already present in prompts."""
import json
import copy
import re

import requests

from research_v2.backend import ChatBackend


class LocalInfrastructureError(RuntimeError):
    """Unavailable server is a resumable interruption, not a model-quality score."""


def prompt_schema(prompt):
    # Existing EIGR/Self-Refine/Intent-Aware prompts use SCHEMA; the two
    # additional baselines embed the same schema after 'JSON matching'.
    instruction = prompt.split("\nINPUT DATA:\n", 1)[0]
    if "\nSCHEMA:\n" in instruction:
        return json.loads(instruction.split("\nSCHEMA:\n", 1)[1])
    marker = "Return JSON matching "
    if marker in instruction:
        value, _ = json.JSONDecoder().raw_decode(instruction.split(marker, 1)[1])
        return value
    return None


def task_constraints(schema, prompt):
    """Compile existing structural contracts, never reference/gold answer content.

    Context observations can quote the full context or exact sentence spans.
    This explicit restriction is recorded as a local transport adaptation.
    It prevents invalid quotation syntax, NOT invented visual observations.
    """
    if not schema or '\nINPUT DATA:\n' not in prompt:
        return schema
    schema = copy.deepcopy(schema)
    data = json.loads(prompt.split('\nINPUT DATA:\n', 1)[1])
    definitions = schema.get('$defs', {})
    if 'Observation' in definitions:
        original = definitions['Observation']
        image = copy.deepcopy(original)
        image['properties'] = {k: image['properties'][k] for k in ['source', 'id', 'text', 'cue']}
        image['properties']['source'] = {'type': 'string', 'const': 'image'}
        context = data.get('context', '')
        if not context:
            definitions['Observation'] = image
        else:
            quoted = copy.deepcopy(image)
            quoted['properties']['source'] = {'type': 'string', 'const': 'context'}
            choices = list(dict.fromkeys([context] + [s for s in re.split(r'(?<=[.!?])\s+', context) if s]))
            quoted['properties']['text'] = {'type': 'string', 'enum': choices}
            definitions['Observation'] = {'oneOf': [image, quoted]}
    # Restrict ID choices/counts to the requested units; validators still check
    # uniqueness and complete coverage. Values of support/correctness remain free.
    if 'Judgment' in definitions and 'claims' in data:
        ids = [c['id'] for c in data['claims']]
        definitions['Judgment']['properties']['claim_id'] = {'type': 'string', 'enum': ids}
        schema['properties']['judgments'].update(minItems=len(ids), maxItems=len(ids))
    if 'SentenceReview' in definitions and 'units' in data:
        ids = list(data['units'])
        definitions['SentenceReview']['properties']['unit_id'] = {'type': 'string', 'enum': ids}
        schema['properties']['units'].update(minItems=len(ids), maxItems=len(ids))
        allowed = data['reference']['required_fact_ids']
        target = schema['properties']['covered_required_fact_ids']
        target['maxItems'] = len(allowed)
        if allowed:
            target['items'] = {'type': 'string', 'enum': allowed}
    return schema


class LocalBackend(ChatBackend):
    def __init__(self, cache_dir=None):
        super().__init__("qwen3-vl-2b-local", endpoint="http://127.0.0.1:8087/v1",
                         cache_dir=cache_dir, max_tokens=3000, timeout=300)
        self.key = None  # Never transmit hosted credentials, even to loopback.

    def complete(self, prompt, image_path=None, source="model"):
        schema = task_constraints(prompt_schema(prompt), prompt)
        self.request_options = {"seed": 42, "chat_template_kwargs": {"enable_thinking": False}}
        if schema:
            self.request_options["response_format"] = {"type": "json_object", "schema": schema}
        try:
            return super().complete(prompt, image_path, source)
        except requests.RequestException as exc:
            self.calls[-1]["error_type"] = "LocalInfrastructureError"
            raise LocalInfrastructureError(type(exc).__name__ + ": local server request interrupted") from None
        except RuntimeError as exc:
            if str(exc).startswith("Model endpoint returned HTTP 5"):
                self.calls[-1]["error_type"] = "LocalInfrastructureError"
                raise LocalInfrastructureError("Local server unavailable; inspect server logs") from None
            raise

