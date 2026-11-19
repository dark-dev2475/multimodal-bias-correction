import copy
import pytest

from scripts.prepare_validation_packet import graph_diagnostic, reference_flags
from scripts.validate_human_reference_packet import validate_form
from scripts.evaluate_reference_study import digest
from scripts.evaluate_independent_reference_text import tasks_for, complete_fraction


def test_gate_diagnostic_does_not_force_missing_cues():
    graph = {'observations': [{'id': 'o1', 'text': 'A wheelchair.', 'source': 'image', 'cue': 'none'}],
             'claims': [{'id': 'c1', 'text': 'The person is a patient.', 'kind': 'inference',
                         'social_assumption': True, 'evidence_ids': ['o1'], 'support': 'supported', 'utility': 1}]}
    before = copy.deepcopy(graph)
    row = graph_diagnostic({'example_id': 'x', 'graph': graph})
    assert row['activation_blocker'] == 'no_tagged_cues'
    assert row['planned_interventions'] == 0
    assert graph == before


def test_pending_forms_never_become_human_approvals():
    original = {'example_id': 'x', 'image_path': 'i', 'image_sha256': 'h', 'question': 'Q', 'context': 'A laptop.',
                'reference': {'facts': [{'id': 'f1', 'text': 'A laptop.', 'source': 'context'}],
                              'reference_answer': 'A laptop.', 'required_fact_ids': ['f1'],
                              'answerability': 'answerable', 'uncertainty_reason': '', 'unsupported_social_inferences': []}}
    form = {k: original[k] for k in ['example_id', 'image_path', 'image_sha256', 'question', 'context']}
    form.update(source_reference_hash=digest(original), reference_draft=original['reference'], decision='pending')
    assert not validate_form(form, original)
    form['decision'] = 'approve_as_is'
    with pytest.raises(ValueError, match='reviewer'):
        validate_form(form, original)


def test_reference_audit_flags_inconsistent_answerability_without_relabeling():
    row = {'example_id': 'x', 'context': 'A laptop.', 'reference': {
        'facts': [{'id': 'f1', 'text': 'A laptop.', 'source': 'context'}], 'required_fact_ids': ['f1'],
        'answerability': 'answerable', 'reference_answer': 'The identity cannot be determined.'}}
    assert 'answerable_label_with_uncertainty_language_review_needed' in reference_flags(row)
    assert row['reference']['answerability'] == 'answerable'


def test_fact_coverage_and_support_are_opposite_directions():
    ref = {'facts': [{'id': 'f1', 'text': 'There is a red cup.'}], 'required_fact_ids': ['f1']}
    units, facts = tasks_for('There is a cup.', ref)
    assert units[0]['premise'] == facts[0]['hypothesis'] == 'There is a red cup.'
    assert units[0]['hypothesis'] == facts[0]['premise'] == 'There is a cup.'
    assert complete_fraction([]) is None
    assert complete_fraction([{'status': 'failed', 'prediction': None}]) is None
    assert complete_fraction([{'status': 'completed', 'prediction': 'unknown'}]) == 0
