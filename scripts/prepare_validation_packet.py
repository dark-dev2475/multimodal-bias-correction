"""Audit all 40 existing graphs/references and prepare UNAPPROVED human forms."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from scripts.evaluate_reference_study import read, digest
from scripts.run_reference_study import save
from research_v2.contracts import EvidenceGraph
from research_v2.interventions import plan_interventions

SOURCE = ROOT / 'data/evaluation/reference_study_40_local_graphids'
STUDY = ROOT / 'data/evaluation/validation_v2_round1'
OUTPUT = ROOT / 'output/version2/VALIDATION_ROUND_1'


def graph_diagnostic(result):
    graph = EvidenceGraph.model_validate(result['graph'])
    cue_ids = {o.id for o in graph.observations if o.cue != 'none'}
    social = [c for c in graph.claims if c.kind == 'inference' and c.social_assumption]
    linked = [c.id for c in social if graph.evidence_closure(c.id) & cue_ids]
    texts = [(' '.join(o.text.casefold().split()), o.source) for o in graph.observations]
    draft = {' '.join(c.text.casefold().split()) for c in graph.claims}
    return {'example_id': result['example_id'], 'observations': len(graph.observations),
            'tagged_cue_count': len(cue_ids), 'social_inference_count': len(social),
            'cue_linked_social_claim_ids': linked, 'planned_interventions': len(plan_interventions(graph, 4)),
            'duplicate_observation_texts': len(texts)-len(set(texts)),
            'observations_equal_to_draft_claims': sum(t in draft for t, _ in texts),
            'observation_limit_reached': len(graph.observations) == 32,
            'activation_blocker': 'no_tagged_cues' if not cue_ids else ('no_social_inference' if not social else
                                 ('no_cue_linked_social_inference' if not linked else 'planner_eligible')),
            'warning': 'Text overlap/duplication flags are diagnostics, not proof that a visual fact is false.'}


def reference_flags(record):
    reference = record['reference']
    flags = []
    if reference['answerability'] == 'answerable' and re.search(
            r'cannot be determined|cannot determine|not possible to determine|insufficient|not enough',
            reference['reference_answer'], re.I):
        flags.append('answerable_label_with_uncertainty_language_review_needed')
    if len(reference['facts']) > 1 and len(reference['required_fact_ids']) == len(reference['facts']):
        flags.append('all_facts_required_check_for_decorative_information')
    if any(f['source'] == 'context' and f['text'] not in record['context'] for f in reference['facts']):
        flags.append('nonliteral_context_quote')
    if record['example_id'] == '04_20_2306_1_01':
        flags.append('known_reference_answer_vs_unsupported_inference_list_inconsistency')
    return flags


def main():
    from scripts import run_local_graph_study as generator
    generator.configure()
    protocol = generator.runner.frozen(SOURCE)
    references = read(SOURCE / 'frozen_references.json')
    if digest(references) != protocol['reference_hash']:
        raise ValueError('Frozen reference hash changed')
    graph_rows, reference_rows = [], []
    lines = ['# Human reference review — all 40 images', '',
             '**Nothing in this packet is human-approved.** The initial drafts were AI-assisted. '
             'A person must inspect the image and supplied context before approving or correcting them.', '',
             'Review references without viewing method answers. Do not infer beliefs, diagnosis, ability, '
             'occupation or character from appearance. Identify which facts are actually necessary to '
             'answer the question; decorative details need not be required. Unsupported-inference lists '
             'must not contradict the proposed reference answer. Mark uncertainty when identity/attribution is missing.', '',
             'In each JSON fill decision (approve_as_is or revise), reviewer, reviewed_at (ISO date/time), '
             'and attestation with the exact string personally_reviewed_image_and_context. For revise, '
             'fill revised_reference using the same schema as reference_draft. Leave these fields unset '
             'until an actual person has reviewed the case. No software can verify a reviewer identity.', '',
             'Use scripts/validate_human_reference_packet.py to check a completed packet. '
             'Corrections belong to a new reference version; the old frozen experiment stays unchanged.', '',
             '| Case | Category | Review flags | Image | Review JSON |', '|---|---|---|---|---|']
    for case in protocol['cases']:
        cid = case['example_id']
        record = references[cid]
        flags = reference_flags(record)
        reference_rows.append({'example_id': cid, 'flags': flags, 'human_approved': False})
        graph_rows.append(graph_diagnostic(read(SOURCE / 'responses' / cid / 'eigr.json')))
        path = STUDY / 'human_references' / (cid + '.json')
        form = {'example_id': cid, 'image_path': case['image_path'], 'image_sha256': case['image_sha256'],
                'question': case['question'], 'context': case['context'],
                'source_reference_hash': digest(record), 'source_protocol_hash': protocol['protocol_hash'],
                'reference_draft': record['reference'], 'draft_origin': 'ai_assisted_not_human_approved',
                'review_flags_not_automatic_errors': flags, 'decision': 'pending',
                'reviewer': None, 'reviewed_at': None, 'attestation': None,
                'revised_reference': None, 'reviewer_notes': ''}
        if not path.exists():
            save(path, form)
        else:
            previous = read(path)
            if previous['source_reference_hash'] != form['source_reference_hash']:
                raise ValueError('Existing review form targets a different reference')
        image_path = Path(case['image_path']).as_posix()
        lines.append(f"| {cid} | {case['category']} | {', '.join(flags) or 'routine review'} | [Image](<{image_path}>) | [JSON](<{path.as_posix()}>) |")
    summary = {'original_run_unchanged': True, 'case_count': len(graph_rows),
               'blocker_counts': dict(Counter(r['activation_blocker'] for r in graph_rows)),
               'observations': sum(r['observations'] for r in graph_rows),
               'duplicate_observation_texts': sum(r['duplicate_observation_texts'] for r in graph_rows),
               'cases_with_draft_overlap': sum(r['observations_equal_to_draft_claims'] > 0 for r in graph_rows),
               'cases_at_observation_limit': sum(r['observation_limit_reached'] for r in graph_rows),
               'planned_interventions': sum(r['planned_interventions'] for r in graph_rows),
               'references_flagged_for_extra_review': sum(bool(r['flags']) for r in reference_rows),
               'human_approvals_generated': 0, 'graph_cases': graph_rows, 'reference_cases': reference_rows}
    save(OUTPUT / 'failure_mechanism_audit.json', summary)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'HUMAN_REFERENCE_REVIEW.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    overview = ['# Why interventions did not activate', '',
                f"Audited all {len(graph_rows)} saved EIGR graphs without changing them.", '',
                '| Immediate prerequisite failure | Cases |', '|---|---:|']
    overview += [f'| {reason} | {count} |' for reason, count in summary['blocker_counts'].items()]
    overview += ['', f"The graphs contain {summary['observations']} observations, including "
                 f"{summary['duplicate_observation_texts']} repeated text/source entries. "
                 f"{summary['cases_at_observation_limit']} graphs reach the 32-observation cap. "
                 f"{summary['cases_with_draft_overlap']} contain observations exactly matching a draft claim.", '',
                 'These are measured structural facts. Overlap can also occur for genuinely visible claims, '
                 'so it is a review flag, not automatic proof of hallucination. The root cause inside the '
                 'model has not been causally isolated.', '',
                 'The planner requires a tagged cue, a social-inference claim, and an evidence path linking '
                 'them. None of the 40 graphs meets all three prerequisites. Merely increasing the call '
                 'budget cannot fix these upstream labels/links. No cue removal was artificially forced.', '',
                 'Next experimental requirement: independently check observation truth, cue recall, social-claim '
                 'recall and evidence links before a new end-to-end comparison. Text calibration alone does '
                 'not validate visual extraction. Human reference forms cover all 40 images.']
    (OUTPUT / 'FAILURE_MECHANISM_AUDIT.md').write_text('\n'.join(overview) + '\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if not isinstance(v, list)}, indent=2))


if __name__ == '__main__':
    main()
