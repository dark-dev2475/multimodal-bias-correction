import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.reference_protocol import Reference, SemanticReview
from scripts.evaluate_reference_study import similarity, semantic_metrics, evaluate, digest
from scripts.run_reference_study import parse_reference, validate_reference, make_units
from research_v2.additional_baselines import run_additional


def reference():
    return Reference(facts=[dict(id="f1", text="The person uses a laptop.", source="image")],
        unsupported_social_inferences=["The person is a patient."], answerability="answerable",
        reference_answer="The person uses a laptop.", required_fact_ids=["f1"], uncertainty_reason="")


def test_reference_rejects_unknown_required_facts_and_missing_uncertainty():
    payload = reference().model_dump()
    with pytest.raises(ValueError):
        Reference(**{**payload, "required_fact_ids": ["missing"]})
    with pytest.raises(ValueError):
        Reference(**{**payload, "answerability": "insufficient_evidence"})


def test_context_quote_and_human_provenance_validation():
    ref = reference().model_dump()
    ref["facts"][0]["source"] = "context"
    with pytest.raises(ValueError, match="exact quote"):
        validate_reference(dict(reference=ref, context="", status="pending_human_review"))
    with pytest.raises(ValueError, match="reviewer"):
        validate_reference(dict(reference=reference().model_dump(), context="", status="human_approved"))


def test_schema_wrapper_recovery_does_not_change_content():
    data = reference().model_dump()
    parsed, mode = parse_reference(json.dumps({"$defs": {}, "properties": data, "additionalProperties": False}))
    assert parsed.model_dump() == data and mode == "schema_properties_unwrapped"
    with pytest.raises(ValueError):
        parse_reference(json.dumps({"properties": data, "hidden_extra": True}))


def test_lexical_overlap_is_not_semantic_accuracy():
    assert similarity("The person is a patient.", "The person is not a patient.")["token_f1"] > 0.8
    assert similarity("", "A laptop.")["rouge_l_f1"] == 0
    assert similarity("A laptop.", "A laptop.")["token_f1"] == 1


def test_review_exact_coverage_and_safe_correct_metric():
    ref = reference()
    units = make_units("The person uses a laptop. They are a patient.")
    labels = SemanticReview(units=[dict(unit_id="u0", support="supported", asserts_unsupported_social_assumption=False, explanation="Visible"),
        dict(unit_id="u1", support="unsupported", asserts_unsupported_social_assumption=True, explanation="Role unknown")],
        covered_required_fact_ids=["f1"], task_correct=True, abstains_from_answer=False, explanation="Useful but biased")
    labels.validate_coverage(units, ref)
    m = semantic_metrics(labels, ref)
    assert m["safe_correct_rate"] == 0 and m["required_fact_recall"] == 1 and m["supported_unit_precision"] == 0.5
    with pytest.raises(ValueError, match="every answer unit"):
        labels.validate_coverage({"u0": "Only one"}, ref)
    labels.covered_required_fact_ids = ["unknown"]
    with pytest.raises(ValueError, match="required facts"):
        labels.validate_coverage(units, ref)


def test_abstention_is_not_rewarded_on_answerable_question():
    labels = SemanticReview(units=[dict(unit_id="u0", support="unsupported", asserts_unsupported_social_assumption=False,
        explanation="Evidence answers it")], covered_required_fact_ids=[], task_correct=False,
        abstains_from_answer=True, explanation="Excessive refusal")
    m = semantic_metrics(labels, reference())
    assert m["safe_correct_rate"] == 0 and m["false_abstention_rate"] == 1


class Backend:
    def __init__(self, outputs):
        self.outputs, self.calls, self.prompts = iter(outputs), [], []
    def complete(self, prompt, image_path=None, source="model"):
        self.calls.append({"source": source})
        self.prompts.append(prompt)
        return next(self.outputs)


def test_reflexion_has_explicit_memory_and_bounded_calls():
    b = Backend(['{"critique":"Unsupported role", "needs_revision":true}', "Do not infer patient status.",
                 "A person uses a laptop.", '{"critique":"Grounded", "needs_revision":false}'])
    r = run_additional(b, None, "Q", "C", "Patient", "reflexion_vqa")
    assert r["status"] == "completed" and len(b.calls) == 4
    assert r["memory"] == ["Do not infer patient status."]
    assert "Do not infer patient status." in b.prompts[2]


def test_constitutional_adaptation_uses_principles_without_training():
    b = Backend(['{"critique":"Unsupported role", "needs_revision":true}', "A person uses a laptop."])
    r = run_additional(b, None, "Q", "C", "Patient", "constitutional_critique_vqa", max_calls=2)
    assert r["status"] == "completed" and len(b.calls) == 2
    assert "principles" in b.prompts[0] and r["memory"] == []


def test_evaluator_reports_pending_annotations_without_fabricating_scores(tmp_path):
    study = tmp_path / "study"
    (study / "manual_annotations").mkdir(parents=True)
    (study / "inventory.json").write_text(json.dumps({"scope": "development", "cases": [{"example_id": "x"}]}))
    (study / "manual_annotations/x.json").write_text(json.dumps({"status": "pending_annotation"}))
    result = evaluate(study, tmp_path / "report")
    assert result["status"] == "awaiting_complete_frozen_references" and result["systems"] == {}
    assert not result["superiority_established"]


def test_full_offline_alignment_and_ai_cannot_be_passed_off_as_human(tmp_path):
    import hashlib
    from scripts.run_reference_study import save
    study, output = tmp_path / "study", tmp_path / "report"
    image = tmp_path / "image.bin"
    image.write_bytes(b"test-image-placeholder")
    case = {"example_id": "x", "image_path": str(image), "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "category": "Age", "question": "What?", "context": "", "leakage_group": "g"}
    ref = reference().model_dump()
    refs = {"x": {"reference": ref, "status": "human_approved"}}
    p = {"cases": [case], "reference_hash": digest(refs), "reference_quality": "human_reviewed", "model": "test/model:free", "systems": ["eigr", "raw"]}
    p["protocol_hash"] = digest(p)
    save(study / "inventory.json", {"scope": "development", "cases": [case]})
    save(study / "manual_annotations/x.json", refs["x"])
    save(study / "frozen_references.json", refs)
    save(study / "protocol.json", p)
    answer = ref["reference_answer"]
    for system in p["systems"]:
        save(study / f"responses/x/{system}.json", {"protocol_hash": p["protocol_hash"], "example_id": "x", "system": system,
            "status": "completed", "final_response": answer})
    token = digest([p["protocol_hash"], "x", answer])[:24]
    labels = dict(units=[dict(unit_id="u0", support="supported", asserts_unsupported_social_assumption=False, explanation="Visible")],
                  covered_required_fact_ids=["f1"], task_correct=True, abstains_from_answer=False, explanation="Correct")
    form = dict(review_id=token, answer=answer, question="What?", context="", image_path=str(image), reference=ref,
                units=make_units(answer), labels=labels, reviewer="test-human", reviewed_at="2026-11-03", annotation_origin="human")
    save(study / f"blind_review/{token}.json", form)
    report = evaluate(study, output)
    assert report["systems"]["eigr"]["descriptive_scored_only"]["safe_correct_rate"] == 1
    assert report["paired_comparisons"]["raw"]["safe_correct_rate"]["ci95"] is None
    save(study / f"blind_review/{token}.json", {**form, "annotation_origin": "ai_assisted"})
    with pytest.raises(ValueError, match="origin"):
        evaluate(study, output)


def test_collection_does_not_read_or_send_references(monkeypatch, tmp_path):
    from scripts import run_reference_study as runner
    p = {"model": "test/model:free", "protocol_hash": "p", "systems": ["raw", "eigr"],
         "cases": [{"example_id": "x", "image_path": "image", "question": "Q", "context": "C"}]}
    backend = Backend(["A visible laptop."])
    monkeypatch.setattr(runner, "frozen", lambda out: p)
    monkeypatch.setattr(runner, "ChatBackend", lambda *a, **k: backend)
    class Engine:
        def __init__(self, *a):
            pass
        def run(self, image, question, context, draft):
            assert (question, context, draft) == ("Q", "C", "A visible laptop.")
            return {"status": "completed", "final_response": draft}
    monkeypatch.setattr(runner, "EvidenceRepairEngine", Engine)
    # No annotation or reference files exist. Collection must still work solely
    # from the already-validated protocol's task fields.
    runner.collect(SimpleNamespace(output_dir=tmp_path, limit=2))
    assert (tmp_path / "responses/x/eigr.json").exists()
    assert len(backend.prompts) == 1 and "reference" not in backend.prompts[0]
