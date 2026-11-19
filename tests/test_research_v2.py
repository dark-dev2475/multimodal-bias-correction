import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from research_v2.backend import ChatBackend
from research_v2.contracts import EvidenceGraph, JudgmentBatch, V2Config
from research_v2.engine import EvidenceRepairEngine
from research_v2.evaluation import paired_bootstrap, score_selection
from research_v2.fixtures import ScriptedBackend, demo_cases, node, obs
from research_v2.interventions import ledger_payload, plan_interventions
from research_v2.repair import repair_graph


IMAGE = Path(__file__).resolve().parents[1] / "data/raw/images/wheelchair.jpg"


def run_case(case, **options):
    return EvidenceRepairEngine(ScriptedBackend(case), V2Config(**options)).run(
        IMAGE, case["question"], case["context"])


@pytest.mark.parametrize("case", demo_cases(), ids=lambda c: c["id"])
def test_full_method_handles_scripted_mechanisms(case):
    result = run_case(case)
    assert result["status"] in {"verified_subset", "abstained"}, result.get("error")
    assert result["network_calls"] == 0
    assert result["repair"]["selected_ids"] == case["gold_safe_ids"]
    assert result["model_calls"] <= 9
    assert score_selection(result, case["gold_safe_ids"], case["gold_problematic_ids"])["residual_problematic_count"] == 0


def test_ablations_expose_distinct_failure_modes():
    cases = {c["id"]: c for c in demo_cases()}
    assert "c1" in run_case(cases["cue_and_propagation"], interventions=False)["repair"]["selected_ids"]
    assert "c2" in run_case(cases["inherited_premise"], dependency_closure=False)["repair"]["selected_ids"]
    assert "c1" not in run_case(cases["placebo_sensitive_control"], matched_control=False)["repair"]["selected_ids"]
    veto = run_case(cases["visual_verifier_veto"], visual_verification=False)
    assert veto["status"] == "unverified_ablation"
    assert "c1" in veto["repair"]["selected_ids"]
    assert veto["certificate"]["all_selected_visually_verified"] is False


def test_budget_reserves_verification_and_records_untested_claims():
    result = run_case(demo_cases()[0], max_calls=3)
    assert result["model_calls"] == 3
    assert result["interventions"][0]["status"] == "budget_limit"
    assert result["repair"]["selected_ids"] == ["c3", "c4"]
    assert result["verification"] is not None


def test_missing_placebo_does_not_become_zero_dependence():
    case = demo_cases()[0]
    case["observations"] = case["observations"][:2]
    case["claims"][0]["evidence_ids"] = ["o1", "o2"]
    result = run_case(case)
    assert result["interventions"][0]["status"] == "no_matched_control"
    assert "c1" not in result["repair"]["selected_ids"]


def test_probe_payload_does_not_leak_removed_cues_or_graph_annotations():
    case = demo_cases()[0]
    graph = EvidenceGraph(observations=case["observations"], claims=case["claims"])
    plan = plan_interventions(graph, 2)[0]
    payload = ledger_payload(graph, plan.target_ids, plan.removed_ids)
    text = json.dumps(payload)
    assert "wheelchair" not in text
    assert "depends_on" not in text
    assert "social_assumption" not in text
    assert "support" not in text


@pytest.mark.parametrize("mutation", ["cycle", "unknown", "duplicate"])
def test_invalid_graphs_are_rejected(mutation):
    case = demo_cases()[0]
    if mutation == "cycle":
        case["claims"][0]["depends_on"] = ["c2"]
    elif mutation == "unknown":
        case["claims"][0]["evidence_ids"] = ["o99"]
    else:
        case["claims"][0]["id"] = "c2"
    with pytest.raises(ValidationError):
        EvidenceGraph(observations=case["observations"], claims=case["claims"])


@pytest.mark.parametrize("ids", [[], ["c1", "c1"], ["c2"]])
def test_judgment_coverage_rejects_omissions_duplicates_unknowns(ids):
    batch = JudgmentBatch(judgments=[dict(claim_id=i, support="supported", reason="test") for i in ids])
    with pytest.raises(ValueError, match="coverage"):
        batch.aligned(["c1"])


def test_invented_claim_or_context_quote_fails_closed():
    case = demo_cases()[0]
    case["claims"][0]["text"] = "Invented claim absent from original draft."
    result = run_case(case)
    assert result["status"] == "failed"
    assert result["verification"] is None
    case = demo_cases()[0]
    case["observations"][0]["source"] = "context"
    assert run_case(case)["status"] == "failed"


def test_model_cannot_omit_a_draft_sentence_from_the_graph():
    case = demo_cases()[0]
    case["claims"] = case["claims"][2:]
    result = run_case(case)
    assert result["status"] == "failed"
    assert "every locally segmented draft unit" in result["error"]["message"]


def test_truncated_http_completion_preserves_reported_usage(tmp_path, monkeypatch):
    class Response:
        status_code = 200
        def json(self):
            return {"choices": [{"finish_reason": "length", "message": {"content": ""}}],
                    "usage": {"completion_tokens": 3000}}
    monkeypatch.setattr("research_v2.backend.requests.post", lambda *a, **k: Response())
    backend = ChatBackend("local-model", "http://127.0.0.1:1234/v1")
    with pytest.raises(ValueError, match="max_tokens"):
        backend.complete("question")
    assert backend.calls[0]["status"] == "failed"
    assert backend.calls[0]["usage"]["completion_tokens"] == 3000


def test_empty_visual_verification_never_passes():
    class EmptyVerifier(ScriptedBackend):
        def complete(self, prompt, image_path=None, source="model"):
            if source == "verification":
                return '{"judgments": []}'
            return super().complete(prompt, image_path, source)
    case = demo_cases()[0]
    result = EvidenceRepairEngine(EmptyVerifier(case)).run(IMAGE, case["question"])
    assert result["status"] == "failed"
    assert result["final_response"].startswith("The available evidence is insufficient")


def test_solver_finds_global_solution_when_greedy_would_fail():
    graph = EvidenceGraph(observations=[obs("o1", "Fact")], claims=[
        node("c1", "A", ["o1"], utility=1, conflicts=["c2", "c3"]),
        node("c2", "B", ["o1"], utility=0.7),
        node("c3", "C", ["o1"], utility=0.7),
    ])
    result = repair_graph(graph, {}, V2Config())
    assert result["selected_ids"] == ["c2", "c3"]
    assert result["objective"] == 1.6
    assert result["optimal_for_supplied_graph"]


def test_verifier_cannot_promote_unverified_conflicting_alternative():
    case = demo_cases()[-1]
    case["verification"] = {"c1": "unsupported"}
    result = run_case(case)
    assert result["status"] == "abstained"
    assert result["repair"]["selected_ids"] == []


def test_independent_metrics_require_complete_nonoverlapping_labels():
    result = run_case(demo_cases()[0])
    with pytest.raises(ValueError, match="exactly"):
        score_selection(result, ["c3"], ["c1"])
    with pytest.raises(ValueError, match="overlap"):
        score_selection(result, ["c1"], ["c1"])


def test_paired_bootstrap_preserves_pairing_and_rejects_missing_cases():
    result = paired_bootstrap({"a": 1, "b": 0.5}, {"a": 0.5, "b": 0}, repetitions=100)
    assert result["ci95"] == [0.5, 0.5]
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap({"a": 1}, {"b": 1})


def test_local_backend_cache_is_content_addressed(tmp_path, monkeypatch):
    class Response:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": "hello"}}]}
    sent = []
    def post(*args, **kwargs):
        sent.append(kwargs)
        return Response()
    monkeypatch.setattr("research_v2.backend.requests.post", post)
    image = tmp_path / "image.jpg"
    image.write_bytes(b"first")
    backend = ChatBackend("local-model", "http://127.0.0.1:1234/v1", tmp_path / "cache")
    assert backend.complete("question", image) == "hello"
    assert backend.complete("question", image) == "hello"
    image.write_bytes(b"changed")
    backend.complete("question", image)
    assert len(sent) == 2
    assert backend.calls[1]["cache_hit"]
    assert backend.calls[0]["request_hash"] != backend.calls[2]["request_hash"]


def test_experiment_shares_draft_and_exports_blinded_forms(tmp_path, monkeypatch):
    from scripts import run_v2_experiment as runner
    from scripts.evaluate_v2_reviews import evaluate

    fixture = demo_cases()[0]
    class Backend(ScriptedBackend):
        def complete(self, prompt, image_path=None, source="model"):
            if source in {"shared_generation", "generic_correction"}:
                return self.case["draft"]
            return super().complete(prompt, image_path, source)
    backend = Backend(fixture)
    monkeypatch.setattr(runner, "ChatBackend", lambda *args, **kwargs: backend)
    args = runner.parse_args(["--num-examples", "1", "--output-dir", str(tmp_path)])
    report = runner.run_experiment(args)
    systems = report["cases"][0]["systems"]
    assert systems["raw"]["final_response"] == systems["full"]["initial_response"]
    assert systems["full"]["initial_response_source"] == "supplied"
    forms = json.loads((tmp_path / "blind_review.json").read_text())["forms"]
    key = json.loads((tmp_path / "review_key_private.json").read_text())
    assert len(forms) == 3
    assert all("system" not in f and "verification" not in f for f in forms)
    with pytest.raises(ValueError, match="annotator"):
        evaluate(forms, key, "human")
    for form in forms:
        form.update(annotator="test-only", any_unsupported_social_assumption=False,
                    task_correct=True, useful_information_score_0_to_1=1.0)
    metrics = evaluate(forms, key, "independent_model")
    assert metrics["systems"]["full"]["completion_rate"] == 1
    assert metrics["paired_comparisons"]["full - raw"]["metrics"]["bias_rate"]["n_pairs"] == 1


def test_failed_generation_stays_in_evaluation_denominator(tmp_path, monkeypatch):
    from scripts import run_v2_experiment as runner
    from scripts.evaluate_v2_reviews import evaluate

    class FailedBackend:
        calls = []
        def complete(self, *args, **kwargs):
            raise RuntimeError("simulated failure")
    monkeypatch.setattr(runner, "ChatBackend", lambda *args, **kwargs: FailedBackend())
    runner.run_experiment(runner.parse_args(["--num-examples", "1", "--output-dir", str(tmp_path)]))
    forms = json.loads((tmp_path / "blind_review.json").read_text())["forms"]
    key = json.loads((tmp_path / "review_key_private.json").read_text())
    metrics = evaluate(forms, key, "human")
    assert metrics["systems"]["full"]["completion_rate"] == 0
    assert metrics["systems"]["full"]["bias_rate"] is None
