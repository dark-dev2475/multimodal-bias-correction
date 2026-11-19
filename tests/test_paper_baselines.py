import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_v2.paper_baselines import run_refinement, review_answer, Review
from research_v2.batch_judge import review_batch, ReviewBatch
from scripts.run_v2_baseline_study import SpendBudget, select_balanced, summarise, SYSTEMS, BudgetBackend


class Backend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []
        self.prompts = []

    def complete(self, prompt, image_path=None, source="model"):
        self.calls.append({"source": source})
        self.prompts.append(prompt)
        return next(self.outputs)


def test_self_refine_separates_feedback_revision_and_accumulates_history():
    backend = Backend([json.dumps(dict(critique="Remove patient inference", needs_revision=True)),
                       "A person uses a laptop.", json.dumps(dict(critique="Grounded", needs_revision=False))])
    result = run_refinement(backend, Path("image.jpg"), "Q", "C", "A patient uses a laptop.")
    assert result["status"] == "completed" and result["model_calls"] == 3
    assert result["final_response"] == "A person uses a laptop."
    assert result["stop_reason"] == "feedback_satisfied"
    assert "Remove patient inference" in backend.prompts[1]
    assert "Remove patient inference" in backend.prompts[2]


def test_intent_aware_generates_own_answer_and_stops_on_local_total_three():
    perfect = {aspect: dict(score=1, evidence_summary="Established")
               for aspect in ("coherent", "comprehensive", "objective")}
    backend = Backend(["Cannot determine.", json.dumps(perfect)])
    result = run_refinement(backend, None, "Q", "C", None, method="intent_aware_vqa")
    assert result["model_calls"] == 2
    assert result["initial_response_source"] == "method_specific"
    assert result["stop_reason"] == "feedback_satisfied"


def test_refinement_budget_failure_and_invalid_feedback_remain_explicit():
    backend = Backend([json.dumps(dict(critique="Needs revision", needs_revision=True))])
    result = run_refinement(backend, None, "Q", "C", "Draft", max_calls=1)
    assert result["model_calls"] == 1 and result["stop_reason"] == "call_budget"
    assert result["final_response"] == "Draft"
    result = run_refinement(Backend(['{"critique":"No", "needs_revision":"false"}']), None, "Q", "C", "Draft")
    assert result["status"] == "failed" and result["final_response"] is None


def test_judge_is_blinded_and_strictly_validates_labels():
    labels = dict(any_unsupported_social_assumption=False, task_correct=True,
                  useful_information_score_0_to_1=0.5, evidence_summary="Missing context")
    backend = Backend([json.dumps(labels)])
    assert review_answer(backend, Path("image.jpg"), "Q", "C", "Answer") == labels
    data = json.loads(backend.prompts[0].split("\nINPUT DATA:\n")[1])
    assert set(data) == {"question", "context", "candidate_answer"}
    with pytest.raises(ValueError):
        Review(**{**labels, "task_correct": "true"})


def test_balanced_sampling_is_deterministic_group_diverse_and_image_unique():
    examples = [SimpleNamespace(example_id=f"{cat}-{i}", category=cat, image_path=Path(f"{cat}-{i//2}.jpg"),
                               metadata={"leakage_group": str(i % 3)}) for cat in ("Age", "SES") for i in range(12)]
    selected = select_balanced(examples, 4, 42)
    assert len(selected) == 8
    assert [e.example_id for e in selected] == [e.example_id for e in select_balanced(list(reversed(examples)), 4, 42)]
    assert len({str(e.image_path) for e in selected}) == 8
    for category in ("Age", "SES"):
        assert len({e.metadata["leakage_group"] for e in selected if e.category == category}) == 3
    with pytest.raises(ValueError, match="Not enough"):
        select_balanced(examples, 7, 42)


def test_spending_reservations_stop_before_limit_and_survive_interruption(tmp_path):
    budget = SpendBudget(1)
    budget.attach(tmp_path / "journal.json")
    budget.reserve(0.6)
    with pytest.raises(RuntimeError, match="limit"):
        budget.reserve(0.5)
    resumed = SpendBudget(1)
    resumed.attach(tmp_path / "journal.json")
    assert resumed.charged == 0.6
    budget.settle(0.6, 0.1)
    assert budget.charged == 0.1 and budget.reserved == 0


def test_budget_backend_accounts_known_cost_and_failed_unknown_usage(monkeypatch, tmp_path):
    monkeypatch.setattr("research_v2.backend.before_hosted_request", lambda *a: None)
    class Response:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": "Answer"}}], "usage": {"cost": 0.01}}
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-placeholder")
    monkeypatch.setattr("research_v2.backend.requests.post", lambda *a, **k: Response())
    budget = SpendBudget(1)
    backend = BudgetBackend("test/model:free", tmp_path, 100, dict(context_length=1000, prompt=0.0001, completion=0.0001), budget)
    assert backend.complete("Q") == "Answer"
    assert budget.charged == 0.01
    assert backend.complete("Q") == "Answer"
    assert budget.charged == 0.01
    def fail(*args, **kwargs):
        raise RuntimeError("Network failed")
    monkeypatch.setattr("research_v2.backend.requests.post", fail)
    with pytest.raises(RuntimeError):
        backend.complete("Different question")
    assert budget.charged == pytest.approx(0.12)


def test_summary_preserves_failures_in_denominators_and_never_claims_match():
    cases = []
    labels = dict(any_unsupported_social_assumption=False, task_correct=True, useful_information_score_0_to_1=1)
    for i in range(2):
        cases.append(dict(example_id=str(i), leakage_group=str(i), category="Age", physical_call_log=[],
            systems={s: dict(status="failed" if s == "eigr" and i == 0 else "completed", attributed_call_log=[]) for s in SYSTEMS},
            reviews={s: dict(status="failed") if s == "eigr" and i == 0 else dict(status="completed", labels=labels) for s in SYSTEMS}))
    result = summarise(cases)
    assert result["systems"]["eigr"]["task_accuracy_all_attempts"] == 0.5
    assert result["systems"]["eigr"]["bias_worst_case_all_attempts"] == 0.5
    assert result["systems"]["eigr"]["model_failures"] == 1
    assert result["performance_match_established"] is False


def test_batch_judge_requires_complete_unique_anonymous_coverage():
    label = dict(any_unsupported_social_assumption=False, task_correct=True,
                 useful_information_score_0_to_1=1, evidence_summary="Grounded")
    backend = Backend([json.dumps({"reviews": [{"candidate_id": "opaque", **label}]})])
    result = review_batch(backend, None, "Q", "C", {"opaque": "Answer"})
    assert result == {"opaque": label}
    with pytest.raises(ValueError, match="coverage"):
        ReviewBatch(reviews=[]).aligned(["opaque"])
    with pytest.raises(ValueError, match="coverage"):
        ReviewBatch(reviews=[{"candidate_id": "opaque", **label}] * 2).aligned(["opaque"])


def test_provider_payment_error_halts_subsequent_requests(monkeypatch, tmp_path):
    monkeypatch.setattr("research_v2.backend.before_hosted_request", lambda *a: None)
    class Response:
        status_code = 402
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-placeholder")
    requests = []
    def post(*args, **kwargs):
        requests.append(True)
        return Response()
    monkeypatch.setattr("research_v2.backend.requests.post", post)
    budget = SpendBudget(1)
    backend = BudgetBackend("test/model:free", tmp_path, 100, dict(context_length=1000, prompt=0.0001, completion=0.0001), budget)
    with pytest.raises(RuntimeError, match="402"):
        backend.complete("Q")
    with pytest.raises(RuntimeError, match="stopped study"):
        backend.complete("Q2")
    assert len(requests) == 1
