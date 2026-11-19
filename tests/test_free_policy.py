import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils import free_policy as policy
from research_v2.backend import ChatBackend
from models.vlm import OpenRouterVLM
from scripts import run_v2_baseline_study as study


@pytest.mark.parametrize("model", ["openai/gpt-4o-mini", "inclusionai/ling-3.0-flash-vl", "openrouter/free", "test"])
def test_paid_and_auto_routes_blocked_before_network(model):
    with pytest.raises(ValueError, match="Free-only"):
        ChatBackend(model)
    with pytest.raises(ValueError, match="Free-only"):
        OpenRouterVLM(model)


def test_remote_endpoints_blocked_and_loopback_allowed():
    with pytest.raises(ValueError, match="Free-only"):
        ChatBackend("model", "https://example.com/v1")
    assert ChatBackend("my-local-vlm", "http://127.0.0.1:1234/v1").local


@pytest.mark.parametrize("pricing", [{}, {"prompt": "0", "completion": "1"},
    {"prompt": "NaN", "completion": "0"}, {"prompt": "0", "completion": "0", "image": "0.01"}])
def test_unknown_nonzero_and_image_prices_rejected(pricing):
    with pytest.raises(ValueError):
        policy.validate_entry({"pricing": pricing, "architecture": {"input_modalities": ["image"]}})


def test_quota_is_persistent_and_does_not_store_key(tmp_path):
    db = tmp_path / "quota.sqlite3"
    for _ in range(2):
        policy.reserve_free_request("secret-unit-key", db, daily_limit=2, interval=0)
    with pytest.raises(policy.FreeQuotaPause):
        policy.reserve_free_request("secret-unit-key", db, daily_limit=2, interval=0)
    assert b"secret-unit-key" not in db.read_bytes()


def test_free_transport_caps_price_and_cache_uses_no_quota(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-key")
    quotas, payloads = [], []
    monkeypatch.setattr("research_v2.backend.before_hosted_request", lambda *a: quotas.append(True))
    class Response:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": "Answer"}}], "usage": {"cost": 0}}
    def post(*args, **kwargs):
        payloads.append(kwargs["json"])
        return Response()
    monkeypatch.setattr("research_v2.backend.requests.post", post)
    backend = ChatBackend(policy.DEFAULT_MODEL, cache_dir=tmp_path)
    assert backend.complete("Q") == backend.complete("Q") == "Answer"
    assert len(quotas) == len(payloads) == 1
    assert payloads[0]["provider"]["max_price"] == {"prompt": 0, "completion": 0}
    assert payloads[0]["provider"]["allow_fallbacks"] is False


def test_study_defaults_are_zero_cost_separate_free_models():
    args = study.parse_args([])
    assert args.max_cost_usd == 0 and args.workers == 1
    assert args.model.endswith(":free") and args.judge_model.endswith(":free")
    assert args.model != args.judge_model
    assert "free_baseline" in str(args.output_dir)


def test_quota_pause_checkpoints_and_resume_reuses_completed_stages(monkeypatch, tmp_path):
    args = study.parse_args(["--output-dir", str(tmp_path)])
    example = SimpleNamespace(example_id="case", category="Age", image_path=Path("image.jpg"),
                              question="Q", metadata={"leakage_group": "g", "context": ""})
    calls = []
    class Backend:
        def __init__(self, model, cache, tokens, price, budget):
            self.calls, self.budget = [], budget
        def complete(self, prompt, image_path=None, source="model"):
            calls.append(source)
            if source == "generic_correction" and not resumed[0]:
                self.budget.halt_reason = "Free quota pause"
                raise policy.FreeQuotaPause("Free quota pause")
            self.calls.append({"source": source, "network_attempted": True, "cache_hit": False})
            return "Answer"
    resumed = [False]
    monkeypatch.setattr(study, "BudgetBackend", Backend)
    monkeypatch.setattr(study, "SYSTEMS", ("raw", "generic"))
    monkeypatch.setattr(study, "review_batch", lambda b, i, q, c, ordered: {t: {} for t in ordered})
    price = {args.model: {}, args.judge_model: {}}
    with pytest.raises(policy.FreeQuotaPause):
        study.run_case(example, args, "protocol", price, study.SpendBudget(0))
    checkpoint = json.loads((tmp_path / "checkpoints/case.json").read_text())
    assert set(checkpoint["systems"]) == {"raw"}
    assert not (tmp_path / "cases/case.json").exists()
    resumed[0] = True
    result = study.run_case(example, args, "protocol", price, study.SpendBudget(0))
    assert result["systems"]["generic"]["status"] == "completed"
    assert calls.count("shared_generation") == 1
