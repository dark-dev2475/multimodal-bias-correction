import json
import subprocess
from pathlib import Path


DEFAULT_PRICING_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "model_pricing.json"
)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_pricing(path=DEFAULT_PRICING_PATH):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def estimate_cost(model, prompt_tokens, completion_tokens, pricing):
    rates = pricing.get(model, pricing.get("default", {}))
    if prompt_tokens is None or completion_tokens is None:
        return None

    return round(
        prompt_tokens * rates.get("prompt_usd_per_million", 0) / 1_000_000
        + completion_tokens
        * rates.get("completion_usd_per_million", 0)
        / 1_000_000,
        8
    )


def provenance(
    model,
    benchmark_version,
    benchmark_hash,
    configuration,
    prompt_versions=None,
    pipeline_version=None
):
    return {
        "model": model,
        "benchmark_version": benchmark_version,
        "benchmark_hash": benchmark_hash,
        "pipeline_commit": git_commit(),
        "pipeline_version": pipeline_version,
        "prompt_versions": prompt_versions or {},
        "configuration": configuration,
    }