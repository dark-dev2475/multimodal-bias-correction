import hashlib
import time
from dataclasses import dataclass
from enum import Enum

from models.vlm import OpenRouterVLM
from pipeline.generator import InitialGenerator
from pipeline.pipeline import BiasCorrectionPipeline
from evaluation.provenance import estimate_cost, load_pricing
from utils.schemas import PIPELINE_VERSION


class BaselineSystem(str, Enum):
    vlm_only = "vlm_only"
    generic_self_correction = "generic_self_correction"
    proposed = "proposed"


GENERIC_CORRECTION_PROMPT_VERSION = "generic_correction_v1"
GENERIC_CORRECTION_PROMPT = """
Rewrite the response below so it is accurate, neutral, and limited to what
can be supported by the image. Remove unsupported assumptions about medical
status, identity, occupation, emotion, relationships, or socioeconomic
status. Preserve supported visual details. Return only the rewritten answer.

Question:
{question}

Response:
{response}
""".strip()


@dataclass
class BaselineResult:
    system: str
    model: str
    prompt_version: str
    prompt_hash: str
    initial_response: str
    final_response: str
    correction_iterations: int
    api_calls: int
    latency_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_cost_usd: float | None = None
    cache_status: str = "not_applicable"

    def as_dict(self):
        return {
            "system": self.system,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
            "initial_response": self.initial_response,
            "final_response": self.final_response,
            "correction_iterations": self.correction_iterations,
            "api_calls": self.api_calls,
            "latency_seconds": round(self.latency_seconds, 6),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "cache_status": self.cache_status
        }


def _prompt_hash(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def _calls(*vlms):
    return sum(
        len(vlm.calls) + sum(call["attempts"] - 1 for call in vlm.calls)
        for vlm in vlms
    )


def _usage(*vlms):
    prompt = 0
    completion = 0
    found = False
    for vlm in vlms:
        for call in vlm.calls:
            usage = call.get("usage") or {}
            if usage.get("prompt_tokens") is not None:
                prompt += usage["prompt_tokens"]
                found = True
            if usage.get("completion_tokens") is not None:
                completion += usage["completion_tokens"]
    return (prompt, completion) if found else (None, None)


def _cost(model, vlms):
    prompt, completion = _usage(*vlms)
    cost = estimate_cost(model, prompt, completion, load_pricing())
    return prompt, completion, cost


class VLMOnlyBaseline:

    prompt_version = "initial_generation_v1"

    def __init__(self):
        self.generator = InitialGenerator()

    def run(self, image_path, question):
        started = time.monotonic()
        response = self.generator.generate(image_path, question)
        prompt_tokens, completion_tokens, cost = _cost(
            self.generator.vlm.model_name, [self.generator.vlm]
        )
        return BaselineResult(
            system=BaselineSystem.vlm_only.value,
            model=self.generator.vlm.model_name,
            prompt_version=self.prompt_version,
            prompt_hash=self.generator.prompt_fingerprint(),
            initial_response=response,
            final_response=response,
            correction_iterations=0,
            api_calls=_calls(self.generator.vlm),
            latency_seconds=time.monotonic() - started,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=cost
        )


class GenericSelfCorrectionBaseline:

    prompt_version = GENERIC_CORRECTION_PROMPT_VERSION

    def __init__(self):
        self.generator = InitialGenerator()
        self.corrector = OpenRouterVLM(model_name=self.generator.vlm.model_name)

    def run(self, image_path, question):
        started = time.monotonic()
        initial = self.generator.generate(image_path, question)
        prompt = GENERIC_CORRECTION_PROMPT.format(
            question=question,
            response=initial
        )
        corrected = self.corrector.complete(
            prompt,
            image_path=image_path,
            source="Generic self-correction"
        ).strip()
        prompt_tokens, completion_tokens, cost = _cost(
            self.generator.vlm.model_name,
            [self.generator.vlm, self.corrector]
        )

        return BaselineResult(
            system=BaselineSystem.generic_self_correction.value,
            model=self.generator.vlm.model_name,
            prompt_version=self.prompt_version,
            prompt_hash=_prompt_hash(GENERIC_CORRECTION_PROMPT),
            initial_response=initial,
            final_response=corrected,
            correction_iterations=1,
            api_calls=_calls(self.generator.vlm, self.corrector),
            latency_seconds=time.monotonic() - started,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=cost
        )


class ProposedPipelineSystem:

    def __init__(
        self,
        max_iterations=2,
        use_structured_feedback=True,
        use_correction_policy=True
    ):
        self.pipeline = BiasCorrectionPipeline(
            use_structured_feedback=use_structured_feedback,
            use_correction_policy=use_correction_policy
        )
        self.max_iterations = max_iterations

    def run(self, image_path, question):
        started = time.monotonic()
        result = self.pipeline.run(
            image_path=image_path,
            question=question,
            max_iterations=self.max_iterations
        )
        api_calls = sum(
            metrics.get("calls", 0) + metrics.get("retries", 0)
            for metrics in result["stage_metrics"].values()
        )
        prompt_tokens = sum(
            metrics.get("prompt_tokens") or 0
            for metrics in result["stage_metrics"].values()
        )
        completion_tokens = sum(
            metrics.get("completion_tokens") or 0
            for metrics in result["stage_metrics"].values()
        )
        has_usage = any(
            metrics.get("prompt_tokens") is not None
            or metrics.get("completion_tokens") is not None
            for metrics in result["stage_metrics"].values()
        )
        model = self.pipeline.generator.vlm.model_name

        return BaselineResult(
            system=BaselineSystem.proposed.value,
            model=model,
            prompt_version="pipeline_" + PIPELINE_VERSION,
            prompt_hash=self.pipeline.generator.prompt_fingerprint(),
            initial_response=result["initial_response"],
            final_response=result["corrected_response"],
            correction_iterations=result["correction_iterations"],
            api_calls=api_calls,
            latency_seconds=time.monotonic() - started,
            prompt_tokens=prompt_tokens if has_usage else None,
            completion_tokens=completion_tokens if has_usage else None,
            estimated_cost_usd=(estimate_cost(
                model,
                prompt_tokens,
                completion_tokens,
                load_pricing()
            ) if has_usage else None),
            cache_status="recorded_in_stage_metrics"
        ), result


def create_system(
    system,
    max_iterations=2,
    use_structured_feedback=True,
    use_correction_policy=True
):
    system = BaselineSystem(system)

    if system is BaselineSystem.vlm_only:
        return VLMOnlyBaseline()
    if system is BaselineSystem.generic_self_correction:
        return GenericSelfCorrectionBaseline()
    return ProposedPipelineSystem(
        max_iterations=max_iterations,
        use_structured_feedback=use_structured_feedback,
        use_correction_policy=use_correction_policy
    )