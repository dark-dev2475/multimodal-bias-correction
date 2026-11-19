"""Reference and review contracts only; no model client or pipeline imports."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


class Fact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: Literal["image", "context"]


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    facts: list[Fact] = Field(min_length=1, max_length=12)
    unsupported_social_inferences: list[str] = Field(max_length=8)
    answerability: Literal["answerable", "insufficient_evidence"]
    reference_answer: str = Field(min_length=1)
    required_fact_ids: list[str]
    uncertainty_reason: str

    @model_validator(mode="after")
    def aligned(self):
        ids = [f.id for f in self.facts]
        if len(ids) != len(set(ids)) or len(self.required_fact_ids) != len(set(self.required_fact_ids)):
            raise ValueError("Duplicate fact identifiers")
        if not set(self.required_fact_ids) <= set(ids):
            raise ValueError("Unknown required fact")
        if self.answerability == "answerable" and not self.required_fact_ids:
            raise ValueError("Answerable question requires at least one supporting fact")
        if self.answerability == "insufficient_evidence" and not self.uncertainty_reason.strip():
            raise ValueError("Insufficient evidence requires a reason")
        return self


class SentenceReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit_id: str
    support: Literal["supported", "unsupported", "uncertain"]
    asserts_unsupported_social_assumption: StrictBool
    explanation: str = Field(min_length=1)


class SemanticReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    units: list[SentenceReview] = Field(min_length=1)
    covered_required_fact_ids: list[str]
    task_correct: StrictBool
    abstains_from_answer: StrictBool
    explanation: str = Field(min_length=1)

    def validate_coverage(self, units, reference):
        ids = [u.unit_id for u in self.units]
        if len(ids) != len(set(ids)) or set(ids) != set(units):
            raise ValueError("Review must cover every answer unit exactly once")
        covered = self.covered_required_fact_ids
        if len(covered) != len(set(covered)) or not set(covered) <= set(reference.required_fact_ids):
            raise ValueError("Unknown/duplicate required facts in review")
        return self
