"""Validated public evidence artifacts; these are not private chain of thought."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Support = Literal["supported", "uncertain", "unsupported"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Observation(Strict):
    id: str = Field(pattern=r"^o[1-9][0-9]*$")
    text: str = Field(min_length=1)
    source: Literal["image", "context"]
    # A potential stereotype cue is an observable feature, not an inferred
    # protected identity. An assistive device is a cue, not a diagnosis.
    cue: Literal["none", "assistive_device", "appearance", "clothing", "setting"] = "none"


class ClaimNode(Strict):
    id: str = Field(pattern=r"^c[1-9][0-9]*$")
    text: str = Field(min_length=1)
    kind: Literal["observation", "inference"]
    social_assumption: bool
    evidence_ids: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)
    support: Support
    utility: float = Field(ge=0, le=1, allow_inf_nan=False)


class EvidenceGraph(Strict):
    observations: list[Observation] = Field(max_length=32)
    claims: list[ClaimNode] = Field(min_length=1, max_length=18)

    @model_validator(mode="after")
    def validate_graph(self):
        obs = {o.id for o in self.observations}
        nodes = {c.id: c for c in self.claims}
        if len(obs) != len(self.observations) or len(nodes) != len(self.claims):
            raise ValueError("Duplicate graph ids")
        for c in self.claims:
            for refs in (c.evidence_ids, c.depends_on, c.contradicts):
                if len(refs) != len(set(refs)):
                    raise ValueError("Duplicate graph references")
            if not set(c.evidence_ids) <= obs:
                raise ValueError("Unknown evidence reference")
            if not set(c.depends_on + c.contradicts) <= nodes.keys():
                raise ValueError("Unknown claim reference")
            if c.id in c.depends_on + c.contradicts:
                raise ValueError("Self dependency or self contradiction")
        # A dependency graph must be acyclic, regardless of output ordering.
        visiting, visited = set(), set()

        def visit(cid):
            if cid in visiting:
                raise ValueError("Cyclic claim dependency")
            if cid in visited:
                return
            visiting.add(cid)
            for parent in nodes[cid].depends_on:
                visit(parent)
            visiting.remove(cid)
            visited.add(cid)

        for cid in nodes:
            visit(cid)
        return self

    def evidence_closure(self, cid: str) -> set[str]:
        nodes = {c.id: c for c in self.claims}
        seen = set()

        def visit(key):
            seen.update(nodes[key].evidence_ids)
            for parent in nodes[key].depends_on:
                visit(parent)

        visit(cid)
        return seen


class Judgment(Strict):
    claim_id: str
    support: Support
    reason: str = Field(min_length=1)


class JudgmentBatch(Strict):
    judgments: list[Judgment]

    def aligned(self, expected: list[str]) -> dict[str, Judgment]:
        actual = [item.claim_id for item in self.judgments]
        if len(actual) != len(set(actual)) or set(actual) != set(expected):
            raise ValueError("Judgment coverage must match every requested id exactly once")
        return {item.claim_id: item for item in self.judgments}


class V2Config(Strict):
    max_calls: int = Field(default=9, ge=3, le=64)
    max_cue_groups: int = Field(default=2, ge=0, le=4)
    dependence_threshold: float = Field(default=0.5, gt=0, le=1, allow_inf_nan=False)
    risk_weight: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    edit_penalty: float = Field(default=0.1, ge=0, allow_inf_nan=False)
    interventions: bool = True
    dependency_closure: bool = True
    matched_control: bool = True
    visual_verification: bool = True


ABLATIONS = {
    "full": {},
    "no_interventions": {"interventions": False},
    "no_dependency_closure": {"dependency_closure": False},
    "no_matched_control": {"matched_control": False},
    "no_visual_verification": {"visual_verification": False},
}
