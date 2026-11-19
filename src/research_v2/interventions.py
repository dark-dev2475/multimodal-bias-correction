"""Controlled deletion in a textual evidence ledger, never image editing.

The score measures model sensitivity conditional on this ledger. It is not
a causal effect in the photographed world or a calibrated bias probability.
"""

from dataclasses import asdict, dataclass

from research_v2.contracts import EvidenceGraph, JudgmentBatch


ORDINAL_SUPPORT = {"unsupported": 0.0, "uncertain": 0.5, "supported": 1.0}


@dataclass(frozen=True)
class Intervention:
    cue: str
    target_ids: list[str]
    removed_ids: list[str]
    control_removed_ids: list[str]

    def as_dict(self):
        return asdict(self)


def plan_interventions(graph: EvidenceGraph, limit: int) -> list[Intervention]:
    groups = sorted({o.cue for o in graph.observations if o.cue != "none"})
    plans = []
    for group in groups:
        removed = [o.id for o in graph.observations if o.cue == group]
        targets = [c.id for c in graph.claims if c.kind == "inference"
                   and c.social_assumption
                   and graph.evidence_closure(c.id).intersection(removed)]
        if not targets:
            continue
        used = set().union(*(graph.evidence_closure(cid) for cid in targets))
        # The control has the same number of observations, and removes none
        # of the cited task evidence for these targets. Matching is by count
        # only; semantic equivalence remains an empirical limitation.
        controls = [o.id for o in graph.observations if o.cue == "none" and o.id not in used]
        control = controls[:len(removed)] if len(controls) >= len(removed) else []
        plans.append(Intervention(group, targets, removed, control))
    # Prioritise the groups affecting most claims, with stable tie breaking.
    return sorted(plans, key=lambda p: (-len(p.target_ids), p.cue))[:limit]


def ledger_payload(graph: EvidenceGraph, targets: list[str], removed: list[str]) -> dict:
    # No original draft, image, dependency labels, cue names or deleted text
    # enters a probe. Otherwise the ablated evidence could leak back in.
    return {
        "observations": [{"id": o.id, "text": o.text, "source": o.source}
                         for o in graph.observations if o.id not in removed],
        "claims": [{"id": c.id, "text": c.text} for c in graph.claims if c.id in targets],
    }


def dependence_scores(plan, base: JudgmentBatch, ablated: JudgmentBatch,
                      control: JudgmentBatch | None, matched_control=True):
    b = base.aligned(plan.target_ids)
    a = ablated.aligned(plan.target_ids)
    k = control.aligned(plan.target_ids) if control is not None else None
    rows = []
    for cid in plan.target_ids:
        base_score = ORDINAL_SUPPORT[b[cid].support]
        cue_drop = max(0.0, base_score - ORDINAL_SUPPORT[a[cid].support])
        control_drop = max(0.0, base_score - ORDINAL_SUPPORT[k[cid].support]) if k else None
        comparable = k is not None or not matched_control
        score = max(0.0, cue_drop - (control_drop or 0.0)) if comparable else None
        rows.append({"claim_id": cid, "cue": plan.cue, "cue_drop": cue_drop,
                     "control_drop": control_drop, "dependence_score": score,
                     "status": "measured" if comparable else "no_matched_control"})
    return rows
