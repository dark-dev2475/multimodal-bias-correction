"""Exact bounded binary optimisation over claims, executed entirely locally."""

from research_v2.contracts import EvidenceGraph, V2Config


def repair_graph(graph: EvidenceGraph, scores: dict[str, float], config: V2Config,
                 blocked: dict[str, str] | None = None) -> dict:
    """Maximise sum x_i*(utility_i-risk_weight*risk_i+edit_penalty).

    Constraints: no unsupported/uncertain node or node without evidence;
    x_child <= x_parent; x_i+x_j <= 1 for contradictions; risk below threshold.
    Enumeration is exact up to the schema limit of 18 claims (262144 subsets).
    The +edit_penalty term is equivalent, up to a constant, to penalising
    every deleted draft claim. No new natural-language assertions are made.
    """
    nodes = graph.claims
    index = {node.id: i for i, node in enumerate(nodes)}
    unknown = (set(scores) | set(blocked or {})) - index.keys()
    if unknown:
        raise ValueError(f"Unknown claim ids in repair input: {sorted(unknown)}")
    for value in scores.values():
        if not 0 <= value <= 1:
            raise ValueError("Dependence scores must be finite values in [0, 1]")
    reasons = dict(blocked or {})
    for node in nodes:
        if node.support != "supported":
            reasons.setdefault(node.id, "not_supported_by_graph")
        if not graph.evidence_closure(node.id):
            reasons.setdefault(node.id, "no_evidence_path")
        if scores.get(node.id, 0) >= config.dependence_threshold:
            reasons.setdefault(node.id, "cue_dependence")
    if config.dependency_closure:
        changed = True
        while changed:
            changed = False
            for node in nodes:
                failed = [p for p in node.depends_on if p in reasons]
                if failed and node.id not in reasons:
                    reasons[node.id] = "blocked_premise:" + ",".join(failed)
                    changed = True
    allowed = sum(1 << index[n.id] for n in nodes if n.id not in reasons)
    dependencies = [sum(1 << index[p] for p in n.depends_on) for n in nodes]
    conflicts = [sum(1 << index[p] for p in n.contradicts) for n in nodes]
    weights = [n.utility - config.risk_weight * scores.get(n.id, 0)
               + config.edit_penalty for n in nodes]
    best_mask, best_value, feasible, visited = 0, 0.0, 0, 0
    mask = allowed
    while True:
        visited += 1
        active = [i for i in range(len(nodes)) if mask & (1 << i)]
        valid = all(not (mask & conflicts[i]) and
                    (not config.dependency_closure or mask & dependencies[i] == dependencies[i])
                    for i in active)
        if valid:
            feasible += 1
            value = sum(weights[i] for i in active)
            if value > best_value + 1e-12 or (
                abs(value - best_value) <= 1e-12 and mask.bit_count() > best_mask.bit_count()
            ):
                best_mask, best_value = mask, value
        if mask == 0:
            break
        mask = (mask - 1) & allowed
    selected = [n.id for i, n in enumerate(nodes) if best_mask & (1 << i)]
    for node in nodes:
        if node.id not in selected:
            reasons.setdefault(node.id, "objective_or_conflict")
    return {"selected_ids": selected, "removed": reasons,
            "objective": round(best_value, 8), "optimal_for_supplied_graph": True,
            "subsets_examined": visited, "feasible_subsets": feasible,
            "dependency_closure_enforced": config.dependency_closure}


def render_claims(graph: EvidenceGraph, selected: list[str]) -> str:
    return " ".join(c.text.rstrip(" .") + "." for c in graph.claims if c.id in selected)
