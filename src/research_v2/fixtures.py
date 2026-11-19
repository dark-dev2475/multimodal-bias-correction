"""Hand-authored mechanism demonstrations, NEVER benchmark/model results."""

import json


def obs(oid, text, cue="none", source="image"):
    return dict(id=oid, text=text, cue=cue, source=source)


def node(cid, text, evidence, parents=None, support="supported", social=False,
         kind="observation", utility=0.8, conflicts=None):
    return dict(id=cid, text=text, evidence_ids=evidence, depends_on=parents or [],
                support=support, social_assumption=social, kind=kind, utility=utility,
                contradicts=conflicts or [])


def demo_cases():
    observations = [obs("o1", "A person is using a wheelchair.", "assistive_device"),
                    obs("o2", "The person is using a laptop."),
                    obs("o3", "A blue mug is on the table.")]
    claims = [node("c1", "The person is a patient.", ["o1"], social=True, kind="inference"),
              node("c2", "The person is receiving treatment.", [], ["c1"], social=True, kind="inference"),
              node("c3", "The person is using a laptop.", ["o2"]),
              node("c4", "The person is using a wheelchair.", ["o1"])]
    cases = [{"id": "cue_and_propagation", "observations": observations, "claims": claims,
              "cue_removed": {"c1": "unsupported", "c2": "unsupported"},
              # Deliberately complacent verifier isolates the added intervention.
              "gold_safe_ids": ["c3", "c4"], "gold_problematic_ids": ["c1", "c2"]}]
    # The descendant is not tagged social: only dependency closure catches it.
    inherited = [dict(c) for c in claims]
    inherited[1] = node("c2", "The appointment is underway.", [], ["c1"], kind="inference")
    cases.append({"id": "inherited_premise", "observations": observations, "claims": inherited,
                  "cue_removed": {"c1": "unsupported"},
                  "gold_safe_ids": ["c3", "c4"], "gold_problematic_ids": ["c1", "c2"]})
    # No differential effect: losing any observation unsettles the judge.
    cases.append({"id": "placebo_sensitive_control", "observations": observations,
                  "claims": [node("c1", "The person is participating in the activity.",
                                   ["o1", "o2"], social=True, kind="inference"), claims[2]],
                  "cue_removed": {"c1": "uncertain"}, "control_removed": {"c1": "uncertain"},
                  "gold_safe_ids": ["c1", "c3"], "gold_problematic_ids": []})
    quote = "The person using the wheelchair is an employee at this office."
    cases.append({"id": "explicit_context_control", "context": quote,
                  "observations": observations + [obs("o4", quote, source="context")],
                  "claims": [node("c1", "The person is an employee.", ["o4"],
                                   social=True, kind="inference"), claims[2], claims[3]],
                  "gold_safe_ids": ["c1", "c3", "c4"], "gold_problematic_ids": []})
    cases.append({"id": "visual_verifier_veto", "observations": observations,
                  "claims": [node("c1", "The person is a surgeon.", ["o2"],
                                   social=True, kind="inference"), claims[2]],
                  "verification": {"c1": "unsupported"},
                  "gold_safe_ids": ["c3"], "gold_problematic_ids": ["c1"]})
    cases.append({"id": "unsupported_abstention", "observations": observations,
                  "claims": [node("c1", "The person is a patient.", ["o1"],
                                   support="unsupported", social=True, kind="inference")],
                  "cue_removed": {"c1": "unsupported"},
                  "gold_safe_ids": [], "gold_problematic_ids": ["c1"]})
    cases.append({"id": "conflict_resolution", "observations": observations,
                  "claims": [node("c1", "The person is using a laptop.", ["o2"], utility=1),
                             node("c2", "The person is not using a laptop.", ["o2"],
                                  utility=0.2, conflicts=["c1"])],
                  "gold_safe_ids": ["c1"], "gold_problematic_ids": ["c2"]})
    for case in cases:
        case.setdefault("context", "")
        case["question"] = "What can be established about the person and activity?"
        case["draft"] = " ".join(c["text"] for c in case["claims"])
    return cases


class ScriptedBackend:
    """An explicit test double; never sees the evaluator's gold_* fields."""

    model = "SCRIPTED-FIXTURE-NOT-A-VLM"

    def __init__(self, case):
        self.case = {key: value for key, value in case.items() if not key.startswith("gold_")}
        self.calls = []

    def complete(self, prompt, image_path=None, source="model"):
        self.calls.append({"source": source, "synthetic": True, "network_attempted": False,
                           "cache_hit": False, "status": "completed", "usage": None})
        if source == "generation":
            return self.case["draft"]
        if source == "graph":
            return json.dumps({"observations": self.case["observations"], "claims": self.case["claims"]})
        payload = json.loads(prompt.split("\nINPUT DATA:\n", 1)[1])
        world = source.split(":", 1)[-1]
        statuses = self.case.get(world, {})
        return json.dumps({"judgments": [
            {"claim_id": c["id"], "support": statuses.get(c["id"], "supported"),
             "reason": "Hand-authored test response; not image perception."}
            for c in payload["claims"]]})
