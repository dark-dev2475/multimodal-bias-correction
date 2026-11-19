"""Separate FREE local experiment, original 40 cases, unchanged EIGR algorithms.

Reference drafting/approval, inference, blind judging and offline scoring remain
separate. This runner never changes the hosted study or labels AI work as human.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.run_reference_study import (read, save, now, digest, sources, task, make_units,
    validate_reference, SYSTEMS, JUDGE, GENERATION_PROMPT, EvidenceRepairEngine, V2Config,
    run_refinement, run_additional, Reference, SemanticReview, parse_model_json)
from local_experiment.backend import LocalBackend, LocalInfrastructureError

OUT = ROOT / "data/evaluation/reference_study_40_local"
SOURCE = ROOT / "data/evaluation/reference_study_40"
CACHE = ROOT / "data/cache/reference_study_local"


def local_sources():
    values = sources()
    paths = [Path(__file__), ROOT / "scripts/setup_local_vlm.py", ROOT / "scripts/start_local_vlm.ps1",
             ROOT / "scripts/evaluate_local_reference_study.py"]
    paths += list((ROOT / "src/local_experiment").glob("*.py"))
    values.update({str(p.relative_to(ROOT)): digest(p.read_text(encoding="utf-8")) for p in paths})
    return values


def initialize(out):
    if (out / "protocol.json").exists():
        frozen(out)
        print("Existing local protocol verified; nothing overwritten.")
        return
    installation = read(ROOT / "data/local_vlm/installation.json")
    server = read(ROOT / "data/local_vlm/server_config.json")
    inventory = read(SOURCE / "inventory.json")
    references = read(SOURCE / "frozen_references.json")
    parent = read(SOURCE / "protocol.json")
    if digest(references) != parent["reference_hash"] or parent["cases"] != inventory["cases"]:
        raise ValueError("Original frozen reference identity mismatch")
    for case in inventory["cases"]:
        ref = references[case["example_id"]]
        validate_reference(ref)
        if any(ref[k] != case[k] for k in case):
            raise ValueError("Reference task/image mismatch")
        save(out / "manual_annotations" / (case["example_id"] + ".json"), ref)
    protocol = {"version": "reference-study-local-v1", "scope": inventory["scope"],
        "cases": inventory["cases"], "systems": SYSTEMS, "model": installation["model_alias"],
        "installation": installation, "server": server, "reference_quality": parent["reference_quality"],
        "reference_authors": parent["reference_authors"], "reference_hash": digest(references),
        "source_hashes": local_sources(), "max_calls_including_draft": 9,
        "max_output_tokens_per_call": 3000, "temperature": 0, "seed": 42,
        "structured_decoding": "llama.cpp JSON schema from existing prompts, uniformly for all systems",
        "structural_schema_specialization": "Observation source first; empty context forbids context source; context quotes restricted to full context or exact sentences; judgment IDs/counts bound to requested IDs. Does not verify visual truth.",
        "algorithm_changes": "EIGR selection/intervention algorithms unchanged; task-derived structural decoding/model differ from hosted study",
        "collection_order": "system-major, shared raw drafts then EIGR then four controls then abstention",
        "judge_model": installation["model_alias"], "judge_independence": False,
        "review_warning": "Self-judged AI estimates; generator and judge share weights. Not independent validation.",
        "published_scores_comparable": False, "parent_reference_protocol_hash": parent["protocol_hash"],
        "primary_metrics": parent["primary_metrics"], "secondary_metrics": parent["secondary_metrics"]}
    protocol["protocol_hash"] = digest(protocol)
    save(out / "inventory.json", inventory)
    save(out / "frozen_references.json", references)
    save(out / "protocol.json", protocol)
    print("Frozen separate local study with 40 pre-existing AI reference drafts.", flush=True)


def frozen(out):
    p = read(out / "protocol.json")
    if digest({k: v for k, v in p.items() if k != "protocol_hash"}) != p["protocol_hash"]:
        raise ValueError("Protocol hash mismatch")
    if p["source_hashes"] != local_sources():
        raise ValueError("Source changed after freeze; use a new version")
    if p["server"] != read(ROOT / "data/local_vlm/server_config.json"):
        raise ValueError("Server configuration changed after freeze")
    for case in p["cases"]:
        if hashlib.sha256(Path(case["image_path"]).read_bytes()).hexdigest() != case["image_sha256"]:
            raise ValueError("Image changed after freeze")
    return p


def collect(out, limit):
    # NO reference files or answer labels are opened in collection.
    p, backend, count = frozen(out), LocalBackend(CACHE), 0
    for system in p["systems"]:
        for case in p["cases"]:
            directory = out / "responses" / case["example_id"]
            path = directory / (system + ".json")
            if path.exists():
                if read(path)["protocol_hash"] != p["protocol_hash"]:
                    raise ValueError("Saved output protocol mismatch")
                continue
            if count >= limit:
                return
            draft_path = directory / "shared_draft.json"
            if draft_path.exists():
                draft = read(draft_path)
                if draft["protocol_hash"] != p["protocol_hash"]:
                    raise ValueError("Saved draft protocol mismatch")
            else:
                start = len(backend.calls)
                answer = backend.complete(GENERATION_PROMPT + json.dumps(task(case)), case["image_path"], "shared_generation")
                draft = {"answer": answer, "calls": backend.calls[start:], "protocol_hash": p["protocol_hash"]}
                save(draft_path, draft)
            start = len(backend.calls)
            if system == "raw":
                result = {"status": "completed", "final_response": draft["answer"]}
            elif system == "always_abstain":
                result = {"status": "abstained", "final_response": "The available evidence does not establish an answer."}
            elif system == "eigr":
                result = EvidenceRepairEngine(backend, V2Config(max_calls=8)).run(case["image_path"], case["question"], case["context"], draft["answer"])
            elif system in {"self_refine_vqa", "intent_aware_vqa"}:
                result = run_refinement(backend, case["image_path"], case["question"], case["context"], draft["answer"], method=system, max_calls=8)
            else:
                result = run_additional(backend, case["image_path"], case["question"], case["context"], draft["answer"], method=system, max_calls=8)
            calls = backend.calls[start:]
            if any(c.get("error_type") == "LocalInfrastructureError" for c in calls):
                raise LocalInfrastructureError("Server interrupted; completed requests cached, no quality score assigned")
            result.update(protocol_hash=p["protocol_hash"], example_id=case["example_id"], system=system,
                          generator_model=p["model"], calls=calls,
                          shared_draft_calls=[] if system in {"intent_aware_vqa", "always_abstain"} else draft["calls"])
            save(path, result)
            print(f"Collected {case['example_id']} / {system}: {result['status']}", flush=True)
            count += 1


def review(out, limit):
    p, backend, count = frozen(out), LocalBackend(ROOT / "data/cache/reference_judge_local"), 0
    references = read(out / "frozen_references.json")
    if digest(references) != p["reference_hash"]:
        raise ValueError("References changed after freeze")
    for case in p["cases"]:
        ref = Reference.model_validate(references[case["example_id"]]["reference"])
        for system in sorted(p["systems"], key=lambda s: digest([case["example_id"], s])):
            source = out / "responses" / case["example_id"] / (system + ".json")
            if not source.exists():
                continue
            result = read(source)
            if result["protocol_hash"] != p["protocol_hash"]:
                raise ValueError("Output protocol mismatch")
            if result["status"] == "failed" or not result.get("final_response"):
                continue
            answer = result["final_response"]
            token = digest([p["protocol_hash"], case["example_id"], answer])[:24]
            units = make_units(answer)
            form = {"review_id": token, **task(case), "image_path": case["image_path"],
                    "reference": ref.model_dump(), "answer": answer, "units": units,
                    "labels": None, "reviewer": None, "reviewed_at": None,
                    "annotation_origin": "pending_human_review"}
            if not (out / "blind_review" / (token + ".json")).exists():
                save(out / "blind_review" / (token + ".json"), form)
            path = out / "ai_reviews" / (token + ".json")
            error_path = out / "review_errors" / (token + ".json")
            if path.exists() or error_path.exists() or count >= limit:
                continue
            prompt_data = {**task(case), "reference": ref.model_dump(), "answer": answer, "units": units}
            start = len(backend.calls)
            try:
                response = backend.complete(JUDGE + "\nSCHEMA:\n" + json.dumps(SemanticReview.model_json_schema()) +
                    "\nINPUT DATA:\n" + json.dumps(prompt_data), case["image_path"], "blind_reference_review")
                labels = SemanticReview.model_validate(parse_model_json(response, "review")).validate_coverage(units, ref)
                save(path, {**form, "labels": labels.model_dump(), "reviewer": p["judge_model"], "reviewed_at": now(),
                    "annotation_origin": "ai_assisted", "warning": p["review_warning"], "calls": backend.calls[start:]})
                print("Self-judged AI review: " + token, flush=True)
            except LocalInfrastructureError:
                raise
            except Exception as exc:
                save(error_path, {"review_id": token, "error_type": type(exc).__name__, "message": str(exc)[:500],
                    "at": now(), "calls": backend.calls[start:]})
                print("Review failed validation: " + token, flush=True)
            count += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["init", "collect", "review"])
    parser.add_argument("--study", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, default=280)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    try:
        if args.stage == "init":
            initialize(args.study)
        else:
            globals()[args.stage](args.study, args.limit)
    except LocalInfrastructureError as exc:
        save(args.study / "PAUSED.json", {"at": now(), "stage": args.stage, "reason": str(exc)})
        print(str(exc), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
