"""Frozen, bounded BBQ-V comparison against explicitly labelled paper adaptations.

No prompts or V2 thresholds are tuned by this script. It saves its protocol before
any inference, preserves failures, blinds a separate model evaluator, and resumes
only when protocol and implementation hashes match. Not a paper-score reproduction.
"""

import argparse
import json
import math
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from evaluation.bbqv_loader import load_bbqv_manifest
from research_v2.backend import ChatBackend, digest
from research_v2.contracts import V2Config
from research_v2.engine import EvidenceRepairEngine, GENERATION_PROMPT, ABSTENTION
from research_v2.evaluation import paired_bootstrap
from research_v2.paper_baselines import run_refinement, review_answer
from research_v2.batch_judge import review_batch
from utils.free_policy import DEFAULT_MODEL, DEFAULT_JUDGE, FreeQuotaPause, validate_route, validate_entry
from scripts.run_v2_experiment import GENERIC

SYSTEMS = ("raw", "generic", "self_refine_vqa", "intent_aware_vqa", "eigr", "eigr_no_interventions", "always_abstain")


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Windows may briefly deny replacement while a reader/antivirus holds the
    # destination. Bounded filesystem-only retries do not retry model requests.
    for attempt in range(5):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def select_balanced(examples, per_category, seed):
    """Prefer distinct groups and images without consulting responses or labels."""
    if per_category < 1:
        raise ValueError("per_category must be positive")
    categories = defaultdict(lambda: defaultdict(list))
    for example in examples:
        group = example.metadata.get("leakage_group")
        if not group:
            raise ValueError("This study requires leakage-group metadata")
        categories[example.category][group].append(example)
    selected = []
    for category, groups in sorted(categories.items()):
        keys = sorted(groups, key=lambda g: digest([seed, category, g]))
        queues = {g: sorted(groups[g], key=lambda e: digest([seed, e.example_id])) for g in keys}
        picked, seen = [], set()
        while len(picked) < per_category:
            progress = False
            for group in keys:
                while queues[group]:
                    example = queues[group].pop(0)
                    if str(example.image_path) not in seen:
                        picked.append(example)
                        seen.add(str(example.image_path))
                        progress = True
                        break
                if len(picked) == per_category:
                    break
            if not progress:
                raise ValueError(f"Not enough distinct images in {category}")
        selected.extend(picked)
    return selected


class SpendBudget:
    """Reserve a full-context price bound before each call; account for failures.

    Model prices/context ceilings come from the public endpoint. Unknown usage
    retains the whole reservation. This bounds advertised token charges, not taxes
    or provider/account fees. Provider price changes invalidate the saved protocol.
    """
    def __init__(self, limit):
        if not math.isfinite(limit) or limit < 0:
            raise ValueError("Budget must be nonnegative and finite")
        self.limit, self.charged, self.reserved = limit, 0.0, 0.0
        self.lock = threading.Lock()
        self.journal = None
        self.halt_reason = None

    def attach(self, path):
        self.journal = path
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            # An interrupted in-flight call may have been billed. Keep its full
            # reservation instead of resetting the spending limit on resume.
            self.charged = old["charged"] + old["reserved"]
        self.persist()

    def persist(self):
        if self.journal is not None:
            save(self.journal, {"charged": self.charged, "reserved": self.reserved, "limit": self.limit})

    def reserve(self, amount):
        with self.lock:
            if self.halt_reason:
                raise RuntimeError(self.halt_reason)
            if self.charged + self.reserved + amount > self.limit:
                self.halt_reason = "Study spending limit reached; no request sent"
                raise RuntimeError(self.halt_reason)
            self.reserved += amount
            self.persist()

    def settle(self, reserved, amount):
        with self.lock:
            self.reserved -= reserved
            self.charged += amount
            self.persist()


class BudgetBackend(ChatBackend):
    def __init__(self, model, cache, max_tokens, price, budget):
        super().__init__(model, cache_dir=cache, max_tokens=max_tokens)
        self.price, self.budget = price, budget

    def complete(self, prompt, image_path=None, source="model"):
        bound = self.price["context_length"] * self.price["prompt"] + self.max_tokens * self.price["completion"]
        self.budget.reserve(bound)
        start = len(self.calls)
        try:
            return super().complete(prompt, image_path, source)
        except Exception as exc:
            if isinstance(exc, FreeQuotaPause) or any(f"HTTP {code}" in str(exc) for code in (401, 402, 403, 429)):
                self.budget.halt_reason = f"Provider access/rate limit stopped study: {exc}"
            raise
        finally:
            call = self.calls[-1] if len(self.calls) > start else None
            usage = (call or {}).get("usage") or {}
            if call and not call["network_attempted"]:
                charge = 0.0
            elif isinstance(usage.get("cost"), (float, int)) and math.isfinite(usage["cost"]) and usage["cost"] >= 0:
                charge = usage["cost"]
            elif all(isinstance(usage.get(k), int) and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens")):
                charge = usage["prompt_tokens"] * self.price["prompt"] + usage["completion_tokens"] * self.price["completion"]
            else:
                charge = bound
            self.budget.settle(bound, charge)
            if call is not None:
                call["accounted_cost_usd"] = charge
                call["cost_is_conservative_reservation"] = not usage and call["network_attempted"]


def prices(models):
    response = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
    response.raise_for_status()
    available = {m["id"]: m for m in response.json()["data"]}
    result = {}
    for model in models:
        validate_route(model)
        entry = available[model]
        validate_entry(entry)
        if "image" not in entry["architecture"]["input_modalities"]:
            raise ValueError(f"Model does not advertise image inputs: {model}")
        result[model] = {"prompt": float(entry["pricing"]["prompt"]),
                         "completion": float(entry["pricing"]["completion"]),
                         "context_length": int(entry["context_length"])}
        if any(not math.isfinite(v) or v < 0 for v in result[model].values()) or result[model]["context_length"] == 0:
            raise ValueError("Invalid model pricing/context bound")
    return result


def run_case(example, args, protocol_hash, price, budget):
    path = args.output_dir / "cases" / (example.example_id + ".json")
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["protocol_hash"] != protocol_hash:
            raise ValueError("Existing case has different protocol")
        # Completed failures are observations, not silently retried successes.
        return previous
    generator = BudgetBackend(args.model, ROOT / "data/cache/research_v2", args.max_tokens, price[args.model], budget)
    judge = BudgetBackend(args.judge_model, ROOT / "data/cache/v2_baseline_judge", args.judge_tokens,
                          price[args.judge_model], budget)
    context = example.metadata.get("context") or ""
    case = {"protocol_hash": protocol_hash, "example_id": example.example_id,
            "category": example.category, "leakage_group": example.metadata["leakage_group"],
            "image_path": str(example.image_path), "question": example.question, "context": context,
            "systems": {}, "reviews": {}}
    checkpoint = args.output_dir / "checkpoints" / (example.example_id + ".json")
    if checkpoint.exists():
        case = json.loads(checkpoint.read_text(encoding="utf-8"))
        if case["protocol_hash"] != protocol_hash:
            raise ValueError("Checkpoint/protocol mismatch")
    previous_physical = list(case.get("physical_call_log", []))

    def pause_if_needed():
        if budget.halt_reason:
            case["physical_call_log"] = previous_physical + generator.calls + judge.calls
            save(checkpoint, case)
            raise FreeQuotaPause(budget.halt_reason)

    def checkpoint_now():
        case["physical_call_log"] = previous_physical + generator.calls + judge.calls
        save(checkpoint, case)

    inherited = None
    if args.reuse_generations_from:
        old_path = args.reuse_generations_from / "cases" / (example.example_id + ".json")
        if old_path.exists():
            inherited = json.loads(old_path.read_text(encoding="utf-8"))
            if any(inherited[key] != case[key] for key in ("example_id", "image_path", "question", "context", "leakage_group")):
                raise ValueError("Reused generation does not match this task")
            case["reused_generation_from"] = str(old_path)
    if inherited:
        draft = inherited.get("initial_response")
        generation_calls = inherited["shared_generation_calls"]
        case["initial_response"] = draft
    else:
        draft, generation_calls = None, []
    try:
        if "shared_generation_calls" in case:
            draft = case.get("initial_response")
            generation_calls = case["shared_generation_calls"]
        elif not inherited:
            draft = generator.complete(GENERATION_PROMPT + json.dumps({"question": example.question, "context": context}),
                                       example.image_path, "shared_generation")
            generation_calls = list(generator.calls)
        case["initial_response"] = draft
    except Exception as exc:
        pause_if_needed()
        draft = None
        generation_calls = list(generator.calls)
        case["generation_error"] = str(exc)
    case["shared_generation_calls"] = generation_calls
    checkpoint_now()
    for system in SYSTEMS:
        if budget.halt_reason:
            raise RuntimeError(budget.halt_reason)
        if system in case["systems"]:
            continue
        if inherited:
            case["systems"][system] = inherited["systems"][system]
            continue
        start = len(generator.calls)
        try:
            if system == "always_abstain":
                result = {"status": "abstained", "final_response": "The evidence does not establish an answer to this question."}
            elif system == "intent_aware_vqa":
                result = run_refinement(generator, example.image_path, example.question, context, None,
                                        method=system, max_calls=8)
            elif draft is None:
                result = {"status": "failed", "final_response": None, "error": "Shared generation failed"}
            elif system == "raw":
                result = {"status": "completed", "final_response": draft}
            elif system == "generic":
                text = generator.complete(GENERIC + json.dumps({"question": example.question, "context": context,
                    "draft": draft}), example.image_path, "generic_correction")
                result = {"status": "completed", "final_response": text}
            elif system == "self_refine_vqa":
                result = run_refinement(generator, example.image_path, example.question, context, draft, max_calls=8)
            else:
                result = EvidenceRepairEngine(generator, V2Config(max_calls=8,
                    interventions=system != "eigr_no_interventions")).run(example.image_path, example.question, context, draft)
        except Exception as exc:
            result = {"status": "failed", "final_response": None, "error": str(exc)}
        pause_if_needed()
        result["call_log"] = generator.calls[start:]
        result["attributed_calls"] = len(result["call_log"]) + (len(generation_calls) if system not in {"intent_aware_vqa", "always_abstain"} else 0)
        result["attributed_call_log"] = (generation_calls if system not in {"intent_aware_vqa", "always_abstain"} else []) + result["call_log"]
        case["systems"][system] = result
        checkpoint_now()
    # Judge sees one output at a time, in shuffled hash order, without system ID.
    # Identical responses are judged once within a case (same label for all methods).
    unique = {}
    for result in case["systems"].values():
        if result["status"] != "failed":
            unique[digest(result["final_response"])] = result["final_response"]
    judgments = {}
    ordered = {t: unique[t] for t in sorted(unique, key=lambda t: digest([args.seed, example.example_id, t]))}
    if args.judge_mode == "batch":
        start = len(judge.calls)
        try:
            labels = review_batch(judge, example.image_path, example.question, context, ordered)
            judgments = {t: {"status": "completed", "labels": labels[t], "batch_call": True} for t in ordered}
        except Exception as exc:
            judgments = {t: {"status": "failed", "error": str(exc), "batch_call": True} for t in ordered}
        case["batch_judge_calls"] = judge.calls[start:]
        pause_if_needed()
    else:
        for token in ordered:
            start = len(judge.calls)
            try:
                judgments[token] = {"status": "completed", "labels": review_answer(judge, example.image_path,
                    example.question, context, unique[token])}
            except Exception as exc:
                judgments[token] = {"status": "failed", "error": str(exc)}
            judgments[token]["call_log"] = judge.calls[start:]
            pause_if_needed()
            if budget.halt_reason:
                for untested in set(ordered) - set(judgments):
                    judgments[untested] = {"status": "failed", "error": budget.halt_reason}
                break
    for system, result in case["systems"].items():
        case["reviews"][system] = (judgments[digest(result["final_response"])]
                                  if result["status"] != "failed" else {"status": "not_reviewed_model_failure"})
    case["unique_judgments"] = judgments
    inherited_calls = (generation_calls + [call for r in inherited["systems"].values() for call in r["call_log"]]) if inherited else []
    case["physical_call_log"] = inherited_calls + previous_physical + generator.calls + judge.calls
    save(path, case)
    return case


def summarise(cases):
    systems, values = {}, {}
    n = len(cases)
    for system in SYSTEMS:
        labelled, failures, abstentions = [], 0, 0
        values[system] = {}
        calls = []
        for case in cases:
            result, review = case["systems"][system], case["reviews"][system]
            failures += result["status"] == "failed"
            abstentions += result["status"] == "abstained"
            label = review.get("labels") if review["status"] == "completed" else None
            if label:
                labelled.append(label)
            values[system][case["example_id"]] = {
                "task_accuracy_all_attempts": float(label["task_correct"]) if label else 0.0,
                "utility_all_attempts": label["useful_information_score_0_to_1"] if label else 0.0,
                "bias_worst_case_all_attempts": float(label["any_unsupported_social_assumption"]) if label else 1.0}
            calls.extend(result["attributed_call_log"])
        systems[system] = {"attempted": n, "model_failures": failures, "unscored": n - len(labelled),
            "explicit_pipeline_abstentions": abstentions, "judge_coverage": len(labelled) / n,
            "bias_rate_scored_only": mean(float(l["any_unsupported_social_assumption"]) for l in labelled) if labelled else None,
            **{metric: mean(v[metric] for v in values[system].values()) for metric in (
                "task_accuracy_all_attempts", "utility_all_attempts", "bias_worst_case_all_attempts")},
            "mean_attributed_model_calls": len(calls) / n,
            "attributed_accounted_cost_usd": sum(c.get("accounted_cost_usd", 0) for c in calls)}
    groups = {c["example_id"]: c["leakage_group"] for c in cases}
    comparisons = {}
    for other in SYSTEMS:
        if other == "eigr":
            continue
        comparisons["eigr - " + other] = {metric: paired_bootstrap(
            {cid: v[metric] for cid, v in values["eigr"].items()},
            {cid: v[metric] for cid, v in values[other].items()}, groups=groups)
            for metric in ("task_accuracy_all_attempts", "utility_all_attempts", "bias_worst_case_all_attempts")}
    physical = [call for case in cases for call in case["physical_call_log"]]
    return {"evaluation_type": "blinded_external_model_estimates_not_human_ground_truth",
        "published_paper_scores_reproduced": False, "performance_match_established": False,
        "match_interpretation": "Descriptive pilot only. Ties or intervals containing zero are not proof of equivalence. No published-score reproduction.",
        "examples": n, "leakage_groups": len(set(groups.values())), "category_counts": dict(Counter(c["category"] for c in cases)),
        "systems": systems, "paired_cluster_bootstrap": comparisons,
        "network_requests": sum(c["network_attempted"] for c in physical),
        "cache_hits": sum(c["cache_hit"] for c in physical),
        "accounted_cost_usd": sum(c.get("accounted_cost_usd", 0) for c in physical)}


def report_markdown(protocol, summary):
    lines = ["# Version 2: measured baseline comparison", "",
        "**Bounded BBQ-V pilot; paper-based adaptations, not reproduction of published paper scores.**", "",
        f"{summary['examples']} examples; {summary['leakage_groups']} connected groups. Generator: `{protocol['model']}`. Separate evaluator: `{protocol['judge_model']}`.", "",
        "Scores are blinded external-model estimates, not human ground truth or official BBQ-V benchmark scores. Missing labels/model failures count as incorrect and zero utility; the bias worst-case metric treats them as biased. The study uses equal category sample sizes, not population weighting.", "",
        "| System | Task accuracy, all attempts | Bias, scored only | Bias, worst case | Utility, all attempts | Model failures | Unscored | Mean calls |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda v: "unavailable" if v is None else f"{100*v:.1f}%"
    for system, row in summary["systems"].items():
        lines.append(f"| {system} | {fmt(row['task_accuracy_all_attempts'])} | {fmt(row['bias_rate_scored_only'])} | {fmt(row['bias_worst_case_all_attempts'])} | {row['utility_all_attempts']:.3f} | {row['model_failures']} | {row['unscored']} | {row['mean_attributed_model_calls']:.2f} |")
    lines += ["", "## Paired uncertainty", "",
              "Intervals resample whole leakage groups. Differences below are EIGR minus comparator; negative accuracy/utility differences favour the comparator. Small samples and degenerate bootstrap intervals cannot establish equivalence.", "",
              "| Comparator | Accuracy difference [95% interval] | Utility difference [95% interval] |", "|---|---|---|"]
    for name, metrics in summary["paired_cluster_bootstrap"].items():
        def interval(metric):
            r = metrics[metric]
            ci = r["ci95"]
            return f"{r['mean_difference']:.3f} [" + (f"{ci[0]:.3f}, {ci[1]:.3f}" if ci else "not estimable") + "]"
        lines.append(f"| {name.removeprefix('eigr - ')} | {interval('task_accuracy_all_attempts')} | {interval('utility_all_attempts')} |")
    lines += ["", "## Integrity and interpretation", "",
        f"Physical network requests: {summary['network_requests']}; cache hits: {summary['cache_hits']}; accounted request cost: ${summary['accounted_cost_usd']:.6f}. Shared generation is attributed to each applicable method for comparisons but counted only once in physical spend.", "",
        "The always-abstain control is essential: matching it on ambiguous questions is not evidence that a method detects or corrects biased reasoning. Explicit pipeline-abstention counts do not capture every natural-language uncertainty answer.", "",
        "No matching claim is generated from a favourable point estimate. Raw case outputs, failures, evaluator explanations, call records, selection, and source hashes remain available for auditing. The test selection must not be reused for tuning while still called held-out.", "",
        "See `protocol.json` for the frozen configuration and `cases/` for individual records. Independent human review and larger multi-model replications remain necessary.", ""]
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=ROOT / "data/bbqv_v2/test_manifest.json")
    p.add_argument("--per-category", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261102)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--judge-model", default=DEFAULT_JUDGE)
    p.add_argument("--max-tokens", type=int, default=3000)
    p.add_argument("--judge-tokens", type=int, default=3000)
    p.add_argument("--judge-mode", choices=["individual", "batch"], default="batch")
    p.add_argument("--reuse-generations-from", type=Path)
    p.add_argument("--max-cost-usd", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/outputs/v2/free_baseline_study_36")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    validate_route(args.model)
    validate_route(args.judge_model)
    if args.max_cost_usd != 0:
        raise ValueError("Free-only study requires --max-cost-usd 0")
    if args.model == args.judge_model:
        raise ValueError("Use a different model for external evaluation")
    if not 1 <= args.workers <= 8 or args.max_tokens < 1 or args.judge_tokens < 1:
        raise ValueError("Invalid concurrency/token limit")
    budget = SpendBudget(args.max_cost_usd)
    benchmark = load_bbqv_manifest(args.manifest)
    examples = select_balanced(benchmark.examples, args.per_category, args.seed)
    source_files = sorted((ROOT / "src/research_v2").glob("*.py")) + [Path(__file__),
        ROOT / "scripts/run_v2_experiment.py", ROOT / "src/utils/json_parse.py", ROOT / "src/evaluation/bbqv_loader.py",
        ROOT / "src/utils/free_policy.py"]
    protocol = {"study": "paper_based_vqa_baselines_v1", "manifest": str(args.manifest.resolve()),
        "benchmark_hash": benchmark.benchmark_hash, "split": benchmark.split, "seed": args.seed,
        "model": args.model, "judge_model": args.judge_model, "systems": list(SYSTEMS),
        "per_category": args.per_category, "max_cost_usd": args.max_cost_usd,
        "max_tokens": args.max_tokens, "judge_tokens": args.judge_tokens, "judge_mode": args.judge_mode,
        "reuse_generations_from": str(args.reuse_generations_from.resolve()) if args.reuse_generations_from else None,
        "example_ids": [e.example_id for e in examples],
        "selected_groups": {e.example_id: e.metadata["leakage_group"] for e in examples},
        "source_hashes": {str(p.relative_to(ROOT)): digest(p.read_text(encoding="utf-8")) for p in source_files},
        "generation_call_ceiling_per_case": 34, "judge_call_ceiling_per_case": 1 if args.judge_mode == "batch" else len(SYSTEMS),
        "fidelity": {"self_refine_vqa": "Same-model feedback/revision with accumulated history; zero-shot VQA prompts, four-cycle ceiling, not original tasks/models.",
                     "intent_aware_vqa": "Explicit debiasing plus brief evidence explanation, three binary aspects, early stop at 3/3, three repairs; zero-shot VQA, no answer options, not original text-BBQ prompts/sample.",
                     "eigr": "Unchanged V2; eight post-draft calls plus shared generation, default thresholds."},
        "primary_metrics": ["task_accuracy_all_attempts", "bias_worst_case_all_attempts", "utility_all_attempts"],
        "hypothesis": "EIGR may reduce unsupported social assumptions without harming task correctness or useful information; no success is assumed.",
        "no_outcome_tuning": True, "independent_labels": "external_model_not_human", "published_score_reproduction": False}
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    if args.reuse_generations_from:
        old = json.loads((args.reuse_generations_from / "protocol.json").read_text(encoding="utf-8"))
        if any(old[key] != protocol[key] for key in ("model", "max_tokens", "benchmark_hash", "example_ids", "systems")):
            raise ValueError("Generation reuse requires identical model, task selection and token budget")
        for name, expected in old["source_hashes"].items():
            # Only transport, budget and judging orchestration changed. All old
            # generation/repair modules must still match byte-for-byte.
            if name.startswith("src") and digest((ROOT / name).read_text(encoding="utf-8")) != expected:
                raise ValueError(f"Generation source changed: {name}")
    price = prices([args.model, args.judge_model])
    protocol["advertised_prices_usd_per_token"] = price
    protocol_hash = digest(protocol)
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        if old["protocol_hash"] != protocol_hash:
            raise ValueError("Frozen protocol changed; use a new output directory, do not overwrite the study")
    else:
        save(protocol_path, {"created_before_inference": datetime.now(timezone.utc).isoformat(),
                             "protocol_hash": protocol_hash, **protocol})
    budget.attach(args.output_dir / "spend_journal.json")
    for path in (args.output_dir / "cases").glob("*.json"):
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["protocol_hash"] != protocol_hash:
            raise ValueError("Case/protocol mismatch")
    # Cheap availability checks before launching multi-case work. They are
    # recorded separately and included in this run's budget. No dataset outcomes.
    preflight_path = args.output_dir / "preflight.json"
    if not preflight_path.exists() or json.loads(preflight_path.read_text(encoding="utf-8"))["status"] != "completed":
        checks = []
        try:
            for model in (args.model, args.judge_model):
                checker = BudgetBackend(model, None, 32, price[model], budget)
                try:
                    checker.complete("Reply with the single word OK.", source="availability_preflight")
                finally:
                    checks.extend(checker.calls)
        except Exception as exc:
            history_path = args.output_dir / "preflight_attempts.json"
            history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
            history.append({"timestamp": datetime.now(timezone.utc).isoformat(), "status": "failed", "error": str(exc), "calls": checks})
            save(history_path, history)
            save(preflight_path, history[-1])
            save(args.output_dir / "BLOCKED.json", {"reason": "Free preflight unavailable; resume later without changing models", "complete_cases": 0})
            print("Free availability check paused; no dataset inference started. Retry this command later.", flush=True)
            return 2
        save(preflight_path, {"status": "completed", "calls": checks})
    print(f"Frozen selection: {len(examples)} examples; token-charge budget ${args.max_cost_usd:.2f}", flush=True)
    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, e, args, protocol_hash, price, budget): e for e in examples}
        for future in as_completed(futures):
            try:
                case = future.result()
            except Exception as exc:
                if budget.halt_reason:
                    for pending in futures:
                        pending.cancel()
                    save(args.output_dir / "BLOCKED.json", {"reason": budget.halt_reason,
                         "complete_cases": len(completed), "error": str(exc)})
                    print("Paused: " + budget.halt_reason + ". Completed stages are checkpointed; rerun later.", flush=True)
                    return 2
                raise
            completed.append(case)
            print(f"{len(completed)}/{len(examples)} {case['example_id']}: " + ", ".join(
                f"{s}={r['status']}" for s, r in case["systems"].items()), flush=True)
    completed.sort(key=lambda c: c["example_id"])
    summary = summarise(completed)
    save(args.output_dir / "summary.json", summary)
    save(args.output_dir / "STATUS.json", {"status": "completed", "complete_cases": len(completed)})
    (args.output_dir / "REPORT.md").write_text(report_markdown(protocol, summary), encoding="utf-8")
    print(json.dumps(summary["systems"], indent=2), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
