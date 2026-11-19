"""Run EIGR V2 on one image, or execute zero-network mechanism demonstrations."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_v2.backend import ChatBackend
from utils.free_policy import DEFAULT_MODEL
from research_v2.contracts import ABLATIONS, V2Config
from research_v2.engine import EvidenceRepairEngine
from research_v2.evaluation import score_selection
from research_v2.fixtures import ScriptedBackend, demo_cases


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_demo(output):
    rows = []
    for case in demo_cases():
        for ablation, options in ABLATIONS.items():
            result = EvidenceRepairEngine(ScriptedBackend(case), V2Config(**options)).run(
                ROOT / "data/raw/images/wheelchair.jpg", case["question"], case["context"])
            metrics = score_selection(result, case["gold_safe_ids"], case["gold_problematic_ids"])
            rows.append({"case_id": case["id"], "ablation": ablation, "result": result,
                         "fixture_metrics": metrics})
    payload = {"evaluation_type": "hand_authored_mechanism_fixtures", "real_model_results": False,
               "warning": "These fixtures test software behavior; they do not measure bias reduction on real images.",
               "network_calls": sum(r["result"]["network_calls"] for r in rows), "cases": rows}
    save(output, payload)
    report = ["# Version 2 mechanism demonstration", "", payload["warning"], "",
              "Seven scripted cases × five configurations. No model or network is used.", "",
              "| Case | Configuration | Status | Kept IDs | Residual problematic fixture claims |",
              "|---|---|---|---|---|"]
    for row in rows:
        result = row["result"]
        report.append(f"| {row['case_id']} | {row['ablation']} | {result['status']} | "
                      f"{', '.join(result.get('repair', {}).get('selected_ids', [])) or 'none'} | "
                      f"{row['fixture_metrics'].get('residual_problematic_count', 'unavailable')} |")
    output.with_suffix(".md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Scripted, zero-network mechanism checks")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--question", default="What is happening in this image?")
    parser.add_argument("--context", default="")
    parser.add_argument("--initial-response", help="Optional draft; explicitly recorded as supplied")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default="https://openrouter.ai/api/v1")
    parser.add_argument("--max-calls", type=int, default=9)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--ablation", choices=list(ABLATIONS), default="full")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "data/outputs/v2/run.json")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.demo:
        result = run_demo(args.output)
        print(f"Scripted demonstrations: {len(result['cases'])}; network calls: 0")
        return 0
    if args.image is None:
        raise SystemExit("--image is required unless --demo is used")
    backend = ChatBackend(args.model, args.endpoint,
                          None if args.no_cache else ROOT / "data/cache/research_v2",
                          max_tokens=args.max_tokens)
    result = EvidenceRepairEngine(backend, V2Config(max_calls=args.max_calls, **ABLATIONS[args.ablation])).run(
        args.image, args.question, args.context, args.initial_response)
    if any(c.get("error_type") == "FreeQuotaPause" for c in backend.calls):
        result["status"] = "paused"
        result["pause_reason"] = "Free endpoint/quota unavailable. Rerun later; no paid fallback."
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    save(args.output, result)
    print(result["final_response"])
    print(f"Status: {result['status']}; model calls: {result['model_calls']}; network: {result['network_calls']}")
    print(f"Saved: {args.output}")
    return 2 if result["status"] == "paused" else (1 if result["status"] == "failed" else 0)


if __name__ == "__main__":
    raise SystemExit(main())
