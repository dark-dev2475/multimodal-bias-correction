"""Readable per-image reference index; reporting only, no inference or scoring."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "data/evaluation/reference_study_40"


def main():
    inventory = json.loads((STUDY / "inventory.json").read_text(encoding="utf-8"))
    records = []
    lines = ["# Reference annotations — original 40 images", "",
             "AI-assisted drafts, NOT human-approved ground truth. All examples are development data.", "",
             "References were prepared before new response collection. Human reviewers should inspect the image and task without model outputs.", ""]
    for case in inventory["cases"]:
        path = STUDY / "manual_annotations" / (case["example_id"] + ".json")
        record = json.loads(path.read_text(encoding="utf-8"))
        records.append(record)
        ref = record.get("reference")
        lines += ["## " + case["example_id"], "",
                  f"[Image]({Path(case['image_path']).as_posix()}) · [Annotation JSON]({path.as_posix()})", "",
                  f"Status: **{record['status']}**. Author: `{record.get('annotator_model', 'not assigned')}`.", "",
                  "Question: " + case["question"], "", "Context: " + case["context"], ""]
        if ref:
            lines += ["Reference answer: " + ref["reference_answer"], "", "Supported facts:", ""]
            lines += [f"- {f['id']} ({f['source']}): {f['text']}" for f in ref["facts"]]
            lines += ["", "Unsupported inferences to avoid (not assertions of truth):", ""]
            lines += ["- " + s for s in ref["unsupported_social_inferences"]]
            lines += [""]
    out = ROOT / "output/version2/reference_study_40"
    out.mkdir(parents=True, exist_ok=True)
    (out / "REFERENCE_INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "all_reference_annotations.json").write_text(json.dumps({"annotation_type": "ai_assisted_pending_human_review", "cases": records}, indent=2), encoding="utf-8")
    print(f"Exported {len(records)} reference records")


if __name__ == "__main__":
    main()
