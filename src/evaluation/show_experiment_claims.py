import json
import glob
import os


files = [
    f
    for f in glob.glob(
        "data/outputs/bbqv_experiment/*/*/*.json"
    )
    if "_cache" not in f
]


for file_path in sorted(files):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "result" not in data:
        continue

    result = data["result"]

    print()
    print("=" * 100)
    print(f"EXAMPLE : {data['example_id']}")
    print(f"SYSTEM  : {data['system']}")
    print(f"CATEGORY: {data['category']}")
    print(f"QUESTION: {data['question']}")
    print("=" * 100)

    print("\n--- INITIAL RESPONSE ---")
    print(result.get("initial_response", ""))

    print("\n--- FINAL RESPONSE ---")
    print(result.get("final_response", ""))

    # Proposed system has structured claims
    if "pipeline_result" in data:
        print("\n--- EXTRACTED CLAIMS ---")

        for i, claim in enumerate(
            data["pipeline_result"].get("claims", []),
            1,
        ):
            print(
                f"{i}. [{claim.get('type')}] "
                f"{claim.get('claim')}"
            )