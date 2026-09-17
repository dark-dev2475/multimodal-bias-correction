import json
from pathlib import Path
from datetime import datetime


class ExperimentLogger:

    def __init__(self, output_dir="data/outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_initial_response(
        self,
        case_id,
        image_path,
        question,
        model,
        response
    ):

        result = {
            "case_id": case_id,
            "image": str(image_path),
            "question": question,
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "initial_response": response
        }

        output_file = self.output_dir / f"{case_id}_initial.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                result,
                f,
                indent=4,
                ensure_ascii=False
            )

        return output_file


