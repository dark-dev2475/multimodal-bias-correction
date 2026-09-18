import hashlib
from pathlib import Path

from models.vlm import OpenRouterVLM
from utils.schemas import GenerationMode


PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "initial_generation.txt"
)


# A known-biased response used as a positive control: it guarantees the
# detection stages have something to find, so a clean run proves the
# detectors are silent rather than broken.
ADVERSARIAL_RESPONSE = """
The image shows a disabled patient working in a woodworking workshop.
He appears to be receiving medical treatment and needs assistance.
The other man is a trained professional helping him.
The patient seems uncomfortable and dependent on the other worker.
"""

MODES = tuple(mode.value for mode in GenerationMode)


class InitialGenerator:

    def __init__(self, mode=GenerationMode.real, prompt_path=PROMPT_PATH):

        try:
            self.mode = GenerationMode(mode)
        except ValueError as exc:
            raise ValueError(
                f"Unknown generation mode {mode!r}. "
                f"Expected one of: {', '.join(MODES)}"
            ) from exc

        self.prompt_path = Path(prompt_path)
        self.vlm = OpenRouterVLM()

    def _load_template(self):

        if not self.prompt_path.exists():
            raise FileNotFoundError(
                f"Generation prompt not found: {self.prompt_path}"
            )

        return self.prompt_path.read_text(encoding="utf-8")

    def prompt_fingerprint(self):
        """Hash of whatever produced the initial response.

        Lets a saved run be tied back to the exact prompt text used, so a
        later prompt edit is visible rather than silently confounding results.
        """

        source = (
            ADVERSARIAL_RESPONSE
            if self.mode is GenerationMode.adversarial
            else self._load_template()
        )

        return hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest()[:12]

    def generate(self, image_path, question):

        if self.mode is GenerationMode.adversarial:
            return ADVERSARIAL_RESPONSE.strip()

        # Substitution rather than str.format so that braces added to the
        # prompt later are not mistaken for format fields.
        prompt = self._load_template().replace("{question}", question)

        response = self.vlm.complete(
            prompt,
            image_path=image_path,
            source="Initial generator"
        )

        return response.strip()
