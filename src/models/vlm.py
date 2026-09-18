import os
import time
import base64
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)


load_dotenv()


# Transient failures worth retrying. Auth and bad-request errors are not
# included on purpose: retrying those just burns quota.
RETRYABLE_ERRORS = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)


class OpenRouterVLM:
    """
    Wrapper around an OpenRouter vision-language model.

    Supports:
        - image + text
        - text only
    """

    def __init__(
        self,
        model_name="inclusionai/ling-3.0-flash-vl:free"
    ):

        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in .env"
            )

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        self.model_name = model_name

    def _encode_image(self, image_path):

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

        extension = image_path.suffix.lower()

        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp"
        }

        mime_type = mime_types.get(
            extension,
            "image/jpeg"
        )

        return f"data:{mime_type};base64,{encoded}"

    def _build_content(self, prompt, image_path):

        # -----------------------------
        # TEXT ONLY
        # -----------------------------

        if image_path is None:
            return prompt

        # -----------------------------
        # IMAGE + TEXT
        # -----------------------------

        return [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": self._encode_image(image_path)
                }
            }
        ]

    def complete(
        self,
        prompt,
        image_path=None,
        source="Model",
        max_attempts=3,
        base_delay=2.0
    ):
        """Call the model and return non-empty text.

        Retries transient API failures and empty completions, so callers
        never have to reason about missing choices or None content.
        """

        content = self._build_content(prompt, image_path)

        last_error = None

        for attempt in range(1, max_attempts + 1):

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ]
                )

            except RETRYABLE_ERRORS as exc:
                last_error = (
                    f"{source} API call failed "
                    f"({type(exc).__name__}: {exc})"
                )

            else:
                if not response.choices:
                    last_error = (
                        f"{source} returned no choices"
                    )

                else:
                    text = response.choices[0].message.content

                    if text and text.strip():
                        return text

                    last_error = (
                        f"{source} returned an empty response"
                    )

            if attempt < max_attempts:
                print(
                    f"  ↳ {last_error} — "
                    f"retrying ({attempt}/{max_attempts - 1})"
                )
                time.sleep(base_delay * (2 ** (attempt - 1)))

        raise RuntimeError(
            f"{last_error} after {max_attempts} attempts"
        )

    def generate(self, image_path=None, prompt="", source="Model"):

        return self.complete(
            prompt,
            image_path=image_path,
            source=source
        )