import os
import base64
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


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

    def generate(self, image_path=None, prompt=""):

        # -----------------------------
        # IMAGE + TEXT
        # -----------------------------

        if image_path is not None:

            image_data = self._encode_image(
                image_path
            )

            content = [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data
                    }
                }
            ]

        # -----------------------------
        # TEXT ONLY
        # -----------------------------

        else:

            content = prompt

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ]
        )

        return response.choices[0].message.content