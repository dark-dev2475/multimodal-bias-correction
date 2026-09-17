import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}"
}

response = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers=headers
)

if response.status_code != 200:
    print("Error:", response.status_code)
    print(response.text)
    exit()

models = response.json()["data"]

print("\nVISION-CAPABLE MODELS")
print("=" * 70)

count = 0

for model in models:

    architecture = model.get("architecture", {})
    modalities = architecture.get("input_modalities", [])

    if "image" in modalities:

        model_id = model.get("id")
        pricing = model.get("pricing", {})

        print(f"\nModel: {model_id}")
        print(f"Input modalities: {modalities}")
        print(f"Pricing: {pricing}")

        count += 1

        if count >= 20:
            break