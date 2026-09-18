import json
import re


_FENCE = re.compile(r"```[A-Za-z0-9_+-]*\s*\n?(.*?)```", re.DOTALL)


def parse_model_json(content, source):
    """Parse JSON out of a model response.

    Tolerates the three things models actually do: fenced blocks with or
    without a language tag, and prose wrapped around the JSON.
    """

    if content is None or not content.strip():
        raise ValueError(f"{source} returned an empty response")

    text = content.strip()

    candidates = [text]

    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise ValueError(
        f"{source} returned invalid JSON:\n{content}"
    )
