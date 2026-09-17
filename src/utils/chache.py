import hashlib
import json
from pathlib import Path


class Cache:
    def __init__(self, cache_dir="data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, data):
        serialized = json.dumps(
            data,
            sort_keys=True,
            ensure_ascii=False
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    def get(self, namespace, key_data):
        key = self._make_key(key_data)

        namespace_dir = self.cache_dir / namespace
        cache_file = namespace_dir / f"{key}.json"

        if not cache_file.exists():
            return None

        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def set(self, namespace, key_data, value):
        key = self._make_key(key_data)

        namespace_dir = self.cache_dir / namespace
        namespace_dir.mkdir(parents=True, exist_ok=True)

        cache_file = namespace_dir / f"{key}.json"

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(
                value,
                f,
                indent=2,
                ensure_ascii=False
            )

        return cache_file