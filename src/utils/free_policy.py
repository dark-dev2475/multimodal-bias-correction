"""Project-wide free-only inference policy. No paid override or model fallback."""

import hashlib
import sqlite3
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import requests

DEFAULT_MODEL = "nvidia/nemotron-3.5-super-vl:free"
DEFAULT_JUDGE = "google/gemma-4-31b-it:free"
FREE_OPTIONS = {"reasoning": {"enabled": False},
                "provider": {"max_price": {"prompt": 0, "completion": 0},
                             "allow_fallbacks": False}}


class FreeQuotaPause(RuntimeError):
    """Infrastructure pause, not a model-quality failure."""


def validate_route(model, endpoint="https://openrouter.ai/api/v1"):
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Invalid free-only endpoint")
    if parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        return "local"
    if endpoint.rstrip("/") != "https://openrouter.ai/api/v1":
        raise ValueError("Free-only policy permits loopback servers or the official OpenRouter endpoint only")
    if not isinstance(model, str) or not model.endswith(":free") or "/" not in model:
        raise ValueError("Free-only policy: select a fixed model ID ending in :free; paid/automatic routes are disabled")
    return "openrouter"


def validate_entry(entry):
    pricing = entry.get("pricing", {})
    if not {"prompt", "completion"} <= pricing.keys():
        raise ValueError("Free-only policy: missing pricing")
    try:
        for value in pricing.values():
            price = Decimal(str(value))
            if not price.is_finite() or price != 0:
                raise ValueError("Free-only policy: nonzero or unknown price")
    except InvalidOperation as exc:
        raise ValueError("Free-only policy: invalid pricing") from exc
    if "image" not in entry.get("architecture", {}).get("input_modalities", []):
        raise ValueError("The selected free model must support image input")


@lru_cache(maxsize=16)
def verify_free_model(model):
    validate_route(model)
    response = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
    response.raise_for_status()
    entry = next((m for m in response.json()["data"] if m["id"] == model), None)
    if entry is None:
        raise ValueError("Selected free model is unavailable; no automatic replacement is allowed")
    validate_entry(entry)
    return entry


def reserve_free_request(key, database=None, daily_limit=45, interval=3.2):
    """Cross-process local accounting; failures count too. Provider caps remain authoritative.

    Key digest only, never the key, is stored. Other apps/account keys are not
    observable here: stop on provider limits, never rotate keys to evade them.
    """
    path = Path(database) if database else Path(__file__).resolve().parents[2] / "data/cache/free_quota.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(key.encode()).hexdigest()
    while True:
        now = time.time()
        day = datetime.fromtimestamp(now, timezone.utc).date().isoformat()
        with sqlite3.connect(path, timeout=10) as db:
            db.execute("CREATE TABLE IF NOT EXISTS quota (identity TEXT PRIMARY KEY, day TEXT, used INTEGER, last REAL)")
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT day, used, last FROM quota WHERE identity=?", (identity,)).fetchone()
            used = row[1] if row and row[0] == day else 0
            if used >= daily_limit:
                raise FreeQuotaPause("Local free quota reached; resume after the next UTC day. No paid fallback.")
            wait = max(0, interval - (now - row[2])) if row else 0
            if not wait:
                db.execute("INSERT OR REPLACE INTO quota VALUES (?, ?, ?, ?)", (identity, day, used + 1, now))
                return
        time.sleep(min(wait, interval))


def before_hosted_request(model, key):
    verify_free_model(model)
    reserve_free_request(key)
