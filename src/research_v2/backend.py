"""Bounded HTTP backend for hosted or locally served vision models."""

import base64
import hashlib
import json
import mimetypes
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

from research_v2 import VERSION
from utils.free_policy import FREE_OPTIONS, validate_route, before_hosted_request, FreeQuotaPause


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class ChatBackend:
    """One request per call; no hidden SDK retries or unbounded recursion.

    Supports chat-completions compatible servers including a local server
    with multimodal support. Cached artifacts include content hashes; image
    bytes changing at the same pathname cannot reuse an old judgment.
    """

    def __init__(self, model, endpoint="https://openrouter.ai/api/v1",
                 cache_dir=None, max_tokens=3000, timeout=60):
        load_dotenv()
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        validate_route(model, self.endpoint)
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Endpoint must be an HTTP(S) API base URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Endpoint must not contain credentials, queries, or fragments")
        self.local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        # This research method uses explicit evidence artifacts. Disable the
        # provider's optional hidden-reasoning allocation on OpenRouter so
        # small JSON tasks do not exhaust their output budget before answering.
        # Do not send this provider-specific option to local servers.
        self.request_options = FREE_OPTIONS if parsed.hostname == "openrouter.ai" else {}
        self.key = (os.getenv("OPENROUTER_API_KEY") if parsed.hostname == "openrouter.ai"
                    else os.getenv("V2_API_KEY"))
        if not self.local and not self.key:
            raise ValueError("Set OPENROUTER_API_KEY or V2_API_KEY for this hosted endpoint")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if max_tokens < 1 or timeout <= 0:
            raise ValueError("max_tokens and timeout must be positive")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.calls = []

    def complete(self, prompt, image_path=None, source="model"):
        image_bytes = Path(image_path).read_bytes() if image_path else None
        image_hash = hashlib.sha256(image_bytes).hexdigest() if image_bytes else None
        identity = {"version": VERSION, "endpoint": self.endpoint, "model": self.model,
                    "prompt": prompt, "image_sha256": image_hash,
                    "temperature": 0, "max_tokens": self.max_tokens,
                    "request_options": self.request_options}
        key = digest(identity)
        cache_path = self.cache_dir / (key + ".json") if self.cache_dir else None
        record = {"source": source, "request_hash": key, "image_sha256": image_hash,
                  "cache_hit": False, "network_attempted": False,
                  "max_tokens": self.max_tokens, "temperature": 0,
                  "request_options": self.request_options,
                  "usage": None, "status": "started"}
        self.calls.append(record)
        started = time.monotonic()
        try:
            if cache_path and cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached["request_hash"] != key or not cached["text"].strip():
                    raise ValueError("Invalid cached completion")
                record.update(cache_hit=True, status="completed")
                return cached["text"]
            content = prompt
            if image_bytes:
                mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
                content = [{"type": "text", "text": prompt},
                           {"type": "image_url", "image_url": {"url":
                               f"data:{mime};base64," + base64.b64encode(image_bytes).decode()}}]
            headers = {"Content-Type": "application/json"}
            if self.key:
                headers["Authorization"] = "Bearer " + self.key
            if not self.local:
                before_hosted_request(self.model, self.key)
            record["network_attempted"] = True
            response = requests.post(
                self.endpoint + "/chat/completions", headers=headers,
                json={"model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
                      "messages": [{"role": "user", "content": content}], **self.request_options},
                timeout=self.timeout,
            )
            if response.status_code >= 400:
                if response.status_code in {401, 402, 403, 429}:
                    raise FreeQuotaPause(f"Free endpoint returned HTTP {response.status_code}; stopped without paid fallback")
                # Avoid copying provider response bodies or credentials into reports.
                raise RuntimeError(f"Model endpoint returned HTTP {response.status_code}")
            payload = response.json()
            record["response_model"] = payload.get("model")
            record["usage"] = payload.get("usage")
            record["finish_reason"] = payload["choices"][0].get("finish_reason")
            if record["finish_reason"] == "length":
                raise ValueError("Completion exhausted max_tokens; increase the output-token budget")
            text = payload["choices"][0]["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Model returned empty/non-text content")
            record.update(status="completed", usage=payload.get("usage"))
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({"request_hash": key, "text": text},
                                                ensure_ascii=False), encoding="utf-8")
            return text
        except Exception as exc:
            record.update(status="failed", error_type=type(exc).__name__)
            raise
        finally:
            record["latency_seconds"] = round(time.monotonic() - started, 6)
