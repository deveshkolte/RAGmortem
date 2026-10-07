"""Deterministic local disk cache for model completions."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


class ModelCache:
    """Simple JSON-file based cache for LLM completions.

    Stores responses indexed by SHA-256 hash of the request parameters
    (model, prompt, system prompt, temperature). Never persists API keys
    or sensitive credentials.
    """

    def __init__(self, cache_dir: str | Path = ".ragmortem_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_key(
        model: str,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
    ) -> str:
        """Deterministically calculate SHA-256 key from request content."""
        payload = {
            "model": model.strip(),
            "prompt": prompt.strip(),
            "system_prompt": system_prompt.strip(),
            "temperature": round(float(temperature), 4),
        }
        raw_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()

    def get(self, key: str) -> str | None:
        """Retrieve cached text answer if it exists, else None."""
        entry_path = self.cache_dir / f"{key}.json"
        if not entry_path.exists():
            return None
        try:
            with open(entry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("response")
        except (json.JSONDecodeError, OSError):
            return None

    def set(
        self,
        key: str,
        response: str,
        model: str,
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store completion in cache without any secret credentials."""
        entry_path = self.cache_dir / f"{key}.json"
        data = {
            "key": key,
            "model": model,
            "prompt": prompt,
            "response": response,
            "created_at": time.time(),
            "metadata": metadata or {},
        }
        tmp_path = entry_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, entry_path)

    def clear(self) -> int:
        """Remove all cache files. Returns count of deleted entries."""
        count = 0
        for p in self.cache_dir.glob("*.json"):
            try:
                p.unlink()
                count += 1
            except OSError:
                pass
        return count
