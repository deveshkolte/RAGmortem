"""Unit tests for local model cache."""

import json
from pathlib import Path

from ragmortem.cache import ModelCache


def test_cache_key_determinism() -> None:
    k1 = ModelCache.compute_key(
        model="qwen/qwen3.8-27b",
        prompt="Tell me about SLA",
        system_prompt="You are an assistant",
        temperature=0.0,
    )
    k2 = ModelCache.compute_key(
        model="qwen/qwen3.8-27b",
        prompt="Tell me about SLA",
        system_prompt="You are an assistant",
        temperature=0.0,
    )
    # Different prompt produces different key
    k3 = ModelCache.compute_key(
        model="qwen/qwen3.8-27b",
        prompt="Different prompt",
        system_prompt="You are an assistant",
        temperature=0.0,
    )

    assert k1 == k2
    assert k1 != k3
    assert len(k1) == 64  # SHA-256 length


def test_cache_set_and_get(tmp_path: Path) -> None:
    cache = ModelCache(cache_dir=tmp_path)
    key = "test_key_123"

    assert cache.get(key) is None

    cache.set(
        key=key,
        response="15 minutes",
        model="test-model",
        prompt="What is SLA?",
    )

    cached_val = cache.get(key)
    assert cached_val == "15 minutes"

    # Verify underlying file contains no secret keys
    file_path = tmp_path / f"{key}.json"
    assert file_path.exists()
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["response"] == "15 minutes"
    assert "api_key" not in data
    assert "Authorization" not in str(data)


def test_cache_clear(tmp_path: Path) -> None:
    cache = ModelCache(cache_dir=tmp_path)
    cache.set("k1", "ans1", "m", "p")
    cache.set("k2", "ans2", "m", "p")

    assert (tmp_path / "k1.json").exists()
    assert (tmp_path / "k2.json").exists()

    cleared = cache.clear()
    assert cleared == 2
    assert cache.get("k1") is None
    assert cache.get("k2") is None
