"""Tests for interactive observatory HTTP server and diagnostic API endpoints."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from typing import Generator

import pytest

from ragmortem.server import DashboardHTTPHandler


@pytest.fixture(scope="module")
def local_server() -> Generator[str, None, None]:
    """Spin up a background test server bound to an ephemeral port."""
    server = HTTPServer(("127.0.0.1", 0), DashboardHTTPHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    try:
        yield base_url
    finally:
        server.shutdown()
        server.server_close()


def test_server_health_endpoint(local_server: str) -> None:
    """GET /api/health returns status ok with version."""
    with urllib.request.urlopen(f"{local_server}/api/health") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "ok"
        assert "version" in data


def test_server_demo_endpoint(local_server: str) -> None:
    """GET /api/demo returns synthetic benchmark cases."""
    with urllib.request.urlopen(f"{local_server}/api/demo") as resp:
        assert resp.status == 200
        cases = json.loads(resp.read().decode("utf-8"))
        assert isinstance(cases, list)
        assert len(cases) >= 5


def test_server_html_index(local_server: str) -> None:
    """GET / serves the interactive enterprise dashboard HTML."""
    with urllib.request.urlopen(f"{local_server}/") as resp:
        assert resp.status == 200
        content = resp.read().decode("utf-8")
        assert "<!DOCTYPE html>" in content
        assert "RAGmortem" in content


def test_server_diagnose_post(local_server: str) -> None:
    """POST /api/diagnose returns root-cause diagnostic verdict."""
    payload = {
        "question": "What is the return policy?",
        "answer": "Returns are accepted within 30 days.",
        "retrieved": [{"chunk_id": "c1", "text": "All items can be returned within 30 days.", "score": 0.92}],
    }
    req = urllib.request.Request(
        f"{local_server}/api/diagnose",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "diagnosis" in data
        assert "diagnosis_english" in data
        assert "mode" in data


def test_server_diagnose_post_empty_body(local_server: str) -> None:
    """POST /api/diagnose with empty body cleanly returns HTTP 400."""
    req = urllib.request.Request(
        f"{local_server}/api/diagnose",
        data=b"",
        headers={"Content-Length": "0"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400


def test_server_diagnose_post_malformed_json(local_server: str) -> None:
    """POST /api/diagnose with malformed JSON cleanly returns HTTP 400."""
    req = urllib.request.Request(
        f"{local_server}/api/diagnose",
        data=b"{not valid json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400


def test_server_not_found(local_server: str) -> None:
    """GET /nonexistent returns HTTP 404."""
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{local_server}/nonexistent")
    assert exc_info.value.code == 404
