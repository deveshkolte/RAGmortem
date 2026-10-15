"""Tests for RAGmortem minimal CLI commands and error handling."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from ragmortem.cli import main


def test_cli_version() -> None:
    """ragmortem --version prints package version and exits 0."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            exit_code = main(["--version"])
        except SystemExit as exc:
            exit_code = exc.code
    assert exit_code == 0
    assert "ragmortem" in buf.getvalue()


def test_cli_schema() -> None:
    """ragmortem schema prints schema specification and full example."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main(["schema"])
    assert exit_code == 0
    out = buf.getvalue()
    assert "RAGmortem Input Case Schema" in out
    assert "retrieved" in out
    assert "score_type" in out
    assert "Full JSON Example:" in out


def test_cli_demo_text() -> None:
    """ragmortem demo prints SYNTHETIC DEMO DATA, 5 cases, and summary."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main(["demo"])
    assert exit_code == 0
    out = buf.getvalue()
    assert "SYNTHETIC DEMO DATA" in out
    assert "Case 1:" in out
    assert "Case 5:" in out
    assert "Diagnosis:" in out
    assert "Mode:" in out
    assert "Confidence:" in out
    assert "What this cannot tell you:" in out
    assert "Suggested next check:" in out
    assert "SUMMARY" in out
    assert "Total Cases: 5" in out
    assert "UNKNOWN Rate:" in out


def test_cli_demo_json() -> None:
    """ragmortem demo --format json outputs valid JSON with notice and cases."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main(["demo", "--format", "json"])
    assert exit_code == 0
    data = json.loads(buf.getvalue())
    assert data["notice"] == "SYNTHETIC DEMO DATA"
    assert len(data["cases"]) == 5
    assert data["summary"]["total_cases"] == 5
    assert "unknown_rate_pct" in data["summary"]


def test_cli_demo_markdown(tmp_path: Path) -> None:
    """ragmortem demo --format md --out FILE outputs markdown to file."""
    out_file = tmp_path / "demo_report.md"
    exit_code = main(["demo", "--format", "md", "--out", str(out_file)])
    assert exit_code == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "# RAGmortem Diagnostic Report" in content
    assert "SYNTHETIC DEMO DATA" in content
    assert "UNKNOWN Rate" in content


def test_cli_diagnose_json_file(tmp_path: Path) -> None:
    """ragmortem diagnose FILE.json works with text and json formats."""
    case = {
        "id": "c1",
        "question": "What is the mandatory response time for a P1 incident?",
        "answer": "15 minutes",
        "retrieved": [
            {"chunk_id": "sla_1", "text": "P1 response time is 15 minutes.", "score": 0.88}
        ],
    }
    input_file = tmp_path / "test_case.json"
    input_file.write_text(json.dumps(case), encoding="utf-8")

    # Text format
    buf_text = io.StringIO()
    with redirect_stdout(buf_text):
        exit_code = main(["diagnose", str(input_file)])
    assert exit_code == 0
    assert "Case 1: [c1]" in buf_text.getvalue()
    assert "Total Cases: 1" in buf_text.getvalue()

    # JSON format
    buf_json = io.StringIO()
    with redirect_stdout(buf_json):
        exit_code = main(["diagnose", str(input_file), "--format", "json"])
    assert exit_code == 0
    data = json.loads(buf_json.getvalue())
    assert len(data["cases"]) == 1
    assert data["cases"][0]["id"] == "c1"


def test_cli_diagnose_jsonl_file(tmp_path: Path) -> None:
    """ragmortem diagnose FILE.jsonl processes multiple cases."""
    cases = [
        {
            "id": "case_1",
            "question": "Question 1?",
            "answer": "Answer 1",
            "retrieved": [{"text": "Content 1", "score": 0.80}],
        },
        {
            "id": "case_2",
            "question": "Question 2?",
            "answer": "Answer 2",
            "retrieved": [{"text": "Content 2", "score": 0.20}],
        },
    ]
    jsonl_file = tmp_path / "cases.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main(["diagnose", str(jsonl_file)])
    assert exit_code == 0
    out = buf.getvalue()
    assert "Case 1: [case_1]" in out
    assert "Case 2: [case_2]" in out
    assert "Total Cases: 2" in out


def test_cli_score_type_guard_distance(tmp_path: Path) -> None:
    """ragmortem diagnose --score-type distance flags unknown distance scale."""
    case = {
        "id": "dist_case",
        "question": "What is the policy?",
        "answer": "Standard policy.",
        "retrieved": [{"text": "Policy chunk", "score": 0.15}],
    }
    input_file = tmp_path / "dist_case.json"
    input_file.write_text(json.dumps(case), encoding="utf-8")

    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main(["diagnose", str(input_file), "--score-type", "distance"])
    assert exit_code == 0
    out = buf.getvalue()
    assert "scores are distances of unknown scale" in out
    assert "UNKNOWN Rate: 100.0%" in out


def test_cli_score_type_guard_cosine_distance(tmp_path: Path) -> None:
    """ragmortem diagnose --score-type cosine_distance converts 1 - d."""
    case = {
        "id": "cos_case",
        "question": "What is the policy?",
        "answer": "I cannot answer this question based on the provided context.",
        # Distance 0.10 -> similarity 0.90 -> false refusal
        "retrieved": [{"text": "Policy chunk", "score": 0.10}],
    }
    input_file = tmp_path / "cos_case.json"
    input_file.write_text(json.dumps(case), encoding="utf-8")

    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main(["diagnose", str(input_file), "--score-type", "cosine_distance"])
    assert exit_code == 0
    out = buf.getvalue()
    assert "generation_suspected" in out.lower()


def test_cli_bad_input_nonexistent_file() -> None:
    """Nonexistent input file produces readable error and exit code 2."""
    buf_err = io.StringIO()
    with redirect_stderr(buf_err):
        exit_code = main(["diagnose", "definitely_nonexistent_file_xyz.json"])
    assert exit_code == 2
    assert "does not exist" in buf_err.getvalue()


def test_cli_bad_input_malformed_json(tmp_path: Path) -> None:
    """Malformed JSON file produces readable error and exit code 2."""
    bad_file = tmp_path / "malformed.json"
    bad_file.write_text("{broken json syntax", encoding="utf-8")

    buf_err = io.StringIO()
    with redirect_stderr(buf_err):
        exit_code = main(["diagnose", str(bad_file)])
    assert exit_code == 2
    assert "Error reading input" in buf_err.getvalue()


def test_cli_bad_input_missing_required_fields(tmp_path: Path) -> None:
    """Trace missing required question/answer produces readable error and exit code 2."""
    missing_q = tmp_path / "missing_q.json"
    missing_q.write_text(json.dumps({"id": "no_q", "answer": "Answer"}), encoding="utf-8")

    buf_err = io.StringIO()
    with redirect_stderr(buf_err):
        exit_code = main(["diagnose", str(missing_q)])
    assert exit_code == 2
    assert "missing required field" in buf_err.getvalue()



def test_cli_bad_input_directory(tmp_path: Path) -> None:
    """Passing a directory path to diagnose produces readable error and exit code 2."""
    dir_path = tmp_path / "a_directory"
    dir_path.mkdir()

    buf_err = io.StringIO()
    with redirect_stderr(buf_err):
        exit_code = main(["diagnose", str(dir_path)])
    assert exit_code == 2
    assert "is a directory" in buf_err.getvalue()


def test_cli_diagnose_non_numeric_score(tmp_path: Path) -> None:
    """Traces with non-numeric or empty string scores process safely without crashing."""
    case = {
        "id": "nan_score_case",
        "question": "What is the policy?",
        "answer": "Policy answer.",
        "retrieved": [{"chunk_id": "c1", "text": "Policy chunk content.", "score": "N/A"}],
    }
    input_file = tmp_path / "nan_case.json"
    input_file.write_text(json.dumps(case), encoding="utf-8")

    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main(["diagnose", str(input_file)])
    assert exit_code == 0
    assert "nan_score_case" in buf.getvalue()


def test_cli_diagnose_explicit_scores_malformed(tmp_path: Path) -> None:
    """Traces with malformed explicit score array process safely without crashing."""
    case = {
        "id": "malformed_explicit_case",
        "question": "What is the policy?",
        "answer": "Policy answer.",
        "retrieved": [{"chunk_id": "c1", "text": "Policy chunk content."}],
        "scores": ["invalid_score_str"],
    }
    input_file = tmp_path / "malformed_scores.json"
    input_file.write_text(json.dumps(case), encoding="utf-8")

    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main(["diagnose", str(input_file)])
    assert exit_code == 0
    assert "malformed_explicit_case" in buf.getvalue()


def test_cli_diagnose_with_profile(tmp_path: Path) -> None:
    """ragmortem diagnose --profile openai applies OpenAI thresholds."""
    case = {
        "id": "openai_profile_test",
        "question": "What is the policy?",
        "answer": "Policy answer.",
        "retrieved": [{"chunk_id": "c1", "text": "Policy chunk content.", "score": 0.70}],
    }
    input_file = tmp_path / "openai_case.json"
    input_file.write_text(json.dumps(case), encoding="utf-8")

    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main(["diagnose", str(input_file), "--profile", "openai"])
    assert exit_code == 0
    out = buf.getvalue()
    assert "openai_profile_test" in out
    # 0.70 is ambiguous under OpenAI
    assert "UNKNOWN" in out


def test_cli_demo_with_profile() -> None:
    """ragmortem demo --profile relative runs cleanly."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        exit_code = main(["demo", "--profile", "relative"])
    assert exit_code == 0
    assert "SYNTHETIC DEMO DATA" in buf.getvalue()
