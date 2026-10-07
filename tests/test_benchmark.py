"""Unit tests for competitor benchmark harness and mapping validity."""

import json
from pathlib import Path

from benchmarks.run_benchmark import run_benchmark
from ragmortem.faults.inject import load_injected_faults


def test_competitor_mappings_schema() -> None:
    mappings_file = Path("benchmarks/mappings.json")
    assert mappings_file.exists(), "mappings.json must exist"

    with open(mappings_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "tools" in data
    tools = data["tools"]
    expected_tools = ["Ragas", "PyVectorHound", "DeepDiag_RAGDiag", "RAG_Oracle"]
    for t in expected_tools:
        assert t in tools, f"Missing tool {t} in mappings.json"
        tool_data = tools[t]
        assert "installation_status" in tool_data
        assert "taxonomy_mapping" in tool_data

        mapping = tool_data["taxonomy_mapping"]
        for cat in ["retrieval_miss", "ranking_miss", "generation_ignored_context", "should_abstain"]:
            assert cat in mapping, f"Missing category {cat} in {t}"
            assert mapping[cat]["status"] in ["SUPPORTED", "PARTIAL", "NOT_SUPPORTED", "UNKNOWN"]
            assert len(mapping[cat]["evidence"]) > 0


def test_benchmark_execution_and_accuracy() -> None:
    cases = load_injected_faults("evals/injected_faults.jsonl")
    results = run_benchmark(cases)

    assert len(results) == 4
    assert "Ragas (Metric-Based)" in results
    assert "PyVectorHound (Retrieval-Only)" in results

    # Verify Ragas has 0% recall on ranking_miss (Top-K Blindness)
    ragas_res = results["Ragas (Metric-Based)"]
    assert ragas_res.category_metrics["ranking_miss"].recall == 0.0
    assert ragas_res.category_metrics["ranking_miss"].tp == 0

    # Verify PyVectorHound covers < 50% of all failures
    hound_res = results["PyVectorHound (Retrieval-Only)"]
    assert hound_res.coverage_pct < 50.0

    # Verify overall accuracy for all existing tools is <= 50%
    for name, res in results.items():
        assert res.overall_accuracy_pct <= 50.0, f"Tool {name} had unexpected >50% overall accuracy"
