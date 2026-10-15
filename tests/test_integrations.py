"""Unit tests for LangChain and LlamaIndex framework adapters."""

from __future__ import annotations

import pytest
from ragmortem.integrations import from_langchain, from_llamaindex
from ragmortem.api import diagnose


class MockLangChainDoc:
    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class MockAIMessage:
    def __init__(self, content: str):
        self.content = content


class MockLlamaIndexNode:
    def __init__(self, text: str, node_id: str):
        self.text = text
        self.node_id = node_id


class MockNodeWithScore:
    def __init__(self, text: str, node_id: str, score: float):
        self.node = MockLlamaIndexNode(text, node_id)
        self.score = score


class MockLlamaIndexResponse:
    def __init__(self, response: str, source_nodes: list, query_str: str | None = None):
        self.response = response
        self.source_nodes = source_nodes
        self.query_str = query_str


def test_from_langchain_dict_input() -> None:
    """Verifies from_langchain handles standard RetrievalQA output dicts."""
    run_dict = {
        "query": "What is the P1 SLA?",
        "result": "P1 SLA is 15 minutes.",
        "source_documents": [
            {
                "page_content": "P1 SLA is 15 minutes.",
                "metadata": {"source": "handbook.md", "id": "chunk_sla_1", "score": 0.92},
            },
            {
                "page_content": "P2 SLA is 4 hours.",
                "metadata": {"source": "handbook.md", "id": "chunk_sla_2", "score": 0.45},
            },
        ],
    }

    t = from_langchain(run_dict)
    assert t.query == "What is the P1 SLA?"
    assert t.answer == "P1 SLA is 15 minutes."
    assert len(t.retrieved_chunks) == 2
    assert t.retrieved_chunks[0]["id"] == "chunk_sla_1"
    assert t.retrieved_chunks[0]["score"] == 0.92
    assert t.scores == [0.92, 0.45]

    # Diagnose the extracted trace
    res = diagnose(t, reference_answer="15 minutes")
    assert res.diagnosis.value.lower() == "no_failure"


def test_from_langchain_objects() -> None:
    """Verifies from_langchain handles Document objects and AIMessage responses."""
    run_dict = {
        "question": "What is the capital of France?",
        "output": MockAIMessage("Paris is the capital of France."),
        "documents": [
            MockLangChainDoc("Paris is the capital and most populous city of France.", {"id": "geo_1", "score": 0.89}),
        ],
    }

    t = from_langchain(run_dict)
    assert t.query == "What is the capital of France?"
    assert t.answer == "Paris is the capital of France."
    assert len(t.retrieved_chunks) == 1
    assert t.retrieved_chunks[0]["id"] == "geo_1"
    assert t.retrieved_chunks[0]["score"] == 0.89
    assert t.scores == [0.89]


def test_from_llamaindex_response_object() -> None:
    """Verifies from_llamaindex handles Response objects with NodeWithScore."""
    source_nodes = [
        MockNodeWithScore("Kubernetes pods restart automatically on failure.", "k8s_node_1", 0.94),
        MockNodeWithScore("Deployments manage replicasets in Kubernetes.", "k8s_node_2", 0.61),
    ]
    resp = MockLlamaIndexResponse(
        response="Pods restart automatically on failure.",
        source_nodes=source_nodes,
        query_str="How do Kubernetes pods handle failure?",
    )

    t = from_llamaindex(resp)
    assert t.query == "How do Kubernetes pods handle failure?"
    assert t.answer == "Pods restart automatically on failure."
    assert len(t.retrieved_chunks) == 2
    assert t.retrieved_chunks[0]["id"] == "k8s_node_1"
    assert t.retrieved_chunks[0]["score"] == 0.94
    assert t.scores == [0.94, 0.61]

    # Diagnose the extracted trace
    res = diagnose(t, reference_answer="Pods restart automatically")
    assert res.diagnosis.value.lower() == "no_failure"


def test_from_llamaindex_dict_input() -> None:
    """Verifies from_llamaindex handles dictionary representations."""
    dict_resp = {
        "query": "What is OAuth2?",
        "response": "OAuth2 is an authorization framework.",
        "source_nodes": [
            {
                "node": {"text": "OAuth 2.0 is an authorization framework that enables applications.", "id": "auth_1"},
                "score": 0.88,
            }
        ],
    }

    t = from_llamaindex(dict_resp)
    assert t.query == "What is OAuth2?"
    assert t.answer == "OAuth2 is an authorization framework."
    assert len(t.retrieved_chunks) == 1
    assert t.retrieved_chunks[0]["id"] == "auth_1"
    assert t.retrieved_chunks[0]["score"] == 0.88
