"""Unit tests for LangChain and LlamaIndex 1-line RAGmortem callbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ragmortem.callbacks import (
    LangChainRAGmortemHandler,
    LlamaIndexRAGmortemHandler,
    RAGmortemCallbackHandler,
)
from ragmortem.taxonomy import FailureType


@dataclass
class MockDocument:
    """Mock LangChain Document."""
    page_content: str
    metadata: dict[str, Any]
    id: str | None = None


@dataclass
class MockNode:
    """Mock LlamaIndex Node."""
    node_id: str
    text: str
    extra_info: dict[str, Any]


@dataclass
class MockNodeWithScore:
    """Mock LlamaIndex NodeWithScore."""
    node: MockNode
    score: float


@dataclass
class MockResponse:
    """Mock LlamaIndex Response."""
    response: str


def test_langchain_callback_handler_telemetry() -> None:
    handler = RAGmortemCallbackHandler()

    # 1. Chain start with user query
    handler.on_chain_start(
        serialized={},
        inputs={"question": "What is the engineering response time for a P1 incident?"},
    )
    assert handler.query == "What is the engineering response time for a P1 incident?"

    # 2. Retriever ends with documents & scores
    docs = [
        MockDocument(
            page_content="The mandatory response SLA for P1 incidents is 15 minutes.",
            metadata={"id": "doc_p1", "score": 0.88},
        ),
        MockDocument(
            page_content="P2 response SLA is 1 hour.",
            metadata={"id": "doc_p2", "score": 0.42},
        ),
    ]
    handler.on_retriever_end(docs)
    assert len(handler.retrieved_chunks) == 2
    assert handler.retrieved_chunks[0]["id"] == "doc_p1"
    assert handler.scores == [0.88, 0.42]

    # 3. Chain ends with generated answer
    handler.on_chain_end(outputs={"answer": "The SLA for P1 is 12 hours."})
    assert handler.generated_answer == "The SLA for P1 is 12 hours."

    # 4. Construct trace and run diagnosis
    trace_obj = handler.get_trace()
    assert trace_obj.query == "What is the engineering response time for a P1 incident?"
    assert trace_obj.scores == [0.88, 0.42]

    diag = handler.diagnose(show_terminal=False)
    # High score (0.88) with incorrect answer -> GENERATION_SUSPECTED
    assert diag.diagnosis == FailureType.GENERATION_SUSPECTED
    assert diag.evidence_score > 0.80


def test_langchain_auto_diagnose() -> None:
    handler = RAGmortemCallbackHandler(auto_diagnose=True)
    handler.on_chain_start(serialized={}, inputs="What is the SLA?")
    handler.on_retriever_end([
        MockDocument(page_content="Irrelevant text.", metadata={"score": 0.15})
    ])
    handler.on_chain_end(outputs="Unknown.", show_terminal=False)

    assert handler.last_diagnosis is not None
    assert handler.last_diagnosis.diagnosis in (FailureType.RETRIEVAL_SUSPECTED, FailureType.NO_FAILURE, FailureType.UNKNOWN)


def test_llamaindex_callback_handler_telemetry() -> None:
    handler = LlamaIndexRAGmortemHandler()

    # 1. Start event with query_str
    handler.on_event_start(
        event_type="query",
        payload={"query_str": "What is the statutory deadline for FDA response?"},
    )
    assert handler.query == "What is the statutory deadline for FDA response?"

    # 2. End retrieve event with nodes
    nodes = [
        MockNodeWithScore(
            node=MockNode(node_id="node_fda_1", text="FDA review takes 30 days.", extra_info={}),
            score=0.85,
        ),
        MockNodeWithScore(
            node=MockNode(node_id="node_fda_2", text="FDA safety guidelines.", extra_info={}),
            score=0.35,
        ),
    ]
    handler.on_event_end(event_type="retrieve", payload={"nodes": nodes})
    assert len(handler.retrieved_chunks) == 2
    assert handler.retrieved_chunks[0]["id"] == "node_fda_1"
    assert handler.scores == [0.85, 0.35]

    # 3. End synthesize event with response
    handler.on_event_end(
        event_type="synthesize",
        payload={"response": MockResponse(response="The deadline is 60 days.")},
    )
    assert handler.generated_answer == "The deadline is 60 days."

    # 4. Diagnose
    diag = handler.diagnose(show_terminal=False)
    assert diag.diagnosis == FailureType.GENERATION_SUSPECTED
