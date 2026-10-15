"""RAGmortem: An open-source RAG failure debugger."""

__version__ = "0.2.0.dev0"

from ragmortem.types import Chunk, RagResult
from ragmortem.adapter import RagAdapter
from ragmortem.taxonomy import FailureType
from ragmortem.trace import Trace, trace
from ragmortem.diagnose import DiagnosticResult
from ragmortem.api import diagnose
from ragmortem.retrieval import (
    CorpusRetriever,
    load_corpus_from_directory,
    parse_chunks_from_markdown,
)

__all__ = [
    "diagnose",
    "trace",
    "Trace",
    "DiagnosticResult",
    "Chunk",
    "RagResult",
    "RagAdapter",
    "FailureType",
    "CorpusRetriever",
    "load_corpus_from_directory",
    "parse_chunks_from_markdown",
    "__version__",
]
