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

from ragmortem.report_html import generate_html_report
from ragmortem.server import run_server

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
    "generate_html_report",
    "run_server",
    "__version__",
]
