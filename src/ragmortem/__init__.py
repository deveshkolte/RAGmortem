"""RAGmortem: An open-source RAG failure debugger."""

from ragmortem.types import Chunk, RagResult
from ragmortem.adapter import RagAdapter
from ragmortem.taxonomy import FailureType
from ragmortem.trace import Trace, trace
from ragmortem.diagnose import DiagnosticResult
from ragmortem.api import diagnose

__version__ = "0.1.0"

__all__ = [
    "diagnose",
    "trace",
    "Trace",
    "DiagnosticResult",
    "Chunk",
    "RagResult",
    "RagAdapter",
    "FailureType",
    "__version__",
]
