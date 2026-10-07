"""RAGmortem: An open-source RAG failure debugger."""

from ragmortem.types import Chunk, RagResult
from ragmortem.adapter import RagAdapter
from ragmortem.taxonomy import FailureType

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "RagResult",
    "RagAdapter",
    "FailureType",
    "__version__",
]
