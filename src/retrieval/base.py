"""Common Retrieval Interface for ProjectGenome.

Defines the normalized RetrievedItem output model, BaseRetriever abstract base class,
and standard retrieval protocols as specified in Phase 7 of the ProjectGenome roadmap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field

from .chunker import CodeChunk


class RetrievedItem(BaseModel):
    """Normalized retrieval result item returned by any ProjectGenome retriever.

    Conforms to Phase 7.2 common output specifications:
    - entity_id
    - file_path
    - source_code
    - score
    - retrieval_method
    - provenance
    """

    entity_id: str = Field(..., description="Deterministic entity identifier (e.g. func:main.py:main)")
    file_path: str = Field(..., description="Relative file path")
    source_code: str = Field(default="", description="Source code snippet for the retrieved entity")
    score: float = Field(..., description="Relevance or similarity score")
    retrieval_method: str = Field(
        ...,
        description="Identifier of the retrieval strategy (e.g. 'structural', 'semantic', 'bm25', 'hybrid', 'graph_aware')",
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Traceability metadata (starting entity, traversal path, relationship types, etc.)",
    )

    # Optional metadata attributes preserved from CodeChunk / KG Node
    entity_type: str | None = Field(default=None, description="Entity type: file, class, function, method")
    name: str | None = Field(default=None, description="Simple entity name")
    qualified_name: str | None = Field(default=None, description="Qualified entity name")
    start_line: int | None = Field(default=None, description="Starting line number (1-indexed)")
    end_line: int | None = Field(default=None, description="Ending line number (1-indexed)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional entity properties")

    def to_dict(self) -> dict[str, Any]:
        """Serialize retrieved item to dictionary."""
        return self.model_dump()

    @classmethod
    def from_chunk(
        cls,
        chunk: CodeChunk,
        score: float,
        retrieval_method: str,
        provenance: dict[str, Any] | None = None,
    ) -> RetrievedItem:
        """Create a RetrievedItem from a CodeChunk."""
        return cls(
            entity_id=chunk.entity_id,
            file_path=chunk.file_path,
            source_code=chunk.source_code,
            score=score,
            retrieval_method=retrieval_method,
            provenance=provenance or {},
            entity_type=chunk.entity_type,
            name=chunk.name,
            qualified_name=chunk.qualified_name,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            metadata=chunk.metadata,
        )


class BaseRetriever(ABC):
    """Abstract base class for all ProjectGenome retrieval systems.

    Implements the common retrieval interface:
        retrieve(question, repository, k) -> list[RetrievedItem]
    """

    @abstractmethod
    def retrieve(
        self,
        question: str,
        repository: Any = None,
        k: int = 10,
    ) -> list[RetrievedItem]:
        """Retrieve top-k relevant repository code units for a question.

        Args:
            question: Natural language question or query string.
            repository: Optional repository representation or KnowledgeGraph.
            k: Maximum number of retrieved items to return.

        Returns:
            List of RetrievedItem objects sorted descending by score.
        """
        pass


class DummyRetriever(BaseRetriever):
    """Reference dummy retriever to validate pipeline compatibility (Phase 7 exit condition)."""

    def __init__(self, fixed_items: list[RetrievedItem] | None = None) -> None:
        self.fixed_items = fixed_items or []

    def retrieve(
        self,
        question: str,
        repository: Any = None,
        k: int = 10,
    ) -> list[RetrievedItem]:
        return self.fixed_items[:k]
