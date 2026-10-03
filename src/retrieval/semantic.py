"""Semantic Retrieval Engine for ProjectGenome using CodeBERT.

Retrieves repository code units by mapping queries and code chunks into a shared
768-dimensional dense vector space using microsoft/codebert-base embeddings
and cosine similarity search via CodeVectorIndex.
Conforms to Phase 7 BaseRetriever common interface and produces normalized RetrievedItem outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

from .base import BaseRetriever, RetrievedItem
from .chunker import CodeChunk, CodeEntityChunker
from .embeddings import CodeBERTEmbedder
from .index import CodeVectorIndex


class SemanticRetriever(BaseRetriever):
    """Semantic code retriever powered by CodeBERT (microsoft/codebert-base).

    Implements the ProjectGenome common retrieval interface:
        retrieve(question, repository, k) -> list[RetrievedItem]
    """

    def __init__(
        self,
        chunks: list[CodeChunk] | None = None,
        embedder: CodeBERTEmbedder | None = None,
        index: CodeVectorIndex | None = None,
        repo_path: str | Path | None = None,
        chunker: CodeEntityChunker | None = None,
        batch_size: int = 16,
    ) -> None:
        """Initialize SemanticRetriever.

        Args:
            chunks: Optional list of CodeChunk objects to index immediately.
            embedder: Optional CodeBERTEmbedder instance (defaults to microsoft/codebert-base).
            index: Optional pre-populated CodeVectorIndex instance.
            repo_path: Optional path to repository files for reading source code.
            chunker: Optional CodeEntityChunker instance.
            batch_size: Batch size for CodeBERT embedding generation.
        """
        self.repo_path = Path(repo_path) if repo_path else None
        self.embedder = embedder or CodeBERTEmbedder()
        self.vector_index = index or CodeVectorIndex()
        self.chunker = chunker or CodeEntityChunker(repo_path=self.repo_path)
        self.batch_size = max(1, batch_size)

        if chunks:
            self.index_chunks(chunks)

    @property
    def chunks(self) -> list[CodeChunk]:
        """Access indexed CodeChunks."""
        return self.vector_index.chunks

    def index_chunks(self, chunks: list[CodeChunk]) -> None:
        """Embed and index a list of CodeChunks. Replaces existing index contents."""
        self.vector_index = CodeVectorIndex()
        if not chunks:
            return

        embeddings = self.embedder.embed_chunks(chunks, batch_size=self.batch_size)
        self.vector_index.add_chunks(chunks, embeddings)

    def index(self, chunks: list[CodeChunk]) -> None:
        """Alias for index_chunks to maintain consistency across retrievers."""
        self.index_chunks(chunks)

    def add_chunks(self, chunks: list[CodeChunk]) -> None:
        """Embed and append new chunks to the existing vector index."""
        if not chunks:
            return
        embeddings = self.embedder.embed_chunks(chunks, batch_size=self.batch_size)
        self.vector_index.add_chunks(chunks, embeddings)

    def retrieve(
        self,
        question: str,
        repository: Any = None,
        k: int = 10,
        entity_type_filter: str | list[str] | None = None,
    ) -> list[RetrievedItem]:
        """Retrieve top-k semantically relevant code units for a question.

        Args:
            question: Natural language question or query string.
            repository: Optional repository representation (list of CodeChunk, KnowledgeGraph,
                        or analysis JSON file path). If provided, indexes the repository.
            k: Maximum number of retrieved items to return (k > 0).
            entity_type_filter: Optional entity type or list of entity types to restrict to
                                (e.g. 'function', ['class', 'method']).

        Returns:
            List of RetrievedItem objects sorted descending by cosine similarity score.
            Ties are broken deterministically by entity_id ascending.
        """
        if k <= 0:
            return []

        # If a repository is explicitly provided, prepare chunks
        if repository is not None:
            self._handle_repository_input(repository)

        if len(self.vector_index) == 0:
            return []

        if not question or not question.strip():
            return []

        # Embed query text
        query_vec = self.embedder.embed_text(question.strip())

        # Retrieve candidates from CodeVectorIndex
        fetch_k = max(k, min(len(self.vector_index), k + 10))
        raw_candidates = self.vector_index.search(
            query_embedding=query_vec,
            top_k=fetch_k,
            entity_type_filter=entity_type_filter,
        )

        if not raw_candidates:
            return []

        # Deterministic sorting: primary key is -score (descending), secondary is entity_id (ascending)
        sorted_candidates = sorted(
            raw_candidates,
            key=lambda item: (-round(float(item[1]), 6), item[0].entity_id),
        )

        top_candidates = sorted_candidates[:k]
        results: list[RetrievedItem] = []

        for chunk, score in top_candidates:
            provenance = {
                "similarity_score": round(float(score), 6),
                "model_name": getattr(self.embedder, "model_name", "microsoft/codebert-base"),
                "similarity_metric": "cosine",
                "entity_type_filter": entity_type_filter,
                "k": k,
            }

            item = RetrievedItem.from_chunk(
                chunk=chunk,
                score=round(float(score), 6),
                retrieval_method="semantic_codebert",
                provenance=provenance,
            )
            results.append(item)

        return results

    def _handle_repository_input(self, repository: Any) -> None:
        """Helper to index repository chunks if passed to retrieve()."""
        if isinstance(repository, list):
            if repository and isinstance(repository[0], CodeChunk):
                self.index_chunks(repository)
            return

        # Check for KnowledgeGraph
        if hasattr(repository, "nodes") or hasattr(repository, "underlying_graph"):
            chunks = self.chunker.chunk_knowledge_graph(repository)
            self.index_chunks(chunks)
            return

        # Check for filepath to analysis JSON
        if isinstance(repository, (str, Path)):
            path = Path(repository)
            if path.is_file():
                chunks = self.chunker.chunk_analysis_file(path)
                self.index_chunks(chunks)
                return

    @classmethod
    def from_chunks(
        cls,
        chunks: list[CodeChunk],
        embedder: CodeBERTEmbedder | None = None,
        index: CodeVectorIndex | None = None,
        repo_path: str | Path | None = None,
        batch_size: int = 16,
    ) -> SemanticRetriever:
        """Create and index a SemanticRetriever from a list of CodeChunks."""
        return cls(
            chunks=chunks,
            embedder=embedder,
            index=index,
            repo_path=repo_path,
            batch_size=batch_size,
        )

    @classmethod
    def from_analysis_file(
        cls,
        analysis_file_path: str | Path,
        repo_path: str | Path | None = None,
        embedder: CodeBERTEmbedder | None = None,
        batch_size: int = 16,
    ) -> SemanticRetriever:
        """Create and index a SemanticRetriever from an analyzer JSON file."""
        chunker = CodeEntityChunker(repo_path=repo_path)
        chunks = chunker.chunk_analysis_file(analysis_file_path)
        return cls(
            chunks=chunks,
            embedder=embedder,
            repo_path=repo_path,
            chunker=chunker,
            batch_size=batch_size,
        )

    @classmethod
    def from_knowledge_graph(
        cls,
        graph: Any,
        repo_path: str | Path | None = None,
        embedder: CodeBERTEmbedder | None = None,
        batch_size: int = 16,
    ) -> SemanticRetriever:
        """Create and index a SemanticRetriever from an in-memory KnowledgeGraph."""
        chunker = CodeEntityChunker(repo_path=repo_path)
        chunks = chunker.chunk_knowledge_graph(graph)
        return cls(
            chunks=chunks,
            embedder=embedder,
            repo_path=repo_path,
            chunker=chunker,
            batch_size=batch_size,
        )

    def save(self, index_dir: str | Path) -> None:
        """Save the underlying vector index and retriever configuration."""
        dir_path = Path(index_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        self.vector_index.save(dir_path)

        config_path = dir_path / "semantic_config.json"
        config = {
            "model_name": getattr(self.embedder, "model_name", "microsoft/codebert-base"),
            "retrieval_method": "semantic_codebert",
            "num_chunks": len(self.vector_index),
            "batch_size": self.batch_size,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        embedder: CodeBERTEmbedder | None = None,
        repo_path: str | Path | None = None,
    ) -> SemanticRetriever:
        """Load an indexed SemanticRetriever from a directory."""
        dir_path = Path(index_dir)
        index = CodeVectorIndex.load(dir_path)

        batch_size = 16
        config_path = dir_path / "semantic_config.json"
        if config_path.is_file():
            try:
                with open(config_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                    batch_size = cfg.get("batch_size", 16)
            except Exception:
                pass

        return cls(
            embedder=embedder,
            index=index,
            repo_path=repo_path,
            batch_size=batch_size,
        )

    def __len__(self) -> int:
        """Return the number of indexed chunks."""
        return len(self.vector_index)
