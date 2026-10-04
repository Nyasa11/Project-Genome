"""Hybrid Retrieval Engine for ProjectGenome.

Combines lexical BM25 retrieval and dense semantic CodeBERT retrieval
into a unified, deterministic hybrid retriever conforming to the BaseRetriever
common interface and producing normalized RetrievedItem outputs.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .base import BaseRetriever, RetrievedItem
from .bm25 import BM25Retriever
from .chunker import CodeChunk, CodeEntityChunker
from .semantic import SemanticRetriever


def normalize_scores(
    scores: dict[str, float],
    method: str = "minmax",
) -> dict[str, float]:
    """Normalize score dictionary onto comparable range [0, 1].

    Supported methods:
      - 'minmax': (s - min) / (max - min) if max > min else 1.0 (or 0.0 if all zero)
      - 'max': s / max if max > 0 else 0.0
      - 'none': raw scores unchanged

    Args:
        scores: Mapping of entity_id -> float score.
        method: Normalization strategy ('minmax', 'max', 'none').

    Returns:
        Mapping of entity_id -> normalized float score.
    """
    if not scores:
        return {}

    if method == "none":
        return dict(scores)

    values = list(scores.values())
    max_val = max(values)
    min_val = min(values)

    if method == "minmax":
        if max_val == min_val:
            return {k: (1.0 if max_val > 0 else 0.0) for k in scores}
        range_val = max_val - min_val
        return {k: (v - min_val) / range_val for k, v in scores.items()}
    elif method == "max":
        if max_val <= 0:
            return {k: 0.0 for k in scores}
        return {k: max(0.0, v / max_val) for k, v in scores.items()}
    else:
        raise ValueError(
            f"Unknown normalization method: {method!r}. Choose 'minmax', 'max', or 'none'."
        )


class HybridRetriever(BaseRetriever):
    """Deterministic Hybrid Retriever combining BM25 and Semantic CodeBERT retrieval.

    Formula:
        hybrid_score = bm25_weight * normalized_bm25_score + semantic_weight * normalized_semantic_score
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever | None = None,
        semantic_retriever: SemanticRetriever | None = None,
        chunks: list[CodeChunk] | None = None,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        normalization: str = "minmax",
        candidate_k: int | None = None,
        repo_path: str | Path | None = None,
        chunker: CodeEntityChunker | None = None,
    ) -> None:
        """Initialize HybridRetriever.

        Args:
            bm25_retriever: Optional pre-configured BM25Retriever instance.
            semantic_retriever: Optional pre-configured SemanticRetriever instance.
            chunks: Optional list of CodeChunk objects to index in both retrievers.
            bm25_weight: Linear combination weight for BM25 (default: 0.5).
            semantic_weight: Linear combination weight for Semantic (default: 0.5).
            normalization: Score normalization strategy ('minmax', 'max', 'none').
            candidate_k: Optional number of candidate units to fetch from each sub-retriever.
            repo_path: Optional repository file path for reading source code.
            chunker: Optional CodeEntityChunker instance.
        """
        self._validate_weights(bm25_weight, semantic_weight)
        self.bm25_weight = float(bm25_weight)
        self.semantic_weight = float(semantic_weight)
        self.normalization = normalization.lower()
        if self.normalization not in ("minmax", "max", "none"):
            raise ValueError(
                f"Invalid normalization strategy: {normalization!r}. Must be 'minmax', 'max', or 'none'."
            )
        self.candidate_k = candidate_k
        self.repo_path = Path(repo_path) if repo_path else None
        self.chunker = chunker or CodeEntityChunker(repo_path=self.repo_path)

        self.bm25_retriever = bm25_retriever or BM25Retriever(
            chunks=chunks,
            repo_path=self.repo_path,
            chunker=self.chunker,
        )
        self.semantic_retriever = semantic_retriever or SemanticRetriever(
            chunks=chunks,
            repo_path=self.repo_path,
            chunker=self.chunker,
        )

        if chunks:
            # Ensure both retrievers are indexed
            if len(self.bm25_retriever) == 0:
                self.bm25_retriever.index(chunks)
            if len(self.semantic_retriever) == 0:
                self.semantic_retriever.index(chunks)

    @staticmethod
    def _validate_weights(bm25_weight: Any, semantic_weight: Any) -> None:
        """Validate combination weights."""
        if not isinstance(bm25_weight, (int, float)) or not isinstance(semantic_weight, (int, float)):
            raise ValueError(
                f"Weights must be numeric. Got bm25_weight={bm25_weight!r}, semantic_weight={semantic_weight!r}"
            )
        if math.isnan(bm25_weight) or math.isnan(semantic_weight) or math.isinf(bm25_weight) or math.isinf(semantic_weight):
            raise ValueError(
                f"Weights must be finite numbers. Got bm25_weight={bm25_weight}, semantic_weight={semantic_weight}"
            )
        if bm25_weight < 0 or semantic_weight < 0:
            raise ValueError(
                f"Weights must be non-negative. Got bm25_weight={bm25_weight}, semantic_weight={semantic_weight}"
            )
        if bm25_weight == 0 and semantic_weight == 0:
            raise ValueError("At least one weight must be greater than 0.")

    @property
    def chunks(self) -> list[CodeChunk]:
        """Access indexed CodeChunks."""
        return self.bm25_retriever.chunks or self.semantic_retriever.chunks

    def index(self, chunks: list[CodeChunk]) -> None:
        """Index repository chunks across both BM25 and Semantic retrievers."""
        self.bm25_retriever.index(chunks)
        self.semantic_retriever.index(chunks)

    def add_chunks(self, chunks: list[CodeChunk]) -> None:
        """Append repository chunks across both BM25 and Semantic retrievers."""
        self.bm25_retriever.add_chunks(chunks)
        self.semantic_retriever.add_chunks(chunks)

    def retrieve(
        self,
        question: str,
        repository: Any = None,
        k: int = 10,
        entity_type_filter: str | list[str] | None = None,
        candidate_k: int | None = None,
    ) -> list[RetrievedItem]:
        """Retrieve top-k code units using weighted combination of BM25 and Semantic scores.

        Args:
            question: Natural language question or query string.
            repository: Optional repository representation (CodeChunk list, KnowledgeGraph, or file path).
            k: Maximum number of retrieved items to return (k > 0).
            entity_type_filter: Optional entity type or list of entity types to filter by.
            candidate_k: Optional candidate pool size to fetch from each retriever.

        Returns:
            List of RetrievedItem objects sorted descending by hybrid score.
            Ties are broken deterministically by entity_id ascending.
        """
        if k <= 0:
            return []

        if not question or not question.strip():
            return []

        fetch_k = candidate_k or self.candidate_k or max(k * 2, k, 20)

        # Retrieve candidate items from both components
        bm25_items = self.bm25_retriever.retrieve(
            question=question,
            repository=repository,
            k=fetch_k,
            entity_type_filter=entity_type_filter,
        )
        semantic_items = self.semantic_retriever.retrieve(
            question=question,
            repository=repository,
            k=fetch_k,
            entity_type_filter=entity_type_filter,
        )

        if not bm25_items and not semantic_items:
            return []

        bm25_scores = {item.entity_id: item.score for item in bm25_items}
        sem_scores = {item.entity_id: item.score for item in semantic_items}

        bm25_item_map = {item.entity_id: item for item in bm25_items}
        sem_item_map = {item.entity_id: item for item in semantic_items}

        norm_bm25 = normalize_scores(bm25_scores, method=self.normalization)
        norm_sem = normalize_scores(sem_scores, method=self.normalization)

        # Candidate universe: union of all retrieved entity IDs
        all_entity_ids = set(bm25_scores.keys()) | set(sem_scores.keys())
        scored_candidates: list[tuple[float, str, RetrievedItem, float, float]] = []

        for entity_id in all_entity_ids:
            s_bm25_norm = norm_bm25.get(entity_id, 0.0)
            s_sem_norm = norm_sem.get(entity_id, 0.0)

            hybrid_score = (
                self.bm25_weight * s_bm25_norm + self.semantic_weight * s_sem_norm
            )

            # Retrieve base entity metadata from whichever retriever returned it
            base_item = bm25_item_map.get(entity_id) or sem_item_map[entity_id]

            scored_candidates.append(
                (hybrid_score, entity_id, base_item, s_bm25_norm, s_sem_norm)
            )

        # Deterministic sorting: primary key is -score (descending), secondary is entity_id (ascending)
        scored_candidates.sort(key=lambda rec: (-round(rec[0], 6), rec[1]))

        top_candidates = scored_candidates[:k]
        results: list[RetrievedItem] = []

        for hybrid_score, entity_id, base_item, s_bm25_norm, s_sem_norm in top_candidates:
            sources = []
            if entity_id in bm25_scores:
                sources.append("bm25")
            if entity_id in sem_scores:
                sources.append("semantic_codebert")

            provenance = {
                "retrieval_method": "hybrid",
                "sources": sources,
                "bm25_weight": self.bm25_weight,
                "semantic_weight": self.semantic_weight,
                "normalization": self.normalization,
                "bm25_raw_score": bm25_scores.get(entity_id),
                "bm25_normalized_score": round(s_bm25_norm, 6),
                "semantic_raw_score": sem_scores.get(entity_id),
                "semantic_normalized_score": round(s_sem_norm, 6),
                "combined_score": round(hybrid_score, 6),
                "hybrid_score": round(hybrid_score, 6),
                "bm25_provenance": bm25_item_map[entity_id].provenance if entity_id in bm25_item_map else {},
                "semantic_provenance": sem_item_map[entity_id].provenance if entity_id in sem_item_map else {},
            }

            item = RetrievedItem(
                entity_id=base_item.entity_id,
                file_path=base_item.file_path,
                source_code=base_item.source_code,
                score=round(hybrid_score, 6),
                retrieval_method="hybrid",
                provenance=provenance,
                entity_type=base_item.entity_type,
                name=base_item.name,
                qualified_name=base_item.qualified_name,
                start_line=base_item.start_line,
                end_line=base_item.end_line,
                metadata=base_item.metadata,
            )
            results.append(item)

        return results

    @classmethod
    def from_retrievers(
        cls,
        bm25_retriever: BM25Retriever,
        semantic_retriever: SemanticRetriever,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        normalization: str = "minmax",
        candidate_k: int | None = None,
    ) -> HybridRetriever:
        """Create a HybridRetriever from existing retriever instances."""
        return cls(
            bm25_retriever=bm25_retriever,
            semantic_retriever=semantic_retriever,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            normalization=normalization,
            candidate_k=candidate_k,
        )

    @classmethod
    def from_chunks(
        cls,
        chunks: list[CodeChunk],
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        normalization: str = "minmax",
        repo_path: str | Path | None = None,
        candidate_k: int | None = None,
    ) -> HybridRetriever:
        """Create and index a HybridRetriever from a list of CodeChunks."""
        return cls(
            chunks=chunks,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            normalization=normalization,
            candidate_k=candidate_k,
            repo_path=repo_path,
        )

    @classmethod
    def from_analysis_file(
        cls,
        analysis_file_path: str | Path,
        repo_path: str | Path | None = None,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        normalization: str = "minmax",
        candidate_k: int | None = None,
    ) -> HybridRetriever:
        """Create and index a HybridRetriever from an analyzer JSON file."""
        chunker = CodeEntityChunker(repo_path=repo_path)
        chunks = chunker.chunk_analysis_file(analysis_file_path)
        return cls(
            chunks=chunks,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            normalization=normalization,
            candidate_k=candidate_k,
            repo_path=repo_path,
            chunker=chunker,
        )

    @classmethod
    def from_knowledge_graph(
        cls,
        graph: Any,
        repo_path: str | Path | None = None,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        normalization: str = "minmax",
        candidate_k: int | None = None,
    ) -> HybridRetriever:
        """Create and index a HybridRetriever from an in-memory KnowledgeGraph."""
        chunker = CodeEntityChunker(repo_path=repo_path)
        chunks = chunker.chunk_knowledge_graph(graph)
        return cls(
            chunks=chunks,
            bm25_weight=bm25_weight,
            semantic_weight=semantic_weight,
            normalization=normalization,
            candidate_k=candidate_k,
            repo_path=repo_path,
            chunker=chunker,
        )

    def __len__(self) -> int:
        """Return the number of indexed chunks."""
        return max(len(self.bm25_retriever), len(self.semantic_retriever))
