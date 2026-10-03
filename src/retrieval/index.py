"""Code Vector Index for ProjectGenome.

Stores CodeChunk metadata and dense embeddings, providing fast top-K cosine similarity search,
persistence, and retrieval functionality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

from .chunker import CodeChunk


class CodeVectorIndex:
    """In-memory vector index for storing and querying CodeChunks by embedding similarity."""

    def __init__(self) -> None:
        """Initialize empty vector index."""
        self.chunks: list[CodeChunk] = []
        self.embeddings: np.ndarray | None = None

    def add_chunks(self, chunks: list[CodeChunk], embeddings: np.ndarray) -> None:
        """Add chunks and corresponding embeddings to the index.

        Args:
            chunks: List of CodeChunk objects.
            embeddings: 2D numpy array of shape (N, D).
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch between number of chunks ({len(chunks)}) and embeddings ({len(embeddings)})"
            )

        if not chunks:
            return

        embeddings = np.asarray(embeddings, dtype=np.float32)
        # Ensure normalized vectors
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_embeddings = embeddings / norms

        if self.embeddings is None or len(self.chunks) == 0:
            self.chunks = list(chunks)
            self.embeddings = normalized_embeddings
        else:
            self.chunks.extend(chunks)
            self.embeddings = np.vstack([self.embeddings, normalized_embeddings])

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        entity_type_filter: str | list[str] | None = None,
    ) -> list[tuple[CodeChunk, float]]:
        """Search the index for the top_k most similar chunks to a query vector.

        Args:
            query_embedding: 1D (D,) or 2D (1, D) numpy array query vector.
            top_k: Maximum number of results to return.
            entity_type_filter: Optional entity_type string or list of strings to filter by.

        Returns:
            List of (CodeChunk, float_similarity_score) tuples sorted descending by score.
        """
        if top_k <= 0 or self.embeddings is None or len(self.chunks) == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).ravel()
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm

        # Compute cosine similarities via matrix-vector multiplication
        scores = np.dot(self.embeddings, query)

        # Filter by entity type if specified
        if entity_type_filter:
            if isinstance(entity_type_filter, str):
                allowed_types = {entity_type_filter.lower()}
            else:
                allowed_types = {t.lower() for t in entity_type_filter}

            valid_indices = [
                i for i, chunk in enumerate(self.chunks) if chunk.entity_type.lower() in allowed_types
            ]
            if not valid_indices:
                return []
            candidate_indices = np.array(valid_indices)
            candidate_scores = scores[candidate_indices]
            top_candidate_pos = np.argsort(candidate_scores)[::-1][:top_k]
            top_indices = candidate_indices[top_candidate_pos]
        else:
            top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[tuple[CodeChunk, float]] = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))

        return results

    def save(self, index_dir: str | Path) -> None:
        """Save the index (chunks metadata + embeddings array) to directory."""
        dir_path = Path(index_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save chunks metadata
        chunks_json_path = dir_path / "chunks.json"
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        with open(chunks_json_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2)

        # Save embeddings numpy array
        if self.embeddings is not None:
            npy_path = dir_path / "embeddings.npy"
            np.save(npy_path, self.embeddings)

    @classmethod
    def load(cls, index_dir: str | Path) -> CodeVectorIndex:
        """Load an index from directory."""
        dir_path = Path(index_dir)
        chunks_json_path = dir_path / "chunks.json"
        npy_path = dir_path / "embeddings.npy"

        if not chunks_json_path.is_file():
            raise FileNotFoundError(f"Index chunks file not found: {chunks_json_path}")

        with open(chunks_json_path, encoding="utf-8") as f:
            raw_chunks = json.load(f)

        chunks = [CodeChunk(**c) for c in raw_chunks]

        embeddings = None
        if npy_path.is_file():
            embeddings = np.load(npy_path)

        index = cls()
        if chunks and embeddings is not None:
            index.add_chunks(chunks, embeddings)

        return index

    def __len__(self) -> int:
        """Return number of indexed chunks."""
        return len(self.chunks)
