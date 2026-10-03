"""BM25 (lexical) Retrieval Engine for ProjectGenome.

Implements Okapi BM25 keyword search over repository code chunks (File, Class, Function, Method),
conforming to the BaseRetriever common interface and producing normalized RetrievedItem outputs.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .base import BaseRetriever, RetrievedItem
from .chunker import CodeChunk, CodeEntityChunker


def tokenize_code(text: str) -> list[str]:
    """Tokenize code and natural-language text for BM25 retrieval.

    Splits on punctuation, whitespace, and snake_case delimiters,
    and breaks camelCase / PascalCase identifiers into constituent sub-tokens.
    All tokens are lowercased to ensure case-insensitive matching.

    Args:
        text: Input string (code, signature, docstring, or natural language query).

    Returns:
        List of lowercase string tokens.
    """
    if not text:
        return []

    words = re.findall(r"[A-Za-z0-9]+", text)
    tokens: list[str] = []
    for w in words:
        w_lower = w.lower()
        tokens.append(w_lower)
        # Extract camelCase / PascalCase sub-words (e.g. UserService -> user, service)
        subparts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z0-9]|\b)", w)
        if len(subparts) > 1:
            for sp in subparts:
                sp_lower = sp.lower()
                if sp_lower != w_lower:
                    tokens.append(sp_lower)
    return tokens


def get_canonical_chunk_text(chunk: CodeChunk) -> str:
    """Extract canonical searchable text from a CodeChunk.

    Uses chunk.text_representation when available; otherwise constructs
    the canonical representation from entity metadata (name, qualified_name,
    file_path, signature, docstring, source_code).
    """
    if chunk.text_representation and chunk.text_representation.strip():
        return chunk.text_representation

    parts: list[str] = []
    if chunk.entity_type:
        parts.append(f"Entity Type: {chunk.entity_type.capitalize()}")
    if chunk.name:
        parts.append(f"Name: {chunk.name}")
    if chunk.qualified_name and chunk.qualified_name != chunk.name:
        parts.append(f"Qualified Name: {chunk.qualified_name}")
    if chunk.file_path:
        parts.append(f"File: {chunk.file_path}")
    if chunk.signature:
        parts.append(f"Signature: {chunk.signature}")
    if chunk.docstring:
        parts.append(f"Docstring: {chunk.docstring}")
    if chunk.source_code:
        parts.append(f"Source Code:\n{chunk.source_code}")
    return "\n".join(parts)


class BM25Retriever(BaseRetriever):
    """Lexical baseline retriever using Okapi BM25 for code search.

    Implements the ProjectGenome common retrieval interface:
        retrieve(question, repository, k) -> list[RetrievedItem]
    """

    def __init__(
        self,
        chunks: list[CodeChunk] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        repo_path: str | Path | None = None,
        chunker: CodeEntityChunker | None = None,
    ) -> None:
        """Initialize BM25Retriever.

        Args:
            chunks: Optional list of CodeChunk objects to index immediately.
            k1: BM25 term frequency saturation parameter (default: 1.5).
            b: BM25 document length normalization parameter (default: 0.75).
            repo_path: Optional path to repository files for reading source code.
            chunker: Optional CodeEntityChunker instance.
        """
        self.k1 = float(k1)
        self.b = float(b)
        self.repo_path = Path(repo_path) if repo_path else None
        self.chunker = chunker or CodeEntityChunker(repo_path=self.repo_path)

        self.chunks: list[CodeChunk] = []
        self.doc_lens: list[int] = []
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}
        self.inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)

        if chunks:
            self.index(chunks)

    def index(self, chunks: list[CodeChunk]) -> None:
        """Build the inverted index and BM25 statistics over the provided chunks.

        Args:
            chunks: List of CodeChunk objects.
        """
        self.chunks = list(chunks)
        self.doc_lens = []
        self.inverted_index = defaultdict(list)
        self.idf = {}

        n_docs = len(self.chunks)
        if n_docs == 0:
            self.avgdl = 0.0
            return

        total_tokens = 0
        df: dict[str, int] = Counter()

        for idx, chunk in enumerate(self.chunks):
            text = get_canonical_chunk_text(chunk)
            tokens = tokenize_code(text)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_tokens += doc_len

            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self.inverted_index[term].append((idx, count))
                df[term] += 1

        self.avgdl = total_tokens / n_docs if n_docs > 0 else 0.0

        # Compute Robertson-Spärck Jones IDF with Lucene/Okapi smoothing
        for term, doc_freq in df.items():
            self.idf[term] = math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    def add_chunks(self, chunks: list[CodeChunk]) -> None:
        """Add chunks to the retriever and rebuild index."""
        combined = list(self.chunks) + list(chunks)
        self.index(combined)

    def retrieve(
        self,
        question: str,
        repository: Any = None,
        k: int = 10,
        entity_type_filter: str | list[str] | None = None,
    ) -> list[RetrievedItem]:
        """Retrieve top-k relevant repository code units for a question using BM25.

        Args:
            question: Natural language question or query string.
            repository: Optional repository representation (list of CodeChunk, KnowledgeGraph,
                        or analysis JSON file path). If provided, indexes the repository.
            k: Maximum number of retrieved items to return (k > 0).
            entity_type_filter: Optional entity type or list of entity types to restrict to
                                (e.g. 'function', ['class', 'method']).

        Returns:
            List of RetrievedItem objects sorted descending by BM25 score.
            Ties are broken deterministically by entity_id ascending.
        """
        if k <= 0:
            return []

        # If a repository is explicitly provided, prepare chunks
        if repository is not None:
            self._handle_repository_input(repository)

        if not self.chunks or len(self.chunks) == 0:
            return []

        query_tokens = tokenize_code(question)
        if not query_tokens:
            return []

        # Normalize entity_type_filter
        allowed_types: set[str] | None = None
        if entity_type_filter is not None:
            if isinstance(entity_type_filter, str):
                allowed_types = {entity_type_filter.lower()}
            else:
                allowed_types = {t.lower() for t in entity_type_filter}

        q_tf = Counter(query_tokens)
        scores: dict[int, float] = defaultdict(float)
        matched_terms_per_doc: dict[int, set[str]] = defaultdict(set)
        tf_per_doc: dict[int, dict[str, int]] = defaultdict(dict)

        for term, q_weight in q_tf.items():
            if term not in self.inverted_index:
                continue

            term_idf = self.idf[term]
            postings = self.inverted_index[term]

            for doc_idx, tf in postings:
                chunk = self.chunks[doc_idx]
                if allowed_types is not None and chunk.entity_type.lower() not in allowed_types:
                    continue

                doc_len = self.doc_lens[doc_idx]
                len_norm = (doc_len / self.avgdl) if self.avgdl > 0 else 1.0
                denom = tf + self.k1 * (1.0 - self.b + self.b * len_norm)
                term_score = term_idf * ((tf * (self.k1 + 1.0)) / denom)

                scores[doc_idx] += term_score * q_weight
                matched_terms_per_doc[doc_idx].add(term)
                tf_per_doc[doc_idx][term] = tf

        if not scores:
            return []

        # Filter strictly positive scores and construct candidate records
        scored_candidates: list[tuple[int, float]] = [
            (doc_idx, score) for doc_idx, score in scores.items() if score > 0.0
        ]

        # Deterministic sorting: primary key is -score (descending), secondary is entity_id (ascending)
        scored_candidates.sort(
            key=lambda item: (-round(item[1], 8), self.chunks[item[0]].entity_id)
        )

        top_candidates = scored_candidates[:k]
        results: list[RetrievedItem] = []

        for doc_idx, score in top_candidates:
            chunk = self.chunks[doc_idx]
            matched_terms = sorted(matched_terms_per_doc[doc_idx])
            provenance = {
                "query_tokens": query_tokens,
                "matched_terms": matched_terms,
                "term_frequencies": {t: tf_per_doc[doc_idx][t] for t in matched_terms},
                "doc_length": self.doc_lens[doc_idx],
                "avg_doc_length": round(self.avgdl, 4),
                "k1": self.k1,
                "b": self.b,
            }

            item = RetrievedItem.from_chunk(
                chunk=chunk,
                score=round(score, 6),
                retrieval_method="bm25",
                provenance=provenance,
            )
            results.append(item)

        return results

    def _handle_repository_input(self, repository: Any) -> None:
        """Helper to index repository chunks if passed to retrieve()."""
        if isinstance(repository, list):
            if repository and isinstance(repository[0], CodeChunk):
                self.index(repository)
            return

        # Check for KnowledgeGraph (duck typing or isinstance)
        if hasattr(repository, "nodes") or hasattr(repository, "underlying_graph"):
            chunks = self.chunker.chunk_knowledge_graph(repository)
            self.index(chunks)
            return

        # Check for filepath to analysis JSON
        if isinstance(repository, (str, Path)):
            path = Path(repository)
            if path.is_file():
                chunks = self.chunker.chunk_analysis_file(path)
                self.index(chunks)
                return

    @classmethod
    def from_chunks(
        cls,
        chunks: list[CodeChunk],
        k1: float = 1.5,
        b: float = 0.75,
        repo_path: str | Path | None = None,
    ) -> BM25Retriever:
        """Create and index a BM25Retriever from a list of CodeChunks."""
        return cls(chunks=chunks, k1=k1, b=b, repo_path=repo_path)

    @classmethod
    def from_analysis_file(
        cls,
        analysis_file_path: str | Path,
        repo_path: str | Path | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Retriever:
        """Create and index a BM25Retriever from an analyzer JSON file."""
        chunker = CodeEntityChunker(repo_path=repo_path)
        chunks = chunker.chunk_analysis_file(analysis_file_path)
        return cls(chunks=chunks, k1=k1, b=b, repo_path=repo_path, chunker=chunker)

    @classmethod
    def from_knowledge_graph(
        cls,
        graph: Any,
        repo_path: str | Path | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Retriever:
        """Create and index a BM25Retriever from an in-memory KnowledgeGraph."""
        chunker = CodeEntityChunker(repo_path=repo_path)
        chunks = chunker.chunk_knowledge_graph(graph)
        return cls(chunks=chunks, k1=k1, b=b, repo_path=repo_path, chunker=chunker)

    def save(self, index_dir: str | Path) -> None:
        """Save the indexed chunks and BM25 configuration to a directory."""
        dir_path = Path(index_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        chunks_json_path = dir_path / "chunks.json"
        config_path = dir_path / "bm25_config.json"

        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        with open(chunks_json_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2)

        config_data = {
            "k1": self.k1,
            "b": self.b,
            "avgdl": self.avgdl,
            "num_chunks": len(self.chunks),
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

    @classmethod
    def load(cls, index_dir: str | Path, repo_path: str | Path | None = None) -> BM25Retriever:
        """Load an indexed BM25Retriever from a directory."""
        dir_path = Path(index_dir)
        chunks_json_path = dir_path / "chunks.json"
        if not chunks_json_path.is_file():
            raise FileNotFoundError(f"Chunks file not found: {chunks_json_path}")

        with open(chunks_json_path, encoding="utf-8") as f:
            raw_chunks = json.load(f)

        chunks = [CodeChunk(**c) for c in raw_chunks]

        config_path = dir_path / "bm25_config.json"
        k1 = 1.5
        b = 0.75
        if config_path.is_file():
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
                k1 = cfg.get("k1", 1.5)
                b = cfg.get("b", 0.75)

        return cls(chunks=chunks, k1=k1, b=b, repo_path=repo_path)

    def __len__(self) -> int:
        """Return number of indexed chunks."""
        return len(self.chunks)
