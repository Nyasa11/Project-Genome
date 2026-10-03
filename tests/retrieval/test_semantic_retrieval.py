"""Focused tests for Phase 9: Semantic Retrieval with CodeBERT.

Tests include fast, deterministic unit tests using a mock embedder,
as well as end-to-end integration tests using the real CodeBERTEmbedder
with sample_analysis.json and sample_repo fixtures.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import numpy as np

from src.retrieval.base import BaseRetriever, RetrievedItem
from src.retrieval.chunker import CodeChunk, CodeEntityChunker
from src.retrieval.embeddings import CodeBERTEmbedder
from src.retrieval.index import CodeVectorIndex
from src.retrieval.semantic import SemanticRetriever
from src.knowledge_graph.builder import build_from_analysis

SAMPLE_ANALYSIS = Path("sample_analysis.json")
SAMPLE_REPO = Path("sample_repo")


class DummyDeterministicEmbedder:
    """Mock embedder providing deterministic 768-dimensional normalized vectors."""

    def __init__(self, model_name: str = "microsoft/codebert-base") -> None:
        self.model_name = model_name
        self._device = "cpu"

    def embed_texts(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        vecs = []
        for text in texts:
            # Deterministic pseudo-vector from text hash
            seed = sum(ord(c) for c in text) % (2**31 - 1)
            rng = np.random.RandomState(seed)
            v = rng.randn(768).astype(np.float32)
            norm = np.linalg.norm(v)
            vecs.append(v / (norm if norm > 0 else 1.0))
        return np.vstack(vecs)

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]

    def embed_chunks(self, chunks: list[CodeChunk], batch_size: int = 16) -> np.ndarray:
        texts = [chunk.text_representation for chunk in chunks]
        return self.embed_texts(texts)


class TestSemanticRetrieverUnit(unittest.TestCase):
    """Fast, deterministic unit tests for SemanticRetriever logic."""

    def setUp(self):
        self.embedder = DummyDeterministicEmbedder()
        self.chunk_a = CodeChunk(
            repository_id="repo",
            entity_id="func:a.py:authenticate_user",
            entity_type="function",
            name="authenticate_user",
            file_path="a.py",
            source_code="def authenticate_user(token): return True",
            text_representation="Function authenticate_user\nFile: a.py",
        )
        self.chunk_b = CodeChunk(
            repository_id="repo",
            entity_id="class:b.py:UserDatabase",
            entity_type="class",
            name="UserDatabase",
            file_path="b.py",
            source_code="class UserDatabase: pass",
            text_representation="Class UserDatabase\nFile: b.py",
        )
        self.chunk_c = CodeChunk(
            repository_id="repo",
            entity_id="method:c.py:UserDatabase.query",
            entity_type="method",
            name="query",
            file_path="c.py",
            source_code="def query(self, q): return []",
            text_representation="Method UserDatabase.query\nFile: c.py",
        )
        self.chunks = [self.chunk_a, self.chunk_b, self.chunk_c]

    def test_is_base_retriever(self):
        retriever = SemanticRetriever(embedder=self.embedder)
        self.assertIsInstance(retriever, BaseRetriever)

    def test_empty_init_and_len(self):
        retriever = SemanticRetriever(embedder=self.embedder)
        self.assertEqual(len(retriever), 0)
        self.assertEqual(retriever.chunks, [])

    def test_index_chunks_and_len(self):
        retriever = SemanticRetriever(embedder=self.embedder)
        retriever.index_chunks(self.chunks)
        self.assertEqual(len(retriever), 3)
        self.assertEqual(len(retriever.chunks), 3)

    def test_index_method_callable_and_retrieves(self):
        """Verify that retriever.index(chunks) is callable, indexes successfully, and enables retrieval."""
        retriever = SemanticRetriever(embedder=self.embedder)
        self.assertEqual(len(retriever), 0)

        # Call the public .index() method directly
        retriever.index(self.chunks)

        self.assertEqual(len(retriever), 3)
        self.assertEqual(len(retriever.chunks), 3)

        results = retriever.retrieve("authenticate", k=2)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], RetrievedItem)
        self.assertEqual(results[0].retrieval_method, "semantic_codebert")

    def test_add_chunks_incrementally(self):
        retriever = SemanticRetriever(embedder=self.embedder)
        retriever.index_chunks([self.chunk_a])
        self.assertEqual(len(retriever), 1)

        retriever.add_chunks([self.chunk_b, self.chunk_c])
        self.assertEqual(len(retriever), 3)

    def test_from_chunks_factory(self):
        retriever = SemanticRetriever.from_chunks(self.chunks, embedder=self.embedder)
        self.assertEqual(len(retriever), 3)

    def test_empty_query_returns_empty(self):
        retriever = SemanticRetriever.from_chunks(self.chunks, embedder=self.embedder)
        self.assertEqual(retriever.retrieve("", k=5), [])
        self.assertEqual(retriever.retrieve("   \t\n  ", k=5), [])

    def test_empty_corpus_returns_empty(self):
        retriever = SemanticRetriever(embedder=self.embedder)
        self.assertEqual(retriever.retrieve("authenticate user", k=5), [])

    def test_top_k_bounds(self):
        retriever = SemanticRetriever.from_chunks(self.chunks, embedder=self.embedder)
        self.assertEqual(len(retriever.retrieve("query", k=1)), 1)
        self.assertEqual(len(retriever.retrieve("query", k=2)), 2)
        self.assertEqual(len(retriever.retrieve("query", k=10)), 3)
        self.assertEqual(retriever.retrieve("query", k=0), [])
        self.assertEqual(retriever.retrieve("query", k=-5), [])

    def test_entity_type_filtering(self):
        retriever = SemanticRetriever.from_chunks(self.chunks, embedder=self.embedder)

        res_class = retriever.retrieve("database", k=5, entity_type_filter="class")
        self.assertEqual(len(res_class), 1)
        self.assertEqual(res_class[0].entity_type, "class")
        self.assertEqual(res_class[0].entity_id, self.chunk_b.entity_id)

        res_multi = retriever.retrieve("user", k=5, entity_type_filter=["function", "method"])
        self.assertEqual(len(res_multi), 2)
        for r in res_multi:
            self.assertIn(r.entity_type, ["function", "method"])

        res_none = retriever.retrieve("user", k=5, entity_type_filter="nonexistent")
        self.assertEqual(res_none, [])

    def test_retrieved_item_structure_and_provenance(self):
        retriever = SemanticRetriever.from_chunks(self.chunks, embedder=self.embedder)
        results = retriever.retrieve("authenticate", k=1)
        self.assertEqual(len(results), 1)
        item = results[0]

        self.assertIsInstance(item, RetrievedItem)
        self.assertEqual(item.retrieval_method, "semantic_codebert")
        self.assertIsInstance(item.score, float)
        self.assertTrue(item.entity_id)
        self.assertTrue(item.file_path)

        prov = item.provenance
        self.assertEqual(prov["model_name"], "microsoft/codebert-base")
        self.assertEqual(prov["similarity_metric"], "cosine")
        self.assertIn("similarity_score", prov)

        # Serialization roundtrip
        item_dict = item.to_dict()
        self.assertEqual(item_dict["retrieval_method"], "semantic_codebert")
        self.assertEqual(item_dict["entity_id"], item.entity_id)

    def test_deterministic_tie_breaking(self):
        # Create a mock embedder where all chunks and queries have identical vector
        class FlatEmbedder:
            model_name = "flat"
            def embed_text(self, text):
                v = np.ones(768, dtype=np.float32)
                return v / np.linalg.norm(v)
            def embed_chunks(self, chunks, batch_size=16):
                return np.vstack([self.embed_text("") for _ in chunks])

        chunks = [
            CodeChunk(
                repository_id="repo",
                entity_id="entity_z",
                entity_type="function",
                name="z",
                file_path="z.py",
                text_representation="Function z",
            ),
            CodeChunk(
                repository_id="repo",
                entity_id="entity_a",
                entity_type="function",
                name="a",
                file_path="a.py",
                text_representation="Function a",
            ),
        ]
        retriever = SemanticRetriever.from_chunks(chunks, embedder=FlatEmbedder())
        results = retriever.retrieve("test query", k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].score, results[1].score)
        # entity_a must precede entity_z alphabetically on tie
        self.assertEqual(results[0].entity_id, "entity_a")
        self.assertEqual(results[1].entity_id, "entity_z")

    def test_save_and_load(self):
        retriever = SemanticRetriever.from_chunks(self.chunks, embedder=self.embedder)
        with tempfile.TemporaryDirectory() as tmp_dir:
            retriever.save(tmp_dir)
            loaded = SemanticRetriever.load(tmp_dir, embedder=self.embedder)
            self.assertEqual(len(loaded), 3)

            res_orig = retriever.retrieve("query", k=3)
            res_load = loaded.retrieve("query", k=3)
            self.assertEqual(
                [r.entity_id for r in res_orig],
                [r.entity_id for r in res_load],
            )

    def test_repository_argument_passed_to_retrieve(self):
        retriever = SemanticRetriever(embedder=self.embedder)
        results = retriever.retrieve("authenticate", repository=self.chunks, k=2)
        self.assertEqual(len(results), 2)


class TestSemanticRetrieverCodeBERTIntegration(unittest.TestCase):
    """End-to-end integration tests using real CodeBERTEmbedder and sample fixtures."""

    @classmethod
    def setUpClass(cls):
        cls.embedder = CodeBERTEmbedder()
        cls.chunker = CodeEntityChunker(repo_path=SAMPLE_REPO)
        cls.chunks = cls.chunker.chunk_analysis_file(SAMPLE_ANALYSIS)
        cls.retriever = SemanticRetriever.from_chunks(
            chunks=cls.chunks,
            embedder=cls.embedder,
            repo_path=SAMPLE_REPO,
            batch_size=8,
        )

    def test_integration_indexing_and_retrieval(self):
        self.assertEqual(len(self.retriever), 16)

        results = self.retriever.retrieve("def create_user(self, name: str):", k=3)
        self.assertEqual(len(results), 3)

        top_result = results[0]
        self.assertIsInstance(top_result, RetrievedItem)
        self.assertEqual(top_result.retrieval_method, "semantic_codebert")
        self.assertTrue(top_result.entity_id)
        self.assertTrue(np.isfinite(top_result.score))
        self.assertGreaterEqual(top_result.score, -1.0)
        self.assertLessEqual(top_result.score, 1.0 + 1e-5)
        self.assertGreaterEqual(results[0].score, results[1].score)
        self.assertGreaterEqual(results[1].score, results[2].score)

    def test_from_analysis_file_factory(self):
        retriever = SemanticRetriever.from_analysis_file(
            analysis_file_path=SAMPLE_ANALYSIS,
            repo_path=SAMPLE_REPO,
            embedder=self.embedder,
            batch_size=8,
        )
        self.assertEqual(len(retriever), 16)
        results = retriever.retrieve("UserService", k=2)
        self.assertEqual(len(results), 2)

    def test_from_knowledge_graph_factory(self):
        kg = build_from_analysis(SAMPLE_ANALYSIS)
        retriever = SemanticRetriever.from_knowledge_graph(
            graph=kg,
            repo_path=SAMPLE_REPO,
            embedder=self.embedder,
            batch_size=8,
        )
        self.assertEqual(len(retriever), 16)
        results = retriever.retrieve("BaseModel", k=2)
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
