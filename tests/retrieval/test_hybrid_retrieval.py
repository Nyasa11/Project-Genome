"""Focused tests for Phase 11: B4 Hybrid Retrieval.

Tests cover:
- Common BaseRetriever interface compatibility
- Score combination and normalization
- Default (0.5 / 0.5) and custom weightings
- Invalid weight rejection
- Deduplication: merging entities present in both retrievers
- Single-source retention: entities in only one retriever are kept
- Top-k and entity-type filtering
- Deterministic tie-breaking by entity_id
- Full provenance recording
- Handling of empty result sets
- End-to-end integration with sample fixtures
"""

from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np

from src.retrieval.base import BaseRetriever, RetrievedItem
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.chunker import CodeChunk, CodeEntityChunker
from src.retrieval.embeddings import CodeBERTEmbedder
from src.retrieval.hybrid import HybridRetriever, normalize_scores
from src.retrieval.semantic import SemanticRetriever
from src.knowledge_graph.builder import build_from_analysis

SAMPLE_ANALYSIS = Path("sample_analysis.json")
SAMPLE_REPO = Path("sample_repo")


class MockFixedRetriever(BaseRetriever):
    """Mock retriever returning a predefined list of RetrievedItems for testing."""

    def __init__(self, fixed_items: list[RetrievedItem] | None = None) -> None:
        self.fixed_items = fixed_items or []

    def retrieve(
        self,
        question: str,
        repository: object = None,
        k: int = 10,
        entity_type_filter: str | list[str] | None = None,
    ) -> list[RetrievedItem]:
        items = list(self.fixed_items)
        if entity_type_filter:
            allowed = {entity_type_filter.lower()} if isinstance(entity_type_filter, str) else {t.lower() for t in entity_type_filter}
            items = [it for it in items if (it.entity_type or "").lower() in allowed]
        return items[:k]

    def index(self, chunks: list[CodeChunk]) -> None:
        pass


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


def _make_item(
    entity_id: str,
    score: float,
    method: str = "bm25",
    entity_type: str = "function",
    file_path: str = "a.py",
) -> RetrievedItem:
    return RetrievedItem(
        entity_id=entity_id,
        file_path=file_path,
        source_code=f"def {entity_id}(): pass",
        score=score,
        retrieval_method=method,
        entity_type=entity_type,
        name=entity_id.split(":")[-1],
        qualified_name=entity_id.split(":")[-1],
        provenance={"original_score": score},
    )


class TestNormalizeScores(unittest.TestCase):
    """Test score normalization routines."""

    def test_empty_scores(self):
        self.assertEqual(normalize_scores({}), {})

    def test_minmax_normalization(self):
        raw = {"e1": 10.0, "e2": 6.0, "e3": 2.0}
        norm = normalize_scores(raw, method="minmax")
        self.assertAlmostEqual(norm["e1"], 1.0)
        self.assertAlmostEqual(norm["e2"], 0.5)
        self.assertAlmostEqual(norm["e3"], 0.0)

    def test_minmax_identical_scores(self):
        raw = {"e1": 5.0, "e2": 5.0}
        norm = normalize_scores(raw, method="minmax")
        self.assertAlmostEqual(norm["e1"], 1.0)
        self.assertAlmostEqual(norm["e2"], 1.0)

    def test_max_normalization(self):
        raw = {"e1": 10.0, "e2": 5.0}
        norm = normalize_scores(raw, method="max")
        self.assertAlmostEqual(norm["e1"], 1.0)
        self.assertAlmostEqual(norm["e2"], 0.5)

    def test_unknown_normalization_raises(self):
        with self.assertRaises(ValueError):
            normalize_scores({"e1": 1.0}, method="invalid_method")


class TestHybridRetrieverInitAndValidation(unittest.TestCase):
    """Test initialization, BaseRetriever contract, and weight validation."""

    def test_is_base_retriever(self):
        retriever = HybridRetriever(
            bm25_retriever=MockFixedRetriever(),
            semantic_retriever=MockFixedRetriever(),
        )
        self.assertIsInstance(retriever, BaseRetriever)

    def test_default_weights(self):
        retriever = HybridRetriever(
            bm25_retriever=MockFixedRetriever(),
            semantic_retriever=MockFixedRetriever(),
        )
        self.assertEqual(retriever.bm25_weight, 0.5)
        self.assertEqual(retriever.semantic_weight, 0.5)
        self.assertEqual(retriever.normalization, "minmax")

    def test_custom_weights(self):
        retriever = HybridRetriever(
            bm25_retriever=MockFixedRetriever(),
            semantic_retriever=MockFixedRetriever(),
            bm25_weight=0.7,
            semantic_weight=0.3,
            normalization="max",
        )
        self.assertEqual(retriever.bm25_weight, 0.7)
        self.assertEqual(retriever.semantic_weight, 0.3)
        self.assertEqual(retriever.normalization, "max")

    def test_invalid_weights_rejected(self):
        # Negative weights
        with self.assertRaises(ValueError):
            HybridRetriever(bm25_weight=-0.1, semantic_weight=0.5)
        with self.assertRaises(ValueError):
            HybridRetriever(bm25_weight=0.5, semantic_weight=-0.2)

        # Both zero
        with self.assertRaises(ValueError):
            HybridRetriever(bm25_weight=0.0, semantic_weight=0.0)

        # Non-numeric
        with self.assertRaises(ValueError):
            HybridRetriever(bm25_weight="heavy", semantic_weight=0.5)  # type: ignore

        # NaN / Inf
        with self.assertRaises(ValueError):
            HybridRetriever(bm25_weight=float("nan"), semantic_weight=0.5)
        with self.assertRaises(ValueError):
            HybridRetriever(bm25_weight=float("inf"), semantic_weight=0.5)

    def test_invalid_normalization_rejected(self):
        with self.assertRaises(ValueError):
            HybridRetriever(normalization="unsupported")


class TestHybridScoreCombination(unittest.TestCase):
    """Test weighted score combination and deduplication."""

    def test_default_half_half_combination(self):
        # e1 in both: bm25 raw=10.0 (norm=1.0), sem raw=0.9 (norm=1.0)
        # e2 in both: bm25 raw=2.0 (norm=0.0), sem raw=0.1 (norm=0.0)
        bm25_items = [_make_item("e1", 10.0, "bm25"), _make_item("e2", 2.0, "bm25")]
        sem_items = [_make_item("e1", 0.9, "semantic_codebert"), _make_item("e2", 0.1, "semantic_codebert")]

        retriever = HybridRetriever(
            bm25_retriever=MockFixedRetriever(bm25_items),
            semantic_retriever=MockFixedRetriever(sem_items),
            bm25_weight=0.5,
            semantic_weight=0.5,
        )

        results = retriever.retrieve("query", k=2)
        self.assertEqual(len(results), 2)

        # e1: 0.5 * 1.0 + 0.5 * 1.0 = 1.0
        self.assertEqual(results[0].entity_id, "e1")
        self.assertAlmostEqual(results[0].score, 1.0)

        # e2: 0.5 * 0.0 + 0.5 * 0.0 = 0.0
        self.assertEqual(results[1].entity_id, "e2")
        self.assertAlmostEqual(results[1].score, 0.0)

    def test_custom_weights_combination(self):
        # bm25_weight=0.8, semantic_weight=0.2
        bm25_items = [_make_item("e1", 10.0), _make_item("e2", 5.0)]
        sem_items = [_make_item("e1", 0.8), _make_item("e2", 0.4)]

        retriever = HybridRetriever(
            bm25_retriever=MockFixedRetriever(bm25_items),
            semantic_retriever=MockFixedRetriever(sem_items),
            bm25_weight=0.8,
            semantic_weight=0.2,
            normalization="max",
        )

        results = retriever.retrieve("query", k=2)
        self.assertEqual(len(results), 2)
        # e1: 0.8 * 1.0 + 0.2 * 1.0 = 1.0
        self.assertAlmostEqual(results[0].score, 1.0)
        # e2: 0.8 * 0.5 + 0.2 * 0.5 = 0.5
        self.assertAlmostEqual(results[1].score, 0.5)

    def test_merging_duplicates_and_preserving_single_source_entities(self):
        # e1: present in both
        # e_bm25: present ONLY in BM25
        # e_sem: present ONLY in Semantic
        bm25_items = [_make_item("e1", 10.0, "bm25"), _make_item("e_bm25", 5.0, "bm25")]
        sem_items = [_make_item("e1", 0.9, "semantic_codebert"), _make_item("e_sem", 0.6, "semantic_codebert")]

        retriever = HybridRetriever(
            bm25_retriever=MockFixedRetriever(bm25_items),
            semantic_retriever=MockFixedRetriever(sem_items),
            bm25_weight=0.5,
            semantic_weight=0.5,
            normalization="max",
        )

        results = retriever.retrieve("query", k=5)
        # Exactly 3 unique entities merged (no duplicates)
        self.assertEqual(len(results), 3)

        retrieved_ids = [r.entity_id for r in results]
        self.assertIn("e1", retrieved_ids)
        self.assertIn("e_bm25", retrieved_ids)
        self.assertIn("e_sem", retrieved_ids)

        # Check provenance sources
        res_map = {r.entity_id: r for r in results}
        self.assertEqual(sorted(res_map["e1"].provenance["sources"]), ["bm25", "semantic_codebert"])
        self.assertEqual(res_map["e_bm25"].provenance["sources"], ["bm25"])
        self.assertEqual(res_map["e_sem"].provenance["sources"], ["semantic_codebert"])


    def test_empty_subretriever_results(self):
        # BM25 empty, semantic has results
        sem_items = [_make_item("e1", 0.9, "semantic_codebert")]
        retriever_sem_only = HybridRetriever(
            bm25_retriever=MockFixedRetriever([]),
            semantic_retriever=MockFixedRetriever(sem_items),
            normalization="max",
        )
        res_sem = retriever_sem_only.retrieve("query", k=3)
        self.assertEqual(len(res_sem), 1)
        self.assertEqual(res_sem[0].entity_id, "e1")
        self.assertAlmostEqual(res_sem[0].score, 0.5 * 1.0)

        # Semantic empty, BM25 has results
        bm25_items = [_make_item("e2", 8.0, "bm25")]
        retriever_bm25_only = HybridRetriever(
            bm25_retriever=MockFixedRetriever(bm25_items),
            semantic_retriever=MockFixedRetriever([]),
            normalization="max",
        )
        res_bm25 = retriever_bm25_only.retrieve("query", k=3)
        self.assertEqual(len(res_bm25), 1)
        self.assertEqual(res_bm25[0].entity_id, "e2")
        self.assertAlmostEqual(res_bm25[0].score, 0.5 * 1.0)

        # Both empty
        retriever_both_empty = HybridRetriever(
            bm25_retriever=MockFixedRetriever([]),
            semantic_retriever=MockFixedRetriever([]),
        )
        self.assertEqual(retriever_both_empty.retrieve("query", k=3), [])

    def test_empty_query_and_non_positive_k(self):
        bm25_items = [_make_item("e1", 10.0)]
        retriever = HybridRetriever(
            bm25_retriever=MockFixedRetriever(bm25_items),
            semantic_retriever=MockFixedRetriever(bm25_items),
        )
        self.assertEqual(retriever.retrieve("", k=5), [])
        self.assertEqual(retriever.retrieve("   ", k=5), [])
        self.assertEqual(retriever.retrieve("valid", k=0), [])
        self.assertEqual(retriever.retrieve("valid", k=-3), [])


class TestTopKAndFiltering(unittest.TestCase):
    """Test top_k clamping and entity_type_filter delegation."""

    def setUp(self):
        self.items = [
            _make_item("f1", 10.0, entity_type="function"),
            _make_item("c1", 8.0, entity_type="class"),
            _make_item("m1", 6.0, entity_type="method"),
            _make_item("f2", 4.0, entity_type="function"),
        ]
        self.retriever = HybridRetriever(
            bm25_retriever=MockFixedRetriever(self.items),
            semantic_retriever=MockFixedRetriever(self.items),
        )

    def test_top_k(self):
        self.assertEqual(len(self.retriever.retrieve("q", k=1)), 1)
        self.assertEqual(len(self.retriever.retrieve("q", k=2)), 2)
        self.assertEqual(len(self.retriever.retrieve("q", k=10)), 4)

    def test_entity_type_filter_single(self):
        results = self.retriever.retrieve("q", k=5, entity_type_filter="class")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].entity_type, "class")
        self.assertEqual(results[0].entity_id, "c1")

    def test_entity_type_filter_multiple(self):
        results = self.retriever.retrieve("q", k=5, entity_type_filter=["function", "method"])
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIn(r.entity_type, ["function", "method"])


class TestDeterministicTieBreaking(unittest.TestCase):
    """Test that equal hybrid scores break ties deterministically by entity_id ascending."""

    def test_tie_breaking(self):
        # Two entities with identical scores in both retrievers
        bm25_items = [_make_item("entity_z", 5.0), _make_item("entity_a", 5.0)]
        sem_items = [_make_item("entity_z", 0.5), _make_item("entity_a", 0.5)]

        retriever = HybridRetriever(
            bm25_retriever=MockFixedRetriever(bm25_items),
            semantic_retriever=MockFixedRetriever(sem_items),
        )

        results = retriever.retrieve("query", k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].score, results[1].score)
        # entity_a must precede entity_z alphabetically
        self.assertEqual(results[0].entity_id, "entity_a")
        self.assertEqual(results[1].entity_id, "entity_z")

    def test_repeated_calls_determinism(self):
        bm25_items = [_make_item(f"e_{i}", float(i)) for i in range(10)]
        sem_items = [_make_item(f"e_{i}", float(10 - i)) for i in range(10)]

        retriever = HybridRetriever(
            bm25_retriever=MockFixedRetriever(bm25_items),
            semantic_retriever=MockFixedRetriever(sem_items),
        )

        res1 = retriever.retrieve("test", k=5)
        res2 = retriever.retrieve("test", k=5)

        self.assertEqual([r.entity_id for r in res1], [r.entity_id for r in res2])
        self.assertEqual([r.score for r in res1], [r.score for r in res2])


class TestRetrievedItemProvenance(unittest.TestCase):
    """Test that provenance and chunk attributes are fully preserved."""

    def test_provenance_structure(self):
        bm25_items = [_make_item("func:math.py:add", 10.0, "bm25")]
        sem_items = [_make_item("func:math.py:add", 0.9, "semantic_codebert")]

        retriever = HybridRetriever(
            bm25_retriever=MockFixedRetriever(bm25_items),
            semantic_retriever=MockFixedRetriever(sem_items),
            bm25_weight=0.6,
            semantic_weight=0.4,
            normalization="minmax",
        )

        results = retriever.retrieve("addition", k=1)
        self.assertEqual(len(results), 1)
        item = results[0]

        self.assertIsInstance(item, RetrievedItem)
        self.assertEqual(item.retrieval_method, "hybrid")
        self.assertEqual(item.entity_id, "func:math.py:add")

        prov = item.provenance
        self.assertEqual(prov["retrieval_method"], "hybrid")
        self.assertEqual(sorted(prov["sources"]), ["bm25", "semantic_codebert"])
        self.assertEqual(prov["bm25_weight"], 0.6)
        self.assertEqual(prov["semantic_weight"], 0.4)
        self.assertEqual(prov["normalization"], "minmax")
        self.assertIn("bm25_raw_score", prov)
        self.assertIn("bm25_normalized_score", prov)
        self.assertIn("semantic_raw_score", prov)
        self.assertIn("semantic_normalized_score", prov)
        self.assertIn("combined_score", prov)

        # Serialization roundtrip
        item_dict = item.to_dict()
        self.assertEqual(item_dict["retrieval_method"], "hybrid")
        self.assertEqual(item_dict["entity_id"], "func:math.py:add")


class TestHybridWithLiveRetrieversAndSampleRepo(unittest.TestCase):
    """End-to-end integration test combining live BM25 and Semantic retrievers."""

    @classmethod
    def setUpClass(cls):
        cls.chunker = CodeEntityChunker(repo_path=SAMPLE_REPO)
        cls.chunks = cls.chunker.chunk_analysis_file(SAMPLE_ANALYSIS)
        # Use DummyDeterministicEmbedder for deterministic, fast embedding
        cls.embedder = DummyDeterministicEmbedder()

        cls.bm25_retriever = BM25Retriever.from_chunks(cls.chunks)
        cls.semantic_retriever = SemanticRetriever.from_chunks(
            chunks=cls.chunks,
            embedder=cls.embedder,
            repo_path=SAMPLE_REPO,
        )
        cls.hybrid_retriever = HybridRetriever.from_retrievers(
            bm25_retriever=cls.bm25_retriever,
            semantic_retriever=cls.semantic_retriever,
            bm25_weight=0.5,
            semantic_weight=0.5,
        )

    def test_hybrid_retrieve_from_chunks(self):
        self.assertGreater(len(self.hybrid_retriever), 0)

        results = self.hybrid_retriever.retrieve("UserService create_user", k=3)
        self.assertEqual(len(results), 3)

        for item in results:
            self.assertIsInstance(item, RetrievedItem)
            self.assertEqual(item.retrieval_method, "hybrid")
            self.assertTrue(item.entity_id)
            self.assertIsInstance(item.score, float)
            self.assertGreater(len(item.provenance["sources"]), 0)

    def test_from_chunks_factory(self):
        retriever = HybridRetriever.from_chunks(
            self.chunks,
            bm25_weight=0.6,
            semantic_weight=0.4,
            repo_path=SAMPLE_REPO,
        )
        self.assertEqual(len(retriever), len(self.chunks))

    def test_from_analysis_file_factory(self):
        retriever = HybridRetriever.from_analysis_file(
            analysis_file_path=SAMPLE_ANALYSIS,
            repo_path=SAMPLE_REPO,
        )
        self.assertEqual(len(retriever), 16)

    def test_from_knowledge_graph_factory(self):
        kg = build_from_analysis(SAMPLE_ANALYSIS)
        retriever = HybridRetriever.from_knowledge_graph(
            graph=kg,
            repo_path=SAMPLE_REPO,
        )
        self.assertEqual(len(retriever), 16)


if __name__ == "__main__":
    unittest.main()
