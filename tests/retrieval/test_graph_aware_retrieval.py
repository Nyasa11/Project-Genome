"""Focused tests for Phase 12: B5 Graph-Aware Retrieval.

Tests cover:
- BaseRetriever compatibility and export
- Initial semantic seeds preservation
- Graph expansion around semantic seeds
- Graph depth behavior (depth 0, depth 1, depth 2)
- Relationship type filtering
- Graph evidence scoring with distance decay
- Linear combination: semantic_weight * norm_sem + graph_weight * graph_evidence
- Default 0.7 / 0.3 and custom weightings
- Deterministic tie-breaking (score descending, semantic score descending, entity_id ascending)
- Duplicate prevention (entities appear at most once)
- Entity-type filtering
- Edge cases: empty query, k <= 0, empty semantic results, empty KG, missing KG entity, candidate_k < k
- Complete provenance recording
- Invalid configuration rejection
- End-to-end integration with sample fixtures
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path
import numpy as np

from src.retrieval.base import BaseRetriever, RetrievedItem
from src.retrieval.chunker import CodeChunk, CodeEntityChunker
from src.retrieval.graph_aware import (
    DEFAULT_GRAPH_RELATIONSHIP_TYPES,
    GraphAwareRetriever,
)
from src.retrieval.semantic import SemanticRetriever
from src.knowledge_graph.builder import build_from_analysis
from src.knowledge_graph.graph import KnowledgeGraph
from src.knowledge_graph.schema import NodeType, RelationshipType

SAMPLE_ANALYSIS = Path("sample_analysis.json")
SAMPLE_REPO = Path("sample_repo")


def _make_item(
    entity_id: str,
    score: float,
    entity_type: str = "function",
    file_path: str = "sample.py",
) -> RetrievedItem:
    return RetrievedItem(
        entity_id=entity_id,
        file_path=file_path,
        source_code=f"def {entity_id}(): pass",
        score=score,
        retrieval_method="semantic_codebert",
        entity_type=entity_type,
        name=entity_id.split(":")[-1],
        qualified_name=entity_id,
        provenance={"similarity_score": score},
    )


class MockSemanticRetriever(BaseRetriever):
    """Mock semantic retriever returning predefined RetrievedItems for testing."""

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
            allowed = (
                {entity_type_filter.lower()}
                if isinstance(entity_type_filter, str)
                else {t.lower() for t in entity_type_filter}
            )
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


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestGraphAwareRetrieverInitAndValidation(unittest.TestCase):
    """Test initialization, validation, and BaseRetriever compatibility."""

    def test_is_base_retriever(self):
        retriever = GraphAwareRetriever()
        self.assertIsInstance(retriever, BaseRetriever)

    def test_default_parameters(self):
        retriever = GraphAwareRetriever()
        self.assertAlmostEqual(retriever.semantic_weight, 0.7)
        self.assertAlmostEqual(retriever.graph_weight, 0.3)
        self.assertEqual(retriever.graph_depth, 1)
        self.assertAlmostEqual(retriever.decay_factor, 0.5)
        self.assertEqual(retriever.normalization, "minmax")
        self.assertEqual(retriever.direction, "both")

    def test_custom_parameters(self):
        retriever = GraphAwareRetriever(
            semantic_weight=0.8,
            graph_weight=0.2,
            graph_depth=2,
            decay_factor=0.6,
            normalization="max",
            direction="out",
        )
        self.assertAlmostEqual(retriever.semantic_weight, 0.8)
        self.assertAlmostEqual(retriever.graph_weight, 0.2)
        self.assertEqual(retriever.graph_depth, 2)
        self.assertAlmostEqual(retriever.decay_factor, 0.6)
        self.assertEqual(retriever.normalization, "max")
        self.assertEqual(retriever.direction, "out")

    def test_invalid_weights_rejected(self):
        with self.assertRaises(ValueError):
            GraphAwareRetriever(semantic_weight=-0.1, graph_weight=0.5)
        with self.assertRaises(ValueError):
            GraphAwareRetriever(semantic_weight=0.5, graph_weight=-0.2)
        with self.assertRaises(ValueError):
            GraphAwareRetriever(semantic_weight=0.0, graph_weight=0.0)
        with self.assertRaises(ValueError):
            GraphAwareRetriever(semantic_weight=float("nan"), graph_weight=0.5)
        with self.assertRaises(ValueError):
            GraphAwareRetriever(semantic_weight=0.5, graph_weight=float("inf"))
        with self.assertRaises(ValueError):
            GraphAwareRetriever(semantic_weight="invalid", graph_weight=0.5)

    def test_invalid_graph_depth_rejected(self):
        with self.assertRaises(ValueError):
            GraphAwareRetriever(graph_depth=-1)
        with self.assertRaises(ValueError):
            GraphAwareRetriever(graph_depth="one")

    def test_invalid_decay_factor_rejected(self):
        with self.assertRaises(ValueError):
            GraphAwareRetriever(decay_factor=0.0)
        with self.assertRaises(ValueError):
            GraphAwareRetriever(decay_factor=-0.5)
        with self.assertRaises(ValueError):
            GraphAwareRetriever(decay_factor=1.5)

    def test_invalid_normalization_rejected(self):
        with self.assertRaises(ValueError):
            GraphAwareRetriever(normalization="zscore")

    def test_invalid_direction_rejected(self):
        with self.assertRaises(ValueError):
            GraphAwareRetriever(direction="sideways")

    def test_invalid_candidate_k_rejected(self):
        with self.assertRaises(ValueError):
            GraphAwareRetriever(candidate_k=0)
        with self.assertRaises(ValueError):
            GraphAwareRetriever(candidate_k=-5)

    def test_invalid_relationship_type_rejected(self):
        with self.assertRaises(ValueError):
            GraphAwareRetriever(allowed_relationship_types=["INVALID_TYPE"])


class TestGraphExpansionAndScoring(unittest.TestCase):
    """Test graph expansion and graph evidence scoring."""

    def setUp(self):
        # Create a toy KnowledgeGraph:
        # func:seed_a (score 1.0) --CALLS--> func:target_b
        # func:seed_c (score 0.5) --CALLS--> func:target_d
        # func:target_b --CALLS--> func:target_e
        self.kg = KnowledgeGraph()
        self.kg.add_node("func:seed_a", NodeType.FUNCTION, name="seed_a", path="a.py")
        self.kg.add_node("func:target_b", NodeType.FUNCTION, name="target_b", path="b.py")
        self.kg.add_node("func:seed_c", NodeType.FUNCTION, name="seed_c", path="c.py")
        self.kg.add_node("func:target_d", NodeType.FUNCTION, name="target_d", path="d.py")
        self.kg.add_node("func:target_e", NodeType.FUNCTION, name="target_e", path="e.py")

        self.kg.add_relationship("func:seed_a", "func:target_b", RelationshipType.CALLS)
        self.kg.add_relationship("func:seed_c", "func:target_d", RelationshipType.CALLS)
        self.kg.add_relationship("func:target_b", "func:target_e", RelationshipType.CALLS)

        self.mock_sem = MockSemanticRetriever([
            _make_item("func:seed_a", 10.0),
            _make_item("func:seed_c", 5.0),
        ])

    def test_semantic_seeds_returned(self):
        retriever = GraphAwareRetriever(
            semantic_retriever=self.mock_sem,
            graph=self.kg,
            graph_depth=0,
        )
        results = retriever.retrieve("query", k=10)
        result_ids = [r.entity_id for r in results]
        self.assertIn("func:seed_a", result_ids)
        self.assertIn("func:seed_c", result_ids)
        self.assertEqual(len(results), 2)

    def test_graph_neighbors_added_at_depth_1(self):
        retriever = GraphAwareRetriever(
            semantic_retriever=self.mock_sem,
            graph=self.kg,
            graph_depth=1,
            decay_factor=0.5,
        )
        results = retriever.retrieve("query", k=10)
        result_ids = [r.entity_id for r in results]

        # seed_a, seed_c should be present
        self.assertIn("func:seed_a", result_ids)
        self.assertIn("func:seed_c", result_ids)
        # target_b and target_d are distance 1 neighbors, so they should be added
        self.assertIn("func:target_b", result_ids)
        self.assertIn("func:target_d", result_ids)
        # target_e is distance 2, should NOT be added at depth 1
        self.assertNotIn("func:target_e", result_ids)

    def test_graph_neighbors_added_at_depth_2(self):
        retriever = GraphAwareRetriever(
            semantic_retriever=self.mock_sem,
            graph=self.kg,
            graph_depth=2,
            decay_factor=0.5,
        )
        results = retriever.retrieve("query", k=10)
        result_ids = [r.entity_id for r in results]
        self.assertIn("func:target_e", result_ids)

    def test_graph_evidence_rewards_connection_to_stronger_seed(self):
        # seed_a has norm 1.0; seed_c has norm 0.0 (minmax normalization of 10.0 and 5.0)
        retriever = GraphAwareRetriever(
            semantic_retriever=self.mock_sem,
            graph=self.kg,
            graph_depth=1,
        )
        results = retriever.retrieve("query", k=10)
        res_map = {r.entity_id: r for r in results}

        # target_b (connected to seed_a) should receive stronger graph evidence than target_d (connected to seed_c)
        ev_b = res_map["func:target_b"].provenance["graph_evidence_score"]
        ev_d = res_map["func:target_d"].provenance["graph_evidence_score"]
        self.assertGreater(ev_b, ev_d)

    def test_weighted_score_calculation(self):
        # sem_weight = 0.7, graph_weight = 0.3
        retriever = GraphAwareRetriever(
            semantic_retriever=self.mock_sem,
            graph=self.kg,
            graph_depth=1,
            semantic_weight=0.7,
            graph_weight=0.3,
        )
        results = retriever.retrieve("query", k=10)
        res_map = {r.entity_id: r for r in results}

        item_a = res_map["func:seed_a"]
        prov_a = item_a.provenance
        # seed_a has norm_sem = 1.0, graph_evidence = 1.0
        expected_score = round(0.7 * 1.0 + 0.3 * 1.0, 6)
        self.assertAlmostEqual(item_a.score, expected_score)
        self.assertAlmostEqual(prov_a["semantic_contribution"], 0.7)
        self.assertAlmostEqual(prov_a["graph_contribution"], 0.3)

    def test_custom_weights(self):
        retriever = GraphAwareRetriever(
            semantic_retriever=self.mock_sem,
            graph=self.kg,
            graph_depth=1,
            semantic_weight=0.5,
            graph_weight=0.5,
        )
        results = retriever.retrieve("query", k=10)
        res_map = {r.entity_id: r for r in results}

        item_b = res_map["func:target_b"]
        # target_b is non-seed (norm_sem = 0.0), connected to seed_a (evidence = 1.0)
        # expected = 0.5 * 0.0 + 0.5 * 1.0 = 0.50
        self.assertAlmostEqual(item_b.score, 0.5)


class TestRelationshipFiltering(unittest.TestCase):
    """Test allowed relationship types filtering."""

    def setUp(self):
        self.kg = KnowledgeGraph()
        self.kg.add_node("func:seed", NodeType.FUNCTION, name="seed", path="s.py")
        self.kg.add_node("func:calls_target", NodeType.FUNCTION, name="calls_target", path="c.py")
        self.kg.add_node("func:imports_target", NodeType.FUNCTION, name="imports_target", path="i.py")

        self.kg.add_relationship("func:seed", "func:calls_target", RelationshipType.CALLS)
        self.kg.add_relationship("func:seed", "func:imports_target", RelationshipType.IMPORTS)

        self.mock_sem = MockSemanticRetriever([_make_item("func:seed", 1.0)])

    def test_filter_only_calls(self):
        retriever = GraphAwareRetriever(
            semantic_retriever=self.mock_sem,
            graph=self.kg,
            graph_depth=1,
            allowed_relationship_types=[RelationshipType.CALLS],
        )
        results = retriever.retrieve("query", k=10)
        result_ids = [r.entity_id for r in results]
        self.assertIn("func:calls_target", result_ids)
        self.assertNotIn("func:imports_target", result_ids)

    def test_filter_only_imports(self):
        retriever = GraphAwareRetriever(
            semantic_retriever=self.mock_sem,
            graph=self.kg,
            graph_depth=1,
            allowed_relationship_types=["IMPORTS"],
        )
        results = retriever.retrieve("query", k=10)
        result_ids = [r.entity_id for r in results]
        self.assertIn("func:imports_target", result_ids)
        self.assertNotIn("func:calls_target", result_ids)


class TestDeterministicRankingAndDeduplication(unittest.TestCase):
    """Test tie breaking and single presence."""

    def setUp(self):
        self.kg = KnowledgeGraph()
        self.kg.add_node("func:z", NodeType.FUNCTION, name="z", path="z.py")
        self.kg.add_node("func:a", NodeType.FUNCTION, name="a", path="a.py")
        self.kg.add_node("func:m", NodeType.FUNCTION, name="m", path="m.py")

        # Two seeds with identical score
        self.mock_sem = MockSemanticRetriever([
            _make_item("func:z", 5.0),
            _make_item("func:a", 5.0),
        ])

    def test_tie_breaking_by_entity_id(self):
        retriever = GraphAwareRetriever(
            semantic_retriever=self.mock_sem,
            graph=self.kg,
            graph_depth=0,
        )
        results = retriever.retrieve("query", k=10)
        self.assertEqual(len(results), 2)
        # func:a must precede func:z when scores are identical
        self.assertEqual(results[0].entity_id, "func:a")
        self.assertEqual(results[1].entity_id, "func:z")

    def test_repeated_calls_are_deterministic(self):
        retriever = GraphAwareRetriever(
            semantic_retriever=self.mock_sem,
            graph=self.kg,
            graph_depth=1,
        )
        res1 = [(r.entity_id, r.score) for r in retriever.retrieve("query", k=5)]
        res2 = [(r.entity_id, r.score) for r in retriever.retrieve("query", k=5)]
        self.assertEqual(res1, res2)

    def test_entity_appears_at_most_once(self):
        # Create cycle: seed_a -> seed_b -> seed_a
        self.kg.add_relationship("func:z", "func:a", RelationshipType.CALLS)
        self.kg.add_relationship("func:a", "func:z", RelationshipType.CALLS)

        retriever = GraphAwareRetriever(
            semantic_retriever=self.mock_sem,
            graph=self.kg,
            graph_depth=2,
        )
        results = retriever.retrieve("query", k=10)
        ids = [r.entity_id for r in results]
        self.assertEqual(len(ids), len(set(ids)))


class TestTopKAndFiltering(unittest.TestCase):
    """Test k limit and entity type filtering."""

    def setUp(self):
        self.kg = KnowledgeGraph()
        self.kg.add_node("func:f1", NodeType.FUNCTION, name="f1", path="f.py")
        self.kg.add_node("class:c1", NodeType.CLASS, name="c1", path="c.py")
        self.kg.add_node("method:m1", NodeType.METHOD, name="m1", path="m.py")

        self.mock_sem = MockSemanticRetriever([
            _make_item("func:f1", 10.0, entity_type="function"),
            _make_item("class:c1", 8.0, entity_type="class"),
            _make_item("method:m1", 6.0, entity_type="method"),
        ])
        self.retriever = GraphAwareRetriever(
            semantic_retriever=self.mock_sem,
            graph=self.kg,
        )

    def test_top_k(self):
        self.assertEqual(len(self.retriever.retrieve("q", k=1)), 1)
        self.assertEqual(len(self.retriever.retrieve("q", k=2)), 2)
        self.assertEqual(len(self.retriever.retrieve("q", k=5)), 3)

    def test_entity_type_filter_single(self):
        results = self.retriever.retrieve("q", k=5, entity_type_filter="class")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].entity_type, "class")
        self.assertEqual(results[0].entity_id, "class:c1")

    def test_entity_type_filter_multiple(self):
        results = self.retriever.retrieve("q", k=5, entity_type_filter=["function", "method"])
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIn(r.entity_type, ["function", "method"])


class TestEdgeCases(unittest.TestCase):
    """Test boundary conditions and edge cases."""

    def test_empty_query(self):
        retriever = GraphAwareRetriever()
        self.assertEqual(retriever.retrieve(""), [])
        self.assertEqual(retriever.retrieve("   "), [])

    def test_non_positive_k(self):
        retriever = GraphAwareRetriever()
        self.assertEqual(retriever.retrieve("query", k=0), [])
        self.assertEqual(retriever.retrieve("query", k=-2), [])

    def test_empty_semantic_results(self):
        mock_sem = MockSemanticRetriever([])
        retriever = GraphAwareRetriever(semantic_retriever=mock_sem)
        self.assertEqual(retriever.retrieve("query", k=5), [])

    def test_empty_kg(self):
        mock_sem = MockSemanticRetriever([_make_item("f1", 1.0)])
        kg = KnowledgeGraph()
        retriever = GraphAwareRetriever(semantic_retriever=mock_sem, graph=kg)
        results = retriever.retrieve("query", k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].entity_id, "f1")

    def test_semantic_entity_not_in_kg(self):
        mock_sem = MockSemanticRetriever([_make_item("func:missing", 1.0)])
        kg = KnowledgeGraph()
        kg.add_node("func:other", NodeType.FUNCTION, name="other", path="o.py")
        retriever = GraphAwareRetriever(semantic_retriever=mock_sem, graph=kg)
        results = retriever.retrieve("query", k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].entity_id, "func:missing")
        self.assertTrue(results[0].provenance["is_semantic_seed"])

    def test_candidate_k_smaller_than_k(self):
        mock_sem = MockSemanticRetriever([
            _make_item("f1", 3.0),
            _make_item("f2", 2.0),
            _make_item("f3", 1.0),
        ])
        retriever = GraphAwareRetriever(semantic_retriever=mock_sem, candidate_k=1)
        # Should clamp candidate fetch so at least k items can be returned if available
        results = retriever.retrieve("query", k=3)
        self.assertEqual(len(results), 3)


class TestProvenanceCompleteness(unittest.TestCase):
    """Verify all required provenance fields are populated."""

    def setUp(self):
        kg = KnowledgeGraph()
        kg.add_node("func:a", NodeType.FUNCTION, name="a", path="a.py")
        kg.add_node("func:b", NodeType.FUNCTION, name="b", path="b.py")
        kg.add_relationship("func:a", "func:b", RelationshipType.CALLS)

        mock_sem = MockSemanticRetriever([_make_item("func:a", 10.0)])
        self.retriever = GraphAwareRetriever(
            semantic_retriever=mock_sem,
            graph=kg,
            graph_depth=1,
            semantic_weight=0.7,
            graph_weight=0.3,
        )

    def test_required_provenance_keys(self):
        results = self.retriever.retrieve("query", k=5)
        self.assertEqual(len(results), 2)

        required_keys = [
            "retrieval_method",
            "semantic_raw_score",
            "semantic_normalized_score",
            "graph_evidence_score",
            "graph_contribution",
            "semantic_contribution",
            "final_graph_aware_score",
            "is_semantic_seed",
            "seed_entities",
            "graph_distance",
            "relationship_types",
            "graph_depth",
            "semantic_weight",
            "graph_weight",
        ]

        for item in results:
            self.assertEqual(item.retrieval_method, "graph_aware")
            for key in required_keys:
                self.assertIn(key, item.provenance, f"Missing provenance key: {key}")


class TestEndToEndWithSampleRepoAndKG(unittest.TestCase):
    """End-to-end integration tests using sample fixtures."""

    @classmethod
    def setUpClass(cls):
        cls.chunker = CodeEntityChunker(repo_path=SAMPLE_REPO)
        cls.chunks = cls.chunker.chunk_analysis_file(SAMPLE_ANALYSIS)
        cls.embedder = DummyDeterministicEmbedder()
        cls.kg = build_from_analysis(SAMPLE_ANALYSIS)

        cls.sem_retriever = SemanticRetriever.from_chunks(
            chunks=cls.chunks,
            embedder=cls.embedder,
            repo_path=SAMPLE_REPO,
        )
        cls.retriever = GraphAwareRetriever.from_retriever(
            semantic_retriever=cls.sem_retriever,
            graph=cls.kg,
            semantic_weight=0.7,
            graph_weight=0.3,
            graph_depth=1,
        )

    def test_retrieve_end_to_end(self):
        results = self.retriever.retrieve("UserService create_user", k=3)
        self.assertEqual(len(results), 3)

        for item in results:
            self.assertIsInstance(item, RetrievedItem)
            self.assertEqual(item.retrieval_method, "graph_aware")
            self.assertTrue(item.entity_id)
            self.assertIsInstance(item.score, float)
            self.assertIn("graph_evidence_score", item.provenance)

    def test_from_knowledge_graph_factory(self):
        retriever = GraphAwareRetriever.from_knowledge_graph(
            graph=self.kg,
            repo_path=SAMPLE_REPO,
            embedder=self.embedder,
        )
        self.assertGreater(len(retriever), 0)

    def test_from_analysis_file_factory(self):
        retriever = GraphAwareRetriever.from_analysis_file(
            analysis_file_path=SAMPLE_ANALYSIS,
            repo_path=SAMPLE_REPO,
            embedder=self.embedder,
        )
        self.assertGreater(len(retriever), 0)

    def test_from_chunks_factory(self):
        retriever = GraphAwareRetriever.from_chunks(
            chunks=self.chunks,
            graph=self.kg,
            repo_path=SAMPLE_REPO,
            embedder=self.embedder,
        )
        self.assertGreater(len(retriever), 0)


if __name__ == "__main__":
    unittest.main()
