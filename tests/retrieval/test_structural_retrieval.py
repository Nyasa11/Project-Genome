"""Tests for Phase 7 (Common Retrieval Interface) and Phase 10 (Structural Retrieval).

Tests are deterministic: no model downloads, no network, no GPU required.
All tests use the existing sample_analysis.json and sample_repo fixtures.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from src.retrieval.base import BaseRetriever, DummyRetriever, RetrievedItem
from src.retrieval.chunker import CodeChunk
from src.retrieval.structural import STRUCTURAL_RELATIONSHIP_TYPES, StructuralRetriever
from src.knowledge_graph.builder import build_from_analysis
from src.knowledge_graph.schema import RelationshipType

SAMPLE_ANALYSIS = Path("sample_analysis.json")
SAMPLE_REPO = Path("sample_repo")


# ---------------------------------------------------------------------------
# Phase 7 — Common Retrieval Interface
# ---------------------------------------------------------------------------


class TestRetrievedItemModel(unittest.TestCase):
    """Verify the normalized output model (Phase 7.2)."""

    def test_required_fields_present(self):
        item = RetrievedItem(
            entity_id="func:a.py:foo",
            file_path="a.py",
            source_code="def foo(): pass",
            score=0.9,
            retrieval_method="structural",
            provenance={"starting_entity": "func:a.py:foo", "relationship_path": []},
        )
        self.assertEqual(item.entity_id, "func:a.py:foo")
        self.assertEqual(item.file_path, "a.py")
        self.assertEqual(item.source_code, "def foo(): pass")
        self.assertAlmostEqual(item.score, 0.9)
        self.assertEqual(item.retrieval_method, "structural")
        self.assertIn("starting_entity", item.provenance)

    def test_from_chunk_factory(self):
        chunk = CodeChunk(
            repository_id="repo",
            entity_id="class:b.py:Bar",
            entity_type="class",
            name="Bar",
            qualified_name="Bar",
            file_path="b.py",
            source_code="class Bar: pass",
            text_representation="Class Bar\nFile: b.py",
        )
        prov = {"starting_entity": "class:b.py:Bar", "relationship_path": []}
        item = RetrievedItem.from_chunk(chunk, score=0.75, retrieval_method="structural", provenance=prov)
        self.assertEqual(item.entity_id, chunk.entity_id)
        self.assertEqual(item.file_path, chunk.file_path)
        self.assertEqual(item.source_code, chunk.source_code)
        self.assertAlmostEqual(item.score, 0.75)
        self.assertEqual(item.retrieval_method, "structural")
        self.assertEqual(item.entity_type, chunk.entity_type)
        self.assertEqual(item.name, chunk.name)

    def test_serialization_roundtrip(self):
        item = RetrievedItem(
            entity_id="method:svc.py:S.m",
            file_path="svc.py",
            score=0.5,
            retrieval_method="bm25",
            provenance={},
        )
        d = item.to_dict()
        for key in ("entity_id", "file_path", "source_code", "score", "retrieval_method", "provenance"):
            self.assertIn(key, d)


class TestDummyRetriever(unittest.TestCase):
    """Verify Phase 7 exit condition: DummyRetriever passes through the pipeline."""

    def test_dummy_retriever_is_base_retriever(self):
        self.assertIsInstance(DummyRetriever(), BaseRetriever)

    def test_retrieve_returns_fixed_items(self):
        items = [
            RetrievedItem(
                entity_id=f"func:a.py:f{i}",
                file_path="a.py",
                score=float(i),
                retrieval_method="dummy",
                provenance={},
            )
            for i in range(5)
        ]
        retriever = DummyRetriever(fixed_items=items)
        result = retriever.retrieve("any question", k=3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].entity_id, "func:a.py:f0")

    def test_retrieve_empty(self):
        retriever = DummyRetriever()
        result = retriever.retrieve("anything", k=10)
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Phase 10 — Structural Retrieval: Initialisation
# ---------------------------------------------------------------------------


class TestStructuralRetrieverInit(unittest.TestCase):

    def test_default_relationship_types_match_phase_10_spec(self):
        expected = {
            RelationshipType.CONTAINS.value,
            RelationshipType.IMPORTS.value,
            RelationshipType.INHERITS.value,
            RelationshipType.CALLS.value,
            RelationshipType.DEPENDS_ON.value,
        }
        retriever = StructuralRetriever()
        self.assertEqual(retriever.allowed_types, expected)

    def test_structural_relationship_types_constant(self):
        values = {rt.value for rt in STRUCTURAL_RELATIONSHIP_TYPES}
        for expected_type in ("CONTAINS", "IMPORTS", "INHERITS", "CALLS", "DEPENDS_ON"):
            self.assertIn(expected_type, values)

    def test_is_base_retriever(self):
        self.assertIsInstance(StructuralRetriever(), BaseRetriever)

    def test_set_graph_builds_indexes(self):
        kg = build_from_analysis(SAMPLE_ANALYSIS)
        retriever = StructuralRetriever()
        retriever.set_graph(kg)
        self.assertGreater(len(retriever._name_to_ids), 0)
        self.assertGreater(len(retriever._lower_name_to_ids), 0)
        self.assertGreater(len(retriever._chunks_by_id), 0)

    def test_graph_via_constructor(self):
        kg = build_from_analysis(SAMPLE_ANALYSIS)
        retriever = StructuralRetriever(graph=kg, repo_path=SAMPLE_REPO)
        self.assertIs(retriever.graph, kg)
        self.assertGreater(len(retriever._chunks_by_id), 0)

    def test_custom_max_depth(self):
        retriever = StructuralRetriever(max_depth=1)
        self.assertEqual(retriever.max_depth, 1)

    def test_custom_relationship_types(self):
        retriever = StructuralRetriever(
            allowed_relationship_types=[RelationshipType.CALLS]
        )
        self.assertEqual(retriever.allowed_types, {"CALLS"})


# ---------------------------------------------------------------------------
# Phase 10.1 — Candidate Identification
# ---------------------------------------------------------------------------


class TestCandidateIdentification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        kg = build_from_analysis(SAMPLE_ANALYSIS)
        cls.retriever = StructuralRetriever(graph=kg, repo_path=SAMPLE_REPO)

    def test_exact_class_name_matches(self):
        candidates = self.retriever.identify_candidates("What does UserService do?")
        ids = [c[0] for c in candidates]
        self.assertIn("class:services/user_service.py:UserService", ids)

    def test_exact_method_name_matches(self):
        candidates = self.retriever.identify_candidates("How does create_user work?")
        ids = [c[0] for c in candidates]
        self.assertIn("method:services/user_service.py:UserService.create_user", ids)

    def test_file_name_matches(self):
        candidates = self.retriever.identify_candidates(
            "What is defined in user_service.py?"
        )
        ids = [c[0] for c in candidates]
        self.assertIn("file:services/user_service.py", ids)

    def test_empty_question_returns_no_candidates(self):
        candidates = self.retriever.identify_candidates("")
        self.assertEqual(candidates, [])

    def test_irrelevant_question_returns_no_candidates(self):
        candidates = self.retriever.identify_candidates("the quick brown fox")
        self.assertEqual(candidates, [])

    def test_candidates_sorted_by_score_descending(self):
        candidates = self.retriever.identify_candidates(
            "How does UserService.get_user work?"
        )
        if len(candidates) >= 2:
            for i in range(len(candidates) - 1):
                self.assertGreaterEqual(candidates[i][1], candidates[i + 1][1])

    def test_seed_score_between_zero_and_one(self):
        candidates = self.retriever.identify_candidates("UserService creates User")
        for _, score, _ in candidates:
            self.assertGreater(score, 0.0)
            self.assertLessEqual(score, 1.0)


# ---------------------------------------------------------------------------
# Phase 10.2 – 10.3 — Graph Traversal
# ---------------------------------------------------------------------------


class TestGraphTraversal(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        kg = build_from_analysis(SAMPLE_ANALYSIS)
        cls.kg = kg
        cls.retriever = StructuralRetriever(graph=kg, repo_path=SAMPLE_REPO)

    def test_seed_entity_has_distance_zero(self):
        candidates = [
            ("class:services/user_service.py:UserService", 0.95, "UserService")
        ]
        records = self.retriever.traverse_structural_context(candidates)
        seed_record = next(
            (r for r in records if r["entity_id"] == "class:services/user_service.py:UserService"),
            None,
        )
        self.assertIsNotNone(seed_record)
        self.assertEqual(seed_record["distance"], 0)

    def test_traversal_expands_via_contains_relationship(self):
        candidates = [
            ("class:services/user_service.py:UserService", 0.95, "UserService")
        ]
        records = self.retriever.traverse_structural_context(candidates)
        ids = {r["entity_id"] for r in records}
        self.assertIn("method:services/user_service.py:UserService.create_user", ids)

    def test_traversal_expands_via_calls_relationship(self):
        candidates = [
            ("method:services/user_service.py:UserService.create_user", 0.95, "create_user")
        ]
        records = self.retriever.traverse_structural_context(candidates)
        ids = {r["entity_id"] for r in records}
        self.assertIn("class:models/user.py:User", ids)

    def test_traversal_via_inherits_relationship(self):
        candidates = [("class:models/user.py:User", 0.95, "User")]
        records = self.retriever.traverse_structural_context(candidates, max_depth=1)
        ids = {r["entity_id"] for r in records}
        self.assertIn("class:models/base.py:BaseModel", ids)

    def test_provenance_path_recorded_for_non_seeds(self):
        candidates = [
            ("class:services/user_service.py:UserService", 0.95, "UserService")
        ]
        records = self.retriever.traverse_structural_context(candidates)
        non_seed = [r for r in records if r["distance"] > 0]
        for r in non_seed:
            self.assertGreater(len(r["relationship_path"]), 0)

    def test_starting_entity_recorded_in_all_records(self):
        candidates = [("class:models/user.py:User", 0.95, "User")]
        records = self.retriever.traverse_structural_context(candidates)
        for r in records:
            self.assertIsNotNone(r["starting_entity"])
            self.assertIsNotNone(r["traversal_path"])

    def test_synthetic_nodes_excluded(self):
        candidates = [
            ("class:services/user_service.py:UserService", 0.95, "UserService")
        ]
        records = self.retriever.traverse_structural_context(candidates)
        for r in records:
            node_data = self.kg.get_node(r["entity_id"]) or {}
            self.assertFalse(node_data.get("is_synthetic", False), r["entity_id"])

    def test_max_depth_zero_returns_only_seed(self):
        candidates = [
            ("class:services/user_service.py:UserService", 0.95, "UserService")
        ]
        records = self.retriever.traverse_structural_context(candidates, max_depth=0)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["entity_id"], "class:services/user_service.py:UserService")

    def test_score_decays_with_distance(self):
        candidates = [
            ("class:services/user_service.py:UserService", 1.0, "UserService")
        ]
        records = self.retriever.traverse_structural_context(candidates, max_depth=2)
        seed_score = next(r["score"] for r in records if r["distance"] == 0)
        hop1 = [r for r in records if r["distance"] == 1]
        for r in hop1:
            self.assertLess(r["score"], seed_score)

    def test_empty_candidates_returns_empty(self):
        records = self.retriever.traverse_structural_context([])
        self.assertEqual(records, [])


# ---------------------------------------------------------------------------
# Phase 10.4 – 10.5 — retrieve() contract
# ---------------------------------------------------------------------------


class TestStructuralRetrieverRetrieve(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        kg = build_from_analysis(SAMPLE_ANALYSIS)
        cls.retriever = StructuralRetriever(graph=kg, repo_path=SAMPLE_REPO)

    def test_returns_retrieved_items(self):
        results = self.retriever.retrieve("What does UserService do?", k=5)
        self.assertIsInstance(results, list)
        for item in results:
            self.assertIsInstance(item, RetrievedItem)

    def test_retrieval_method_is_structural(self):
        results = self.retriever.retrieve("UserService", k=5)
        for item in results:
            self.assertEqual(item.retrieval_method, "structural")

    def test_top_k_bound_respected(self):
        for k in (1, 3, 5):
            results = self.retriever.retrieve("UserService User BaseModel", k=k)
            self.assertLessEqual(len(results), k)

    def test_results_sorted_descending_by_score(self):
        results = self.retriever.retrieve("UserService creates User", k=10)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i].score, results[i + 1].score)

    def test_entity_id_populated(self):
        results = self.retriever.retrieve("UserService", k=5)
        for item in results:
            self.assertTrue(item.entity_id)

    def test_file_path_populated(self):
        results = self.retriever.retrieve("UserService", k=5)
        for item in results:
            self.assertTrue(item.file_path)

    def test_provenance_populated(self):
        results = self.retriever.retrieve("UserService", k=5)
        for item in results:
            self.assertIn("starting_entity", item.provenance)
            self.assertIn("relationship_path", item.provenance)
            self.assertIn("target_entity", item.provenance)

    def test_no_model_used(self):
        import src.retrieval.structural as mod
        module_source = open(mod.__file__).read()
        self.assertNotIn("import torch", module_source)
        self.assertNotIn("import transformers", module_source)

    def test_determinism_repeated_calls(self):
        q = "How does UserService.create_user work?"
        res1 = self.retriever.retrieve(q, k=5)
        res2 = self.retriever.retrieve(q, k=5)
        self.assertEqual(
            [(r.entity_id, r.score) for r in res1],
            [(r.entity_id, r.score) for r in res2],
        )

    def test_graph_passed_via_retrieve(self):
        kg = build_from_analysis(SAMPLE_ANALYSIS)
        retriever = StructuralRetriever(repo_path=SAMPLE_REPO)
        results = retriever.retrieve("UserService", repository=kg, k=5)
        self.assertGreater(len(results), 0)

    def test_irrelevant_question_returns_empty(self):
        results = self.retriever.retrieve("the quick brown fox", k=10)
        self.assertEqual(results, [])

    def test_source_code_preserved(self):
        results = self.retriever.retrieve("UserService", k=5)
        entity_items = [r for r in results if r.entity_type in ("class", "function", "method")]
        self.assertGreater(len(entity_items), 0)
        for item in entity_items:
            self.assertTrue(item.source_code.strip(), f"{item.entity_id} has empty source_code")

    def test_inherits_relationship_retrieves_base_class(self):
        results = self.retriever.retrieve("User model inherits", k=10)
        ids = {r.entity_id for r in results}
        self.assertIn("class:models/base.py:BaseModel", ids)

    def test_calls_relationship_retrieves_called_entity(self):
        results = self.retriever.retrieve("create_user method", k=10)
        ids = {r.entity_id for r in results}
        self.assertIn("class:models/user.py:User", ids)

    def test_no_graph_raises_value_error(self):
        retriever = StructuralRetriever()
        with self.assertRaises(ValueError):
            retriever.retrieve("UserService", k=5)


# ---------------------------------------------------------------------------
# Identifier token extraction helper
# ---------------------------------------------------------------------------


class TestExtractIdentifierTokens(unittest.TestCase):

    def test_snake_case_split(self):
        tokens = StructuralRetriever._extract_identifier_tokens("create_user")
        self.assertIn("create", tokens)
        self.assertIn("user", tokens)

    def test_camel_case_split(self):
        tokens = StructuralRetriever._extract_identifier_tokens("UserService")
        self.assertIn("User", tokens)
        self.assertIn("Service", tokens)

    def test_empty_string_returns_empty(self):
        tokens = StructuralRetriever._extract_identifier_tokens("")
        self.assertEqual(tokens, set())

    def test_single_short_token_excluded(self):
        tokens = StructuralRetriever._extract_identifier_tokens("a")
        self.assertEqual(tokens, set())


if __name__ == "__main__":
    unittest.main()
