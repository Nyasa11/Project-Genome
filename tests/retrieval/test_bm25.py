"""Focused tests for Phase 8: BM25 Retrieval Engine.

Tests are deterministic: no model downloads, no network, no GPU required.
Uses the existing sample_analysis.json and sample_repo fixtures as well as
isolated synthetic CodeChunk fixtures.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.retrieval.base import BaseRetriever, RetrievedItem
from src.retrieval.bm25 import BM25Retriever, get_canonical_chunk_text, tokenize_code
from src.retrieval.chunker import CodeChunk, CodeEntityChunker
from src.knowledge_graph.builder import build_from_analysis

SAMPLE_ANALYSIS = Path("sample_analysis.json")
SAMPLE_REPO = Path("sample_repo")


class TestTokenizeCode(unittest.TestCase):
    """Test tokenization for code identifiers and natural language queries."""

    def test_camel_case_splitting(self):
        tokens = tokenize_code("UserService")
        self.assertIn("userservice", tokens)
        self.assertIn("user", tokens)
        self.assertIn("service", tokens)

    def test_snake_case_splitting(self):
        tokens = tokenize_code("create_new_user")
        self.assertEqual(tokens, ["create", "new", "user"])

    def test_mixed_code_identifiers(self):
        tokens = tokenize_code("def get_HTTPResponse(status_code: int):")
        self.assertIn("def", tokens)
        self.assertIn("get", tokens)
        self.assertIn("httpresponse", tokens)
        self.assertIn("http", tokens)
        self.assertIn("response", tokens)
        self.assertIn("status", tokens)
        self.assertIn("code", tokens)
        self.assertIn("int", tokens)

    def test_empty_and_special_characters(self):
        self.assertEqual(tokenize_code(""), [])
        self.assertEqual(tokenize_code("   \n\t  "), [])
        self.assertEqual(tokenize_code("???!!!***"), [])


class TestCanonicalChunkText(unittest.TestCase):
    """Test extraction of canonical searchable text from CodeChunk."""

    def test_prefers_text_representation_when_present(self):
        chunk = CodeChunk(
            repository_id="repo",
            entity_id="func:a.py:foo",
            entity_type="function",
            name="foo",
            file_path="a.py",
            text_representation="Entity Type: Function\nName: foo\nFile: a.py",
        )
        self.assertEqual(get_canonical_chunk_text(chunk), chunk.text_representation)

    def test_fallback_constructs_from_all_fields(self):
        chunk = CodeChunk(
            repository_id="repo",
            entity_id="method:a.py:A.bar",
            entity_type="method",
            name="bar",
            qualified_name="A.bar",
            file_path="a.py",
            signature="bar(self, x: int)",
            docstring="Bar docstring.",
            source_code="def bar(self, x: int):\n    return x",
            text_representation="",
        )
        text = get_canonical_chunk_text(chunk)
        self.assertIn("Name: bar", text)
        self.assertIn("Qualified Name: A.bar", text)
        self.assertIn("File: a.py", text)
        self.assertIn("Signature: bar(self, x: int)", text)
        self.assertIn("Docstring: Bar docstring.", text)
        self.assertIn("Source Code:\ndef bar(self, x: int):\n    return x", text)


class TestBM25RetrieverInitAndIndexing(unittest.TestCase):
    """Test BM25Retriever initialization, indexing, and persistence."""

    def setUp(self):
        self.chunks = [
            CodeChunk(
                repository_id="repo",
                entity_id="func:a.py:calculate_tax",
                entity_type="function",
                name="calculate_tax",
                file_path="a.py",
                source_code="def calculate_tax(amount): return amount * 0.2",
                text_representation="Name: calculate_tax\nFile: a.py\ncalculate_tax(amount)",
            ),
            CodeChunk(
                repository_id="repo",
                entity_id="class:b.py:DatabaseConnection",
                entity_type="class",
                name="DatabaseConnection",
                file_path="b.py",
                source_code="class DatabaseConnection:\n    def connect(self): pass",
                text_representation="Name: DatabaseConnection\nFile: b.py\nclass DatabaseConnection",
            ),
        ]

    def test_is_base_retriever(self):
        retriever = BM25Retriever()
        self.assertIsInstance(retriever, BaseRetriever)

    def test_empty_init_and_len(self):
        retriever = BM25Retriever()
        self.assertEqual(len(retriever), 0)
        self.assertEqual(retriever.chunks, [])

    def test_index_and_len(self):
        retriever = BM25Retriever(chunks=self.chunks)
        self.assertEqual(len(retriever), 2)
        self.assertGreater(retriever.avgdl, 0)
        self.assertIn("calculate", retriever.inverted_index)

    def test_add_chunks(self):
        retriever = BM25Retriever(chunks=[self.chunks[0]])
        self.assertEqual(len(retriever), 1)
        retriever.add_chunks([self.chunks[1]])
        self.assertEqual(len(retriever), 2)

    def test_from_chunks_factory(self):
        retriever = BM25Retriever.from_chunks(self.chunks)
        self.assertEqual(len(retriever), 2)

    def test_from_analysis_file_factory(self):
        retriever = BM25Retriever.from_analysis_file(
            analysis_file_path=SAMPLE_ANALYSIS,
            repo_path=SAMPLE_REPO,
        )
        self.assertEqual(len(retriever), 16)

    def test_from_knowledge_graph_factory(self):
        kg = build_from_analysis(SAMPLE_ANALYSIS)
        retriever = BM25Retriever.from_knowledge_graph(kg, repo_path=SAMPLE_REPO)
        self.assertEqual(len(retriever), 16)

    def test_save_and_load(self):
        retriever = BM25Retriever.from_chunks(self.chunks, k1=1.8, b=0.8)
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever.save(tmpdir)
            loaded = BM25Retriever.load(tmpdir)
            self.assertEqual(len(loaded), 2)
            self.assertAlmostEqual(loaded.k1, 1.8)
            self.assertAlmostEqual(loaded.b, 0.8)
            # Verify retrieval works on loaded index
            results = loaded.retrieve("calculate tax", k=5)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].entity_id, "func:a.py:calculate_tax")


class TestBasicKeywordRetrieval(unittest.TestCase):
    """Test retrieval precision and keyword matching."""

    @classmethod
    def setUpClass(cls):
        chunker = CodeEntityChunker(repo_path=SAMPLE_REPO)
        cls.chunks = chunker.chunk_analysis_file(SAMPLE_ANALYSIS)
        cls.retriever = BM25Retriever.from_chunks(cls.chunks)

    def test_exact_class_name_retrieval(self):
        results = self.retriever.retrieve("UserService", k=5)
        self.assertTrue(len(results) > 0)
        retrieved_ids = [r.entity_id for r in results]
        self.assertIn("class:services/user_service.py:UserService", retrieved_ids)

    def test_exact_method_name_retrieval(self):
        results = self.retriever.retrieve("create_user", k=5)
        self.assertTrue(len(results) > 0)
        top_item = results[0]
        self.assertIn("create_user", top_item.entity_id)
        self.assertEqual(top_item.retrieval_method, "bm25")

    def test_irrelevant_query_returns_empty(self):
        results = self.retriever.retrieve("quantum neural teleportation completely unrelated", k=5)
        self.assertEqual(results, [])


class TestRankingBehavior(unittest.TestCase):
    """Test BM25 ranking: term frequency saturation and document length normalization."""

    def test_term_frequency_ranks_higher(self):
        chunks = [
            CodeChunk(
                repository_id="repo",
                entity_id="doc1",
                entity_type="function",
                name="f1",
                file_path="f1.py",
                text_representation="search query term once in this document",
            ),
            CodeChunk(
                repository_id="repo",
                entity_id="doc2",
                entity_type="function",
                name="f2",
                file_path="f2.py",
                text_representation="search search search query query term term repeatedly",
            ),
        ]
        retriever = BM25Retriever.from_chunks(chunks)
        results = retriever.retrieve("search query term", k=2)
        self.assertEqual(len(results), 2)
        # doc2 has higher term frequency of all query terms
        self.assertEqual(results[0].entity_id, "doc2")
        self.assertGreater(results[0].score, results[1].score)

    def test_rarer_term_higher_idf_weight(self):
        chunks = [
            CodeChunk(
                repository_id="repo",
                entity_id="doc_common",
                entity_type="function",
                name="c1",
                file_path="c1.py",
                text_representation="common word alpha bravo",
            ),
            CodeChunk(
                repository_id="repo",
                entity_id="doc_rare",
                entity_type="function",
                name="c2",
                file_path="c2.py",
                text_representation="common word uniqueidentifier",
            ),
            CodeChunk(
                repository_id="repo",
                entity_id="doc_other",
                entity_type="function",
                name="c3",
                file_path="c3.py",
                text_representation="common word charlie delta",
            ),
        ]
        retriever = BM25Retriever.from_chunks(chunks)
        # 'common' is in all 3 docs; 'uniqueidentifier' is only in doc_rare
        results = retriever.retrieve("common uniqueidentifier", k=3)
        self.assertEqual(results[0].entity_id, "doc_rare")

    def test_shorter_document_favored_under_equal_term_matches(self):
        chunks = [
            CodeChunk(
                repository_id="repo",
                entity_id="doc_short",
                entity_type="function",
                name="s1",
                file_path="s1.py",
                text_representation="authenticate user token",
            ),
            CodeChunk(
                repository_id="repo",
                entity_id="doc_long",
                entity_type="function",
                name="s2",
                file_path="s2.py",
                text_representation="authenticate user token and a lot of extra verbose filler content repeated many times here in the long text file description",
            ),
        ]
        retriever = BM25Retriever.from_chunks(chunks)
        results = retriever.retrieve("authenticate user token", k=2)
        self.assertEqual(results[0].entity_id, "doc_short")
        self.assertGreater(results[0].score, results[1].score)


class TestTopKBehavior(unittest.TestCase):
    """Test top-k bounds and clamping."""

    @classmethod
    def setUpClass(cls):
        chunker = CodeEntityChunker(repo_path=SAMPLE_REPO)
        chunks = chunker.chunk_analysis_file(SAMPLE_ANALYSIS)
        cls.retriever = BM25Retriever.from_chunks(chunks)

    def test_top_k_1(self):
        results = self.retriever.retrieve("user", k=1)
        self.assertEqual(len(results), 1)

    def test_top_k_3(self):
        results = self.retriever.retrieve("user", k=3)
        self.assertEqual(len(results), 3)

    def test_top_k_zero_or_negative(self):
        self.assertEqual(self.retriever.retrieve("user", k=0), [])
        self.assertEqual(self.retriever.retrieve("user", k=-5), [])

    def test_top_k_exceeding_matches(self):
        results = self.retriever.retrieve("BaseModel", k=100)
        self.assertGreater(len(results), 0)
        self.assertLess(len(results), 100)


class TestDeterministicOrdering(unittest.TestCase):
    """Test deterministic retrieval results and tie-breaking."""

    def test_identical_repeated_calls(self):
        chunker = CodeEntityChunker(repo_path=SAMPLE_REPO)
        chunks = chunker.chunk_analysis_file(SAMPLE_ANALYSIS)
        retriever = BM25Retriever.from_chunks(chunks)

        res1 = retriever.retrieve("UserService get_user", k=5)
        res2 = retriever.retrieve("UserService get_user", k=5)

        self.assertEqual(len(res1), len(res2))
        for r1, r2 in zip(res1, res2):
            self.assertEqual(r1.entity_id, r2.entity_id)
            self.assertEqual(r1.score, r2.score)

    def test_tie_breaking_by_entity_id(self):
        # Two identical chunks with identical text, different entity IDs
        chunks = [
            CodeChunk(
                repository_id="repo",
                entity_id="entity_z",
                entity_type="function",
                name="fn",
                file_path="z.py",
                text_representation="exact identical content for testing tie break",
            ),
            CodeChunk(
                repository_id="repo",
                entity_id="entity_a",
                entity_type="function",
                name="fn",
                file_path="a.py",
                text_representation="exact identical content for testing tie break",
            ),
        ]
        retriever = BM25Retriever.from_chunks(chunks)
        results = retriever.retrieve("exact identical content", k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].score, results[1].score)
        # entity_a should precede entity_z alphabetically
        self.assertEqual(results[0].entity_id, "entity_a")
        self.assertEqual(results[1].entity_id, "entity_z")


class TestEmptyAndEdgeCaseHandling(unittest.TestCase):
    """Test edge cases: empty queries, empty corpora, and repository inputs."""

    def test_empty_query(self):
        retriever = BM25Retriever.from_chunks([
            CodeChunk(
                repository_id="repo",
                entity_id="id1",
                entity_type="file",
                name="f",
                file_path="f.py",
                text_representation="content",
            )
        ])
        self.assertEqual(retriever.retrieve("", k=5), [])
        self.assertEqual(retriever.retrieve("   \t\n ", k=5), [])
        self.assertEqual(retriever.retrieve("????", k=5), [])

    def test_empty_corpus(self):
        retriever = BM25Retriever()
        self.assertEqual(retriever.retrieve("any query", k=5), [])

    def test_repository_passed_to_retrieve_chunks_list(self):
        chunk = CodeChunk(
            repository_id="repo",
            entity_id="func:main.py:hello",
            entity_type="function",
            name="hello",
            file_path="main.py",
            text_representation="def hello(): print('hello')",
        )
        retriever = BM25Retriever()
        results = retriever.retrieve("hello print", repository=[chunk], k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].entity_id, "func:main.py:hello")

    def test_repository_passed_to_retrieve_kg(self):
        kg = build_from_analysis(SAMPLE_ANALYSIS)
        retriever = BM25Retriever(repo_path=SAMPLE_REPO)
        results = retriever.retrieve("UserService create_user", repository=kg, k=5)
        self.assertGreater(len(results), 0)


class TestEntityTypeFiltering(unittest.TestCase):
    """Test filtering by entity_type."""

    @classmethod
    def setUpClass(cls):
        chunker = CodeEntityChunker(repo_path=SAMPLE_REPO)
        chunks = chunker.chunk_analysis_file(SAMPLE_ANALYSIS)
        cls.retriever = BM25Retriever.from_chunks(chunks)

    def test_filter_single_type(self):
        results = self.retriever.retrieve("UserService", k=10, entity_type_filter="class")
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r.entity_type, "class")

    def test_filter_multiple_types(self):
        results = self.retriever.retrieve("user", k=10, entity_type_filter=["function", "method"])
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIn(r.entity_type, ["function", "method"])

    def test_filter_nonexistent_type(self):
        results = self.retriever.retrieve("user", k=10, entity_type_filter="nonexistent_type")
        self.assertEqual(results, [])


class TestRetrievedItemOutputModel(unittest.TestCase):
    """Test RetrievedItem fields and provenance metadata produced by BM25."""

    @classmethod
    def setUpClass(cls):
        chunker = CodeEntityChunker(repo_path=SAMPLE_REPO)
        chunks = chunker.chunk_analysis_file(SAMPLE_ANALYSIS)
        cls.retriever = BM25Retriever.from_chunks(chunks)

    def test_retrieved_item_structure(self):
        results = self.retriever.retrieve("UserService create_user", k=1)
        self.assertEqual(len(results), 1)
        item = results[0]

        self.assertIsInstance(item, RetrievedItem)
        self.assertEqual(item.retrieval_method, "bm25")
        self.assertGreater(item.score, 0.0)
        self.assertTrue(item.entity_id)
        self.assertTrue(item.file_path)
        self.assertIsNotNone(item.source_code)

        # Provenance verification
        prov = item.provenance
        self.assertIn("query_tokens", prov)
        self.assertIn("matched_terms", prov)
        self.assertIn("term_frequencies", prov)
        self.assertIn("doc_length", prov)
        self.assertIn("avg_doc_length", prov)
        self.assertIn("k1", prov)
        self.assertIn("b", prov)

        # Serialization roundtrip
        item_dict = item.to_dict()
        self.assertEqual(item_dict["entity_id"], item.entity_id)
        self.assertEqual(item_dict["retrieval_method"], "bm25")


if __name__ == "__main__":
    unittest.main()
