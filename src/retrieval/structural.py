"""Structural Retrieval Engine for ProjectGenome.

Retrieves repository code units using deterministic lexical signals and explicit
Knowledge Graph relationships (CALLS, INHERITS, IMPORTS, CONTAINS, DEPENDS_ON)
as specified in Phase 10 of the ProjectGenome roadmap.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.knowledge_graph.graph import KnowledgeGraph, validate_relationship_type
from src.knowledge_graph.schema import RelationshipType
from .base import BaseRetriever, RetrievedItem
from .chunker import CodeChunk, CodeEntityChunker


# Canonical relationship types for structural retrieval (Phase 10.2)
STRUCTURAL_RELATIONSHIP_TYPES = [
    RelationshipType.CONTAINS,
    RelationshipType.IMPORTS,
    RelationshipType.INHERITS,
    RelationshipType.CALLS,
    RelationshipType.DEPENDS_ON,
]


class StructuralRetriever(BaseRetriever):
    """Retrieves repository context using deterministic Knowledge Graph traversal."""

    def __init__(
        self,
        graph: KnowledgeGraph | None = None,
        repo_path: str | Path | None = None,
        chunker: CodeEntityChunker | None = None,
        max_depth: int = 2,
        decay_factor: float = 0.6,
        allowed_relationship_types: list[RelationshipType | str] | None = None,
        direction: str = "both",
    ) -> None:
        """Initialize StructuralRetriever.

        Args:
            graph: Optional KnowledgeGraph instance. Can also be passed to retrieve().
            repo_path: Optional path to repository files for reading source code.
            chunker: Optional CodeEntityChunker. If None, created automatically.
            max_depth: Maximum hops for graph traversal from seed candidates (default: 2).
            decay_factor: Multiplicative score attenuation per traversal hop (default: 0.6).
            allowed_relationship_types: Filter for edge types to traverse (default: all Phase 10 types).
            direction: Traversal direction: 'out', 'in', or 'both' (default: 'both').
        """
        self.repo_path = Path(repo_path) if repo_path else None
        self.chunker = chunker or CodeEntityChunker(repo_path=self.repo_path)
        self.max_depth = max(0, max_depth)
        self.decay_factor = decay_factor
        self.direction = direction

        if allowed_relationship_types is not None:
            self.allowed_types = {
                validate_relationship_type(rt).value for rt in allowed_relationship_types
            }
        else:
            self.allowed_types = {rt.value for rt in STRUCTURAL_RELATIONSHIP_TYPES}

        self.graph: KnowledgeGraph | None = None
        self._chunks_by_id: dict[str, CodeChunk] = {}
        self._name_to_ids: dict[str, set[str]] = {}
        self._lower_name_to_ids: dict[str, set[str]] = {}
        self._qualified_to_ids: dict[str, set[str]] = {}
        self._lower_qualified_to_ids: dict[str, set[str]] = {}
        self._file_to_ids: dict[str, set[str]] = {}
        self._token_to_ids: dict[str, set[str]] = {}

        if graph is not None:
            self.set_graph(graph)

    def set_graph(self, graph: KnowledgeGraph) -> None:
        """Load and index a KnowledgeGraph for structural retrieval."""
        self.graph = graph
        self._chunks_by_id.clear()
        self._name_to_ids.clear()
        self._lower_name_to_ids.clear()
        self._qualified_to_ids.clear()
        self._lower_qualified_to_ids.clear()
        self._file_to_ids.clear()
        self._token_to_ids.clear()

        # Chunk the KG nodes to obtain source code, locations, and representations
        chunks = self.chunker.chunk_knowledge_graph(graph)
        for chunk in chunks:
            self._chunks_by_id[chunk.entity_id] = chunk

        # Build fast lookup indexes from all non-synthetic nodes
        all_nodes = graph.get_all_nodes()
        for node_id, data in all_nodes.items():
            if data.get("is_synthetic"):
                continue

            node_type = str(data.get("type", "")).lower()
            if node_type in ("repository", "directory", "unresolved_reference"):
                continue

            name = data.get("name", "")
            file_path = data.get("path", "")
            props = dict(data.get("properties") or {})
            qualified_name = props.get("qualified_name") or name

            if name:
                self._name_to_ids.setdefault(name, set()).add(node_id)
                self._lower_name_to_ids.setdefault(name.lower(), set()).add(node_id)

            if qualified_name:
                self._qualified_to_ids.setdefault(qualified_name, set()).add(node_id)
                self._lower_qualified_to_ids.setdefault(qualified_name.lower(), set()).add(node_id)

            if file_path:
                self._file_to_ids.setdefault(file_path, set()).add(node_id)
                base = Path(file_path).name
                self._file_to_ids.setdefault(base, set()).add(node_id)

            # Sub-token index for identifiers (split CamelCase and snake_case)
            for raw_str in (name, qualified_name):
                sub_tokens = self._extract_identifier_tokens(raw_str)
                for st in sub_tokens:
                    self._token_to_ids.setdefault(st.lower(), set()).add(node_id)

    @staticmethod
    def _extract_identifier_tokens(text: str) -> set[str]:
        """Extract sub-tokens from identifier names like UserService or create_user."""
        if not text:
            return set()
        parts = re.split(r'[_.\s/]+', text)
        tokens = set()
        for p in parts:
            if not p:
                continue
            tokens.add(p)
            subparts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', p)
            if len(subparts) > 1:
                for sp in subparts:
                    if len(sp) > 1:
                        tokens.add(sp)
        return {t for t in tokens if len(t) > 1}

    def identify_candidates(self, question: str) -> list[tuple[str, float, str]]:
        """Identify candidate seed entities from the natural language question.

        Returns:
            List of (entity_id, match_score, matched_signal) sorted descending by score.
        """
        if not self.graph:
            return []

        q_text = question.strip()
        file_candidates = set(re.findall(r'[\w/.-]+\.py\b', q_text))
        dotted_candidates = set(re.findall(r'\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+\b', q_text))
        word_candidates = set(re.findall(r'\b[a-zA-Z_]\w*\b', q_text))

        candidate_scores: dict[str, tuple[float, str]] = {}

        def record_candidate(node_id: str, score: float, signal: str) -> None:
            if node_id not in candidate_scores or score > candidate_scores[node_id][0]:
                candidate_scores[node_id] = (score, signal)

        # 1. Exact match on qualified names (score: 1.0)
        for dc in dotted_candidates:
            if dc in self._qualified_to_ids:
                for nid in self._qualified_to_ids[dc]:
                    record_candidate(nid, 1.0, dc)
            elif dc.lower() in self._lower_qualified_to_ids:
                for nid in self._lower_qualified_to_ids[dc.lower()]:
                    record_candidate(nid, 0.95, dc)

        # 2. Exact match on file paths (score: 0.95) and file basenames (score: 0.90)
        for fc in file_candidates:
            if fc in self._file_to_ids:
                for nid in self._file_to_ids[fc]:
                    record_candidate(nid, 0.95, fc)
            elif fc.lower() in self._file_to_ids:
                for nid in self._file_to_ids[fc.lower()]:
                    record_candidate(nid, 0.90, fc)

        # 3. Exact match on entity names (score: 0.95 case-sensitive, 0.85 case-insensitive)
        for wc in word_candidates:
            if wc in self._name_to_ids:
                for nid in self._name_to_ids[wc]:
                    record_candidate(nid, 0.95, wc)
            elif wc.lower() in self._lower_name_to_ids:
                for nid in self._lower_name_to_ids[wc.lower()]:
                    record_candidate(nid, 0.85, wc)

        # 4. Sub-token matching (score: 0.70)
        # Check if question words match specific identifier sub-tokens
        for wc in word_candidates:
            low_wc = wc.lower()
            if low_wc in self._token_to_ids and len(low_wc) > 2:
                for nid in self._token_to_ids[low_wc]:
                    record_candidate(nid, 0.70, wc)

        # Sort seeds deterministically: score descending, then entity_id ascending
        sorted_candidates = [
            (nid, score_signal[0], score_signal[1])
            for nid, score_signal in sorted(
                candidate_scores.items(),
                key=lambda item: (-item[1][0], item[0]),
            )
        ]
        return sorted_candidates

    def traverse_structural_context(
        self,
        seed_candidates: list[tuple[str, float, str]],
        max_depth: int | None = None,
    ) -> list[dict[str, Any]]:
        """Traverse the Knowledge Graph outward and inward from candidate seed entities.

        Args:
            seed_candidates: List of (seed_id, seed_score, matched_signal) tuples.
            max_depth: Maximum traversal depth. Defaults to self.max_depth.

        Returns:
            List of traversed entity records with structural scores and complete provenance.
        """
        if not self.graph or not seed_candidates:
            return []

        depth_limit = self.max_depth if max_depth is None else max(0, max_depth)
        g = self.graph.underlying_graph

        best_records: dict[str, dict[str, Any]] = {}

        for seed_id, seed_score, matched_signal in seed_candidates:
            if not self.graph.has_node(seed_id):
                continue

            queue: list[tuple[str, int, list[str], list[str]]] = [
                (seed_id, 0, [], [seed_id])
            ]
            visited_in_seed_bfs: set[str] = {seed_id}

            while queue:
                curr_node, dist, rel_path, trav_path = queue.pop(0)

                decayed_score = seed_score * (self.decay_factor ** dist)

                if curr_node not in best_records or decayed_score > best_records[curr_node]["score"]:
                    best_records[curr_node] = {
                        "entity_id": curr_node,
                        "score": round(decayed_score, 6),
                        "distance": dist,
                        "starting_entity": seed_id,
                        "relationship_path": list(rel_path),
                        "traversal_path": list(trav_path),
                        "matched_signal": matched_signal,
                    }

                if dist >= depth_limit:
                    continue

                if self.direction in ("out", "both"):
                    for _, target, edge_data in g.out_edges(curr_node, data=True):
                        rel_type = edge_data.get("type")
                        if self.allowed_types and rel_type not in self.allowed_types:
                            continue
                        if target not in visited_in_seed_bfs:
                            visited_in_seed_bfs.add(target)
                            queue.append((
                                target,
                                dist + 1,
                                rel_path + [rel_type],
                                trav_path + [target],
                            ))

                if self.direction in ("in", "both"):
                    for source, _, edge_data in g.in_edges(curr_node, data=True):
                        rel_type = edge_data.get("type")
                        if self.allowed_types and rel_type not in self.allowed_types:
                            continue
                        if source not in visited_in_seed_bfs:
                            visited_in_seed_bfs.add(source)
                            queue.append((
                                source,
                                dist + 1,
                                rel_path + [rel_type],
                                trav_path + [source],
                            ))

        valid_records: list[dict[str, Any]] = []
        for nid, record in best_records.items():
            node_data = self.graph.get_node(nid) or {}
            if node_data.get("is_synthetic"):
                continue

            node_type = str(node_data.get("type", "")).lower()
            if node_type not in ("file", "class", "function", "method"):
                continue

            record["node_data"] = node_data
            valid_records.append(record)

        valid_records.sort(
            key=lambda r: (-r["score"], r["distance"], r["entity_id"])
        )
        return valid_records

    def retrieve(
        self,
        question: str,
        repository: Any = None,
        k: int = 10,
    ) -> list[RetrievedItem]:
        """Retrieve top-k code units using structural Knowledge Graph traversal.

        Args:
            question: Natural language question or query string.
            repository: Optional KnowledgeGraph instance.
            k: Number of top items to return (k > 0).

        Returns:
            List of RetrievedItem objects sorted descending by structural score.
        """
        if repository is not None and isinstance(repository, KnowledgeGraph):
            if self.graph is not repository:
                self.set_graph(repository)
        elif self.graph is None:
            raise ValueError(
                "KnowledgeGraph is required for StructuralRetriever. Pass graph to __init__ or retrieve(repository=...)."
            )

        candidates = self.identify_candidates(question)
        traversed_records = self.traverse_structural_context(candidates)

        retrieved_items: list[RetrievedItem] = []
        for rec in traversed_records:
            entity_id = rec["entity_id"]
            node_data = rec["node_data"]
            chunk = self._chunks_by_id.get(entity_id)

            file_path = (chunk.file_path if chunk else None) or node_data.get("path", "")
            source_code = (chunk.source_code if chunk else None) or node_data.get("properties", {}).get("source_code", "")
            entity_type = (chunk.entity_type if chunk else None) or node_data.get("type", "").lower()
            name = (chunk.name if chunk else None) or node_data.get("name", "")
            qualified_name = (chunk.qualified_name if chunk else None) or node_data.get("properties", {}).get("qualified_name", name)
            start_line = (chunk.start_line if chunk else None) or (node_data.get("location") or {}).get("start_line")
            end_line = (chunk.end_line if chunk else None) or (node_data.get("location") or {}).get("end_line")
            metadata = dict(chunk.metadata if chunk else (node_data.get("properties") or {}))

            provenance = {
                "starting_entity": rec["starting_entity"],
                "relationship_path": rec["relationship_path"],
                "target_entity": entity_id,
                "distance": rec["distance"],
                "traversal_path": rec["traversal_path"],
                "matched_signal": rec["matched_signal"],
            }

            retrieved_items.append(
                RetrievedItem(
                    entity_id=entity_id,
                    file_path=file_path,
                    source_code=source_code,
                    score=rec["score"],
                    retrieval_method="structural",
                    provenance=provenance,
                    entity_type=entity_type,
                    name=name,
                    qualified_name=qualified_name,
                    start_line=start_line,
                    end_line=end_line,
                    metadata=metadata,
                )
            )

        return retrieved_items[:k]
