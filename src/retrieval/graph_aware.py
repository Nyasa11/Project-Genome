"""Graph-Aware Retrieval Engine for ProjectGenome.

Combines initial semantic CodeBERT retrieval with Knowledge Graph expansion and reranking
using repository structural relationships (CONTAINS, IMPORTS, INHERITS, CALLS, DEPENDS_ON)
conforming to the BaseRetriever common interface and producing normalized RetrievedItem outputs.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from src.knowledge_graph.builder import build_from_analysis
from src.knowledge_graph.graph import KnowledgeGraph, validate_relationship_type
from src.knowledge_graph.schema import RelationshipType
from .base import BaseRetriever, RetrievedItem
from .chunker import CodeChunk, CodeEntityChunker
from .hybrid import normalize_scores
from .semantic import SemanticRetriever

# Canonical repository relationship types used for graph expansion
DEFAULT_GRAPH_RELATIONSHIP_TYPES = [
    RelationshipType.CONTAINS,
    RelationshipType.IMPORTS,
    RelationshipType.INHERITS,
    RelationshipType.CALLS,
    RelationshipType.DEPENDS_ON,
]


class GraphAwareRetriever(BaseRetriever):
    """Deterministic Graph-Aware Retriever.

    Retrieval pipeline:
      1. Initial dense semantic retrieval via CodeBERT to retrieve seed candidates.
      2. Ground seeds in the Software Knowledge Graph.
      3. Controlled graph traversal up to `graph_depth` across allowed relationships.
      4. Graph evidence scoring with distance decay.
      5. Linear score combination:
         graph_aware_score = semantic_weight * normalized_semantic_score + graph_weight * graph_evidence_score
      6. Deterministic tie-breaking and provenance enrichment.
    """

    def __init__(
        self,
        semantic_retriever: SemanticRetriever | None = None,
        graph: KnowledgeGraph | None = None,
        chunks: list[CodeChunk] | None = None,
        semantic_weight: float = 0.7,
        graph_weight: float = 0.3,
        graph_depth: int = 1,
        decay_factor: float = 0.5,
        allowed_relationship_types: list[RelationshipType | str] | None = None,
        candidate_k: int | None = None,
        normalization: str = "minmax",
        direction: str = "both",
        repo_path: str | Path | None = None,
        chunker: CodeEntityChunker | None = None,
    ) -> None:
        """Initialize GraphAwareRetriever.

        Args:
            semantic_retriever: Optional pre-configured SemanticRetriever instance.
            graph: Optional KnowledgeGraph instance.
            chunks: Optional list of CodeChunk objects to index.
            semantic_weight: Linear combination weight for semantic score (default: 0.7).
            graph_weight: Linear combination weight for graph evidence score (default: 0.3).
            graph_depth: Maximum hops for graph expansion from semantic seeds (default: 1).
            decay_factor: Multiplicative attenuation per hop beyond distance 1 (default: 0.5).
            allowed_relationship_types: Filter for edge types to traverse.
            candidate_k: Optional number of candidate units to fetch from semantic retrieval.
            normalization: Score normalization strategy (minmax, max, none).
            direction: Traversal direction: out, in, or both (default: both).
            repo_path: Optional repository file path for reading source code.
            chunker: Optional CodeEntityChunker instance.
        """
        self._validate_weights(semantic_weight, graph_weight)
        self.semantic_weight = float(semantic_weight)
        self.graph_weight = float(graph_weight)

        if not isinstance(graph_depth, int) or graph_depth < 0:
            raise ValueError(
                f"graph_depth must be a non-negative integer. Got {graph_depth!r}"
            )
        self.graph_depth = graph_depth

        if not isinstance(decay_factor, (int, float)) or decay_factor <= 0 or decay_factor > 1:
            raise ValueError(
                f"decay_factor must be in (0, 1]. Got {decay_factor!r}"
            )
        self.decay_factor = float(decay_factor)

        self.normalization = normalization.lower()
        if self.normalization not in ("minmax", "max", "none"):
            raise ValueError(
                f"Invalid normalization strategy: {normalization!r}. Must be minmax, max, or none."
            )

        if direction not in ("out", "in", "both"):
            raise ValueError(
                f"Invalid direction: {direction!r}. Must be out, in, or both."
            )
        self.direction = direction

        if candidate_k is not None:
            if not isinstance(candidate_k, int) or candidate_k <= 0:
                raise ValueError(
                    f"candidate_k must be a positive integer if provided. Got {candidate_k!r}"
                )
        self.candidate_k = candidate_k

        if allowed_relationship_types is not None:
            self.allowed_relationship_types = [
                validate_relationship_type(rt).value for rt in allowed_relationship_types
            ]
        else:
            self.allowed_relationship_types = [
                rt.value for rt in DEFAULT_GRAPH_RELATIONSHIP_TYPES
            ]

        self.repo_path = Path(repo_path) if repo_path else None
        self.chunker = chunker or CodeEntityChunker(repo_path=self.repo_path)

        self.semantic_retriever = semantic_retriever or SemanticRetriever(
            chunks=chunks,
            repo_path=self.repo_path,
            chunker=self.chunker,
        )

        self.graph: KnowledgeGraph | None = None
        self._chunks_by_id: dict[str, CodeChunk] = {}

        if chunks:
            for c in chunks:
                self._chunks_by_id[c.entity_id] = c
            if len(self.semantic_retriever) == 0:
                self.semantic_retriever.index(chunks)

        if graph is not None:
            self.set_graph(graph)

    @staticmethod
    def _validate_weights(semantic_weight: Any, graph_weight: Any) -> None:
        """Validate combination weights."""
        if not isinstance(semantic_weight, (int, float)) or not isinstance(graph_weight, (int, float)):
            raise ValueError(
                f"Weights must be numeric. Got semantic_weight={semantic_weight!r}, graph_weight={graph_weight!r}"
            )
        if (
            math.isnan(semantic_weight)
            or math.isnan(graph_weight)
            or math.isinf(semantic_weight)
            or math.isinf(graph_weight)
        ):
            raise ValueError(
                f"Weights must be finite numbers. Got semantic_weight={semantic_weight}, graph_weight={graph_weight}"
            )
        if semantic_weight < 0 or graph_weight < 0:
            raise ValueError(
                f"Weights must be non-negative. Got semantic_weight={semantic_weight}, graph_weight={graph_weight}"
            )
        if semantic_weight == 0 and graph_weight == 0:
            raise ValueError("At least one weight must be greater than 0.")

    def set_graph(self, graph: KnowledgeGraph) -> None:
        """Load and index a KnowledgeGraph for graph-aware retrieval."""
        self.graph = graph
        chunks = self.chunker.chunk_knowledge_graph(graph)
        for chunk in chunks:
            self._chunks_by_id[chunk.entity_id] = chunk

    @property
    def chunks(self) -> list[CodeChunk]:
        """Access indexed CodeChunks."""
        return self.semantic_retriever.chunks or list(self._chunks_by_id.values())

    def index(self, chunks: list[CodeChunk]) -> None:
        """Index repository chunks in the semantic retriever and local metadata cache."""
        self.semantic_retriever.index(chunks)
        for c in chunks:
            self._chunks_by_id[c.entity_id] = c

    def add_chunks(self, chunks: list[CodeChunk]) -> None:
        """Append repository chunks to the semantic retriever and local metadata cache."""
        self.semantic_retriever.add_chunks(chunks)
        for c in chunks:
            self._chunks_by_id[c.entity_id] = c

    def retrieve(
        self,
        question: str,
        repository: Any = None,
        k: int = 10,
        entity_type_filter: str | list[str] | None = None,
        candidate_k: int | None = None,
    ) -> list[RetrievedItem]:
        """Retrieve top-k code units using graph-expanded semantic retrieval.

        Args:
            question: Natural language question or query string.
            repository: Optional repository representation (KnowledgeGraph, chunks list, or file path).
            k: Maximum number of retrieved items to return (k > 0).
            entity_type_filter: Optional entity type or list of entity types to restrict to.
            candidate_k: Optional candidate pool size to fetch from semantic retrieval.

        Returns:
            List of RetrievedItem objects sorted descending by graph-aware score.
            Ties are broken deterministically by semantic score descending, then entity_id ascending.
        """
        if k <= 0:
            return []

        if not question or not question.strip():
            return []

        # Handle repository input if provided
        if repository is not None:
            self._handle_repository_input(repository)

        fetch_k = candidate_k or self.candidate_k or max(k * 2, 20)
        fetch_k = max(fetch_k, k)

        # Retrieve initial semantic candidates (seeds)
        semantic_items = self.semantic_retriever.retrieve(
            question=question,
            repository=repository,
            k=fetch_k,
        )

        # If entity_type_filter is specified, also fetch type-filtered semantic candidates
        # to ensure strong candidates of that type are not eclipsed before expansion
        if entity_type_filter is not None:
            filtered_items = self.semantic_retriever.retrieve(
                question=question,
                repository=repository,
                k=fetch_k,
                entity_type_filter=entity_type_filter,
            )
            # Merge while preserving order and uniqueness
            seen_ids = {it.entity_id for it in semantic_items}
            for it in filtered_items:
                if it.entity_id not in seen_ids:
                    semantic_items.append(it)
                    seen_ids.add(it.entity_id)

        if not semantic_items:
            return []

        # Normalize semantic scores onto [0, 1]
        raw_sem_scores = {it.entity_id: it.score for it in semantic_items}
        sem_item_map = {it.entity_id: it for it in semantic_items}
        norm_sem_scores = normalize_scores(raw_sem_scores, method=self.normalization)

        # Expand through Knowledge Graph around semantic seeds
        evidence_records = self._expand_and_score_graph(semantic_items, norm_sem_scores)

        # Combine scores for all candidate entities (seeds + expanded neighbors)
        all_candidate_ids = set(norm_sem_scores.keys()) | set(evidence_records.keys())

        # Prepare allowed entity types filter if specified
        allowed_types: set[str] | None = None
        if entity_type_filter is not None:
            if isinstance(entity_type_filter, str):
                allowed_types = {entity_type_filter.lower()}
            else:
                allowed_types = {t.lower() for t in entity_type_filter}

        scored_items: list[RetrievedItem] = []

        for entity_id in all_candidate_ids:
            sem_norm = norm_sem_scores.get(entity_id, 0.0)
            sem_raw = raw_sem_scores.get(entity_id)
            ev_rec = evidence_records.get(entity_id, {})

            graph_evidence = ev_rec.get("evidence_score", 0.0)
            is_seed = entity_id in sem_item_map
            primary_seed = ev_rec.get("seed_id", entity_id if is_seed else "")
            responsible_seeds = ev_rec.get("responsible_seeds", [primary_seed] if primary_seed else [])
            dist = ev_rec.get("distance", 0 if is_seed else 1)
            rel_types = ev_rec.get("relationship_types", [])
            trav_path = ev_rec.get("traversal_path", [entity_id])

            sem_contrib = self.semantic_weight * sem_norm
            graph_contrib = self.graph_weight * graph_evidence
            final_score = sem_contrib + graph_contrib

            # Resolve entity representation (from semantic item, chunk cache, or KG node)
            entity_repr = self._resolve_entity_representation(entity_id, sem_item_map)
            if entity_repr is None:
                continue

            # Apply entity-type filter consistently to final candidates
            if allowed_types is not None:
                e_type = (entity_repr.get("entity_type") or "").lower()
                if e_type not in allowed_types:
                    continue

            provenance = {
                "retrieval_method": "graph_aware",
                "semantic_raw_score": sem_raw,
                "semantic_normalized_score": round(sem_norm, 6),
                "graph_evidence_score": round(graph_evidence, 6),
                "graph_contribution": round(graph_contrib, 6),
                "semantic_contribution": round(sem_contrib, 6),
                "final_graph_aware_score": round(final_score, 6),
                "is_semantic_seed": is_seed,
                "is_seed": is_seed,
                "seed_entity": primary_seed,
                "seed_entities": responsible_seeds,
                "graph_distance": dist,
                "relationship_types": rel_types,
                "traversal_path": trav_path,
                "graph_depth": self.graph_depth,
                "semantic_weight": self.semantic_weight,
                "graph_weight": self.graph_weight,
                "normalization": self.normalization,
                "decay_factor": self.decay_factor,
                "combined_score": round(final_score, 6),
                "score": round(final_score, 6),
            }

            item = RetrievedItem(
                entity_id=entity_id,
                file_path=entity_repr["file_path"],
                source_code=entity_repr["source_code"],
                score=round(final_score, 6),
                retrieval_method="graph_aware",
                provenance=provenance,
                entity_type=entity_repr["entity_type"],
                name=entity_repr["name"],
                qualified_name=entity_repr["qualified_name"],
                start_line=entity_repr["start_line"],
                end_line=entity_repr["end_line"],
                metadata=entity_repr["metadata"],
            )
            scored_items.append(item)

        # Deterministic ranking:
        # 1. graph-aware score descending
        # 2. semantic normalized score descending
        # 3. entity_id ascending
        scored_items.sort(
            key=lambda it: (
                -round(it.score, 6),
                -round(float(it.provenance.get("semantic_normalized_score", 0.0)), 6),
                it.entity_id,
            )
        )

        return scored_items[:k]

    def _expand_and_score_graph(
        self,
        semantic_seeds: list[RetrievedItem],
        norm_sem_scores: dict[str, float],
    ) -> dict[str, dict[str, Any]]:
        """Traverse Knowledge Graph around semantic seeds and compute graph evidence scores."""
        evidence_records: dict[str, dict[str, Any]] = {}

        if self.graph is None or self.graph.number_of_nodes() == 0:
            return evidence_records

        g = self.graph.underlying_graph

        for seed in semantic_seeds:
            seed_id = seed.entity_id
            seed_norm = norm_sem_scores.get(seed_id, 0.0)

            # Check if seed exists in Knowledge Graph
            if not self.graph.has_node(seed_id):
                continue

            # Seed entity itself has graph presence in KG (distance 0)
            if seed_id not in evidence_records or seed_norm > evidence_records[seed_id]["evidence_score"]:
                evidence_records[seed_id] = {
                    "evidence_score": seed_norm,
                    "distance": 0,
                    "seed_id": seed_id,
                    "responsible_seeds": [seed_id],
                    "relationship_types": [],
                    "traversal_path": [seed_id],
                }

            if self.graph_depth == 0:
                continue

            # BFS traversal outward/inward up to graph_depth
            queue: list[tuple[str, int, list[str], list[str]]] = [(seed_id, 0, [], [seed_id])]
            visited_in_seed_bfs: set[str] = {seed_id}

            while queue:
                curr_node, dist, rel_path, trav_path = queue.pop(0)

                if dist >= self.graph_depth:
                    continue

                next_dist = dist + 1
                # Deterministic decay by graph distance:
                # distance 1 -> 1.0
                # distance 2 -> 0.5
                # distance 3 -> 0.25
                decay = 1.0 if next_dist == 1 else (self.decay_factor ** (next_dist - 1))
                propagated_evidence = seed_norm * decay

                # Outward edges
                if self.direction in ("out", "both"):
                    for _, target, edge_data in g.out_edges(curr_node, data=True):
                        rel_type = edge_data.get("type")
                        if self.allowed_relationship_types and rel_type not in self.allowed_relationship_types:
                            continue
                        if target not in visited_in_seed_bfs:
                            visited_in_seed_bfs.add(target)
                            new_rel_path = rel_path + [rel_type]
                            new_trav_path = trav_path + [target]
                            self._update_evidence_record(
                                evidence_records,
                                target,
                                propagated_evidence,
                                next_dist,
                                seed_id,
                                new_rel_path,
                                new_trav_path,
                            )
                            queue.append((target, next_dist, new_rel_path, new_trav_path))

                # Inward edges
                if self.direction in ("in", "both"):
                    for source, _, edge_data in g.in_edges(curr_node, data=True):
                        rel_type = edge_data.get("type")
                        if self.allowed_relationship_types and rel_type not in self.allowed_relationship_types:
                            continue
                        if source not in visited_in_seed_bfs:
                            visited_in_seed_bfs.add(source)
                            new_rel_path = rel_path + [rel_type]
                            new_trav_path = trav_path + [source]
                            self._update_evidence_record(
                                evidence_records,
                                source,
                                propagated_evidence,
                                next_dist,
                                seed_id,
                                new_rel_path,
                                new_trav_path,
                            )
                            queue.append((source, next_dist, new_rel_path, new_trav_path))

        return evidence_records

    def _update_evidence_record(
        self,
        evidence_records: dict[str, dict[str, Any]],
        entity_id: str,
        evidence: float,
        dist: int,
        seed_id: str,
        rel_path: list[str],
        trav_path: list[str],
    ) -> None:
        """Update or insert graph evidence record for a target entity."""
        if entity_id not in evidence_records:
            evidence_records[entity_id] = {
                "evidence_score": round(evidence, 6),
                "distance": dist,
                "seed_id": seed_id,
                "responsible_seeds": [seed_id],
                "relationship_types": list(rel_path),
                "traversal_path": list(trav_path),
            }
        else:
            rec = evidence_records[entity_id]
            if seed_id not in rec["responsible_seeds"]:
                rec["responsible_seeds"].append(seed_id)
                rec["responsible_seeds"].sort()

            # If strictly better evidence is found, update primary seed and path
            if evidence > rec["evidence_score"]:
                rec["evidence_score"] = round(evidence, 6)
                rec["distance"] = dist
                rec["seed_id"] = seed_id
                rec["relationship_types"] = list(rel_path)
                rec["traversal_path"] = list(trav_path)
            elif math.isclose(evidence, rec["evidence_score"], abs_tol=1e-6):
                # Deterministic tie-breaking on path: shorter distance, then seed_id
                if dist < rec["distance"] or (dist == rec["distance"] and seed_id < rec["seed_id"]):
                    rec["distance"] = dist
                    rec["seed_id"] = seed_id
                    rec["relationship_types"] = list(rel_path)
                    rec["traversal_path"] = list(trav_path)

    def _resolve_entity_representation(
        self,
        entity_id: str,
        sem_item_map: dict[str, RetrievedItem],
    ) -> dict[str, Any] | None:
        """Resolve entity metadata and source code from semantic cache, chunk cache, or KG node."""
        if entity_id in sem_item_map:
            it = sem_item_map[entity_id]
            return {
                "entity_id": it.entity_id,
                "file_path": it.file_path,
                "source_code": it.source_code,
                "entity_type": it.entity_type,
                "name": it.name,
                "qualified_name": it.qualified_name,
                "start_line": it.start_line,
                "end_line": it.end_line,
                "metadata": dict(it.metadata or {}),
            }

        chunk = self._chunks_by_id.get(entity_id)
        node_data = self.graph.get_node(entity_id) if self.graph else None

        if chunk is not None:
            return {
                "entity_id": chunk.entity_id,
                "file_path": chunk.file_path,
                "source_code": chunk.source_code,
                "entity_type": chunk.entity_type,
                "name": chunk.name,
                "qualified_name": chunk.qualified_name,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "metadata": dict(chunk.metadata or {}),
            }

        if node_data is not None:
            # Skip synthetic or non-code container entities
            if node_data.get("is_synthetic"):
                return None
            node_type = str(node_data.get("type", "")).lower()
            if node_type not in ("file", "class", "function", "method"):
                return None

            name = node_data.get("name", "")
            props = dict(node_data.get("properties") or {})
            loc = node_data.get("location") or {}

            return {
                "entity_id": entity_id,
                "file_path": node_data.get("path", ""),
                "source_code": props.get("source_code", ""),
                "entity_type": node_type,
                "name": name,
                "qualified_name": props.get("qualified_name", name),
                "start_line": loc.get("start_line"),
                "end_line": loc.get("end_line"),
                "metadata": props,
            }

        return None

    def _handle_repository_input(self, repository: Any) -> None:
        """Handle optional repository parameter passed to retrieve()."""
        if isinstance(repository, KnowledgeGraph):
            if self.graph is not repository:
                self.set_graph(repository)
            return

        if hasattr(repository, "underlying_graph") and hasattr(repository, "has_node"):
            if self.graph is not repository:
                self.set_graph(repository)
            return

        if isinstance(repository, (str, Path)):
            path = Path(repository)
            if path.is_file():
                try:
                    graph = build_from_analysis(path)
                    self.set_graph(graph)
                except Exception:
                    pass

    @classmethod
    def from_retriever(
        cls,
        semantic_retriever: SemanticRetriever,
        graph: KnowledgeGraph | None = None,
        semantic_weight: float = 0.7,
        graph_weight: float = 0.3,
        graph_depth: int = 1,
        decay_factor: float = 0.5,
        allowed_relationship_types: list[RelationshipType | str] | None = None,
        candidate_k: int | None = None,
        normalization: str = "minmax",
        direction: str = "both",
        repo_path: str | Path | None = None,
        chunker: CodeEntityChunker | None = None,
    ) -> GraphAwareRetriever:
        """Create a GraphAwareRetriever from an existing semantic retriever and KnowledgeGraph."""
        return cls(
            semantic_retriever=semantic_retriever,
            graph=graph,
            semantic_weight=semantic_weight,
            graph_weight=graph_weight,
            graph_depth=graph_depth,
            decay_factor=decay_factor,
            allowed_relationship_types=allowed_relationship_types,
            candidate_k=candidate_k,
            normalization=normalization,
            direction=direction,
            repo_path=repo_path,
            chunker=chunker,
        )

    @classmethod
    def from_knowledge_graph(
        cls,
        graph: KnowledgeGraph,
        repo_path: str | Path | None = None,
        semantic_weight: float = 0.7,
        graph_weight: float = 0.3,
        graph_depth: int = 1,
        decay_factor: float = 0.5,
        allowed_relationship_types: list[RelationshipType | str] | None = None,
        candidate_k: int | None = None,
        normalization: str = "minmax",
        direction: str = "both",
        embedder: Any | None = None,
        batch_size: int = 16,
    ) -> GraphAwareRetriever:
        """Create and index a GraphAwareRetriever directly from a KnowledgeGraph."""
        chunker = CodeEntityChunker(repo_path=repo_path)
        chunks = chunker.chunk_knowledge_graph(graph)
        sem_retriever = SemanticRetriever(
            chunks=chunks,
            embedder=embedder,
            repo_path=repo_path,
            chunker=chunker,
            batch_size=batch_size,
        )
        return cls(
            semantic_retriever=sem_retriever,
            graph=graph,
            chunks=chunks,
            semantic_weight=semantic_weight,
            graph_weight=graph_weight,
            graph_depth=graph_depth,
            decay_factor=decay_factor,
            allowed_relationship_types=allowed_relationship_types,
            candidate_k=candidate_k,
            normalization=normalization,
            direction=direction,
            repo_path=repo_path,
            chunker=chunker,
        )

    @classmethod
    def from_analysis_file(
        cls,
        analysis_file_path: str | Path,
        repo_path: str | Path | None = None,
        semantic_weight: float = 0.7,
        graph_weight: float = 0.3,
        graph_depth: int = 1,
        decay_factor: float = 0.5,
        allowed_relationship_types: list[RelationshipType | str] | None = None,
        candidate_k: int | None = None,
        normalization: str = "minmax",
        direction: str = "both",
        embedder: Any | None = None,
        batch_size: int = 16,
    ) -> GraphAwareRetriever:
        """Create and index a GraphAwareRetriever from an analyzer JSON file."""
        graph = build_from_analysis(analysis_file_path)
        return cls.from_knowledge_graph(
            graph=graph,
            repo_path=repo_path,
            semantic_weight=semantic_weight,
            graph_weight=graph_weight,
            graph_depth=graph_depth,
            decay_factor=decay_factor,
            allowed_relationship_types=allowed_relationship_types,
            candidate_k=candidate_k,
            normalization=normalization,
            direction=direction,
            embedder=embedder,
            batch_size=batch_size,
        )

    @classmethod
    def from_chunks(
        cls,
        chunks: list[CodeChunk],
        graph: KnowledgeGraph | None = None,
        repo_path: str | Path | None = None,
        semantic_weight: float = 0.7,
        graph_weight: float = 0.3,
        graph_depth: int = 1,
        decay_factor: float = 0.5,
        allowed_relationship_types: list[RelationshipType | str] | None = None,
        candidate_k: int | None = None,
        normalization: str = "minmax",
        direction: str = "both",
        embedder: Any | None = None,
        batch_size: int = 16,
    ) -> GraphAwareRetriever:
        """Create and index a GraphAwareRetriever from a list of CodeChunks."""
        chunker = CodeEntityChunker(repo_path=repo_path)
        sem_retriever = SemanticRetriever(
            chunks=chunks,
            embedder=embedder,
            repo_path=repo_path,
            chunker=chunker,
            batch_size=batch_size,
        )
        return cls(
            semantic_retriever=sem_retriever,
            graph=graph,
            chunks=chunks,
            semantic_weight=semantic_weight,
            graph_weight=graph_weight,
            graph_depth=graph_depth,
            decay_factor=decay_factor,
            allowed_relationship_types=allowed_relationship_types,
            candidate_k=candidate_k,
            normalization=normalization,
            direction=direction,
            repo_path=repo_path,
            chunker=chunker,
        )

    def __len__(self) -> int:
        """Return the number of indexed chunks."""
        return len(self.semantic_retriever)
