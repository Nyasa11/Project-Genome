"""Knowledge Graph Builder for ProjectGenome.

Converts normalized Repository Intelligence analyzer output (v2.0 frozen contract)
into an in-memory KnowledgeGraph (NetworkX MultiDiGraph).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .graph import (
    KnowledgeGraph,
    KnowledgeGraphError,
    NodeNotFoundError,
)
from .models import AnalyzerOutput, Entity, Relationship
from .schema import NodeType, RelationshipType


class KnowledgeGraphBuilder:
    """Builder that constructs a KnowledgeGraph from normalized analyzer output.

    Guarantees:
    - Parser-independence: operates strictly on normalized models/dicts.
    - Deterministic graph creation and ID preservation.
    - Faithful provenance and metadata preservation.
    - Synthetic placeholder generation for unresolved CALLS (e.g. unresolved:print).
    - Preservation of derived DEPENDS_ON relationships with weight and reasons.
    - Zero separate Module nodes (module_name is kept on File).
    - Canonical CONTAINS hierarchy (does not require DEFINED_IN).
    - Clear rejection of conflicting data and nonexistent entity references.
    - Explicit omission of external IMPORTS (target starts with 'module:') that reference
      stdlib/third-party packages outside the repository; these are not repository entities
      and must not appear as KG nodes or edges (v2.0 contract: IMPORTS is file:->file: only).
    """

    def __init__(self, strict_duplicates: bool = False) -> None:
        """Initialize the builder.

        Args:
            strict_duplicates: If True, raise an error if any entity ID is encountered twice,
                               even with identical attributes. If False, identical re-additions
                               are handled idempotently; conflicting data always raises.
        """
        self.strict_duplicates = strict_duplicates

    def build_from_file(self, file_path: str | Path) -> KnowledgeGraph:
        """Load and build a KnowledgeGraph from a JSON file path."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Analyzer output file not found: {file_path}")

        with open(path, encoding="utf-8") as f:
            raw_data = json.load(f)

        return self.build(raw_data)

    def build(self, data: AnalyzerOutput | dict[str, Any] | str | Path) -> KnowledgeGraph:
        """Build a KnowledgeGraph from AnalyzerOutput, dict, or file path/string."""
        if isinstance(data, (str, Path)):
            potential_path = Path(data)
            if potential_path.is_file():
                return self.build_from_file(potential_path)
            # Otherwise attempt parsing string as JSON
            raw_dict = json.loads(str(data))
            analyzer_output = AnalyzerOutput.model_validate(raw_dict)
        elif isinstance(data, dict):
            analyzer_output = AnalyzerOutput.model_validate(data)
        elif isinstance(data, AnalyzerOutput):
            analyzer_output = data
        else:
            raise TypeError(f"Unsupported analyzer data input type: {type(data)}")

        # Extract metadata
        meta_dict: dict[str, Any] = {}
        if analyzer_output.metadata:
            if hasattr(analyzer_output.metadata, "model_dump"):
                meta_dict = analyzer_output.metadata.model_dump()
            elif isinstance(analyzer_output.metadata, dict):
                meta_dict = dict(analyzer_output.metadata)

        kg = KnowledgeGraph(metadata=meta_dict)

        # Pass 1: Ingest all analyzer entities
        for entity in analyzer_output.entities:
            self._ingest_entity(kg, entity)

        # Pass 2: Ingest all relationships
        for rel in analyzer_output.relationships:
            self._ingest_relationship(kg, rel)

        return kg

    def _ingest_entity(self, kg: KnowledgeGraph, entity: Entity) -> None:
        """Ingest a single repository entity into the graph."""
        # Never create Module nodes - in v2.0 module_name is a property on File
        if entity.type == NodeType.MODULE or entity.type == NodeType.MODULE.value:
            raise KnowledgeGraphError(
                f"Invalid entity '{entity.id}': Module entities are deprecated in v2.0; "
                f"module_name must be a property on File."
            )

        location_data = entity.location.model_dump() if entity.location else None

        kg.add_node(
            node_id=entity.id,
            node_type=entity.type,
            name=entity.name,
            path=entity.path,
            location=location_data,
            properties=dict(entity.properties),
            is_synthetic=False,
            resolved=True,
            raise_if_exists=self.strict_duplicates,
        )

    def _ingest_relationship(self, kg: KnowledgeGraph, rel: Relationship) -> None:
        """Ingest a single relationship, resolving or creating synthetic placeholders as needed."""
        # --- External IMPORTS guard (v2.0 contract) ---
        # IMPORTS edges whose target starts with 'module:' reference stdlib/third-party packages
        # that are outside the repository and can never be File entities in this KG.
        # Per the v2.0 schema, IMPORTS is exclusively file:->file: (inter-repository-file).
        # Do NOT create a Module node, a synthetic node, or an UnresolvedReference for these;
        # simply omit the edge from the KG.
        if (
            rel.type == RelationshipType.IMPORTS
            or rel.type == RelationshipType.IMPORTS.value
        ) and rel.target.startswith("module:"):
            return

        # Check source node
        if not kg.has_node(rel.source):
            raise NodeNotFoundError(
                f"Cannot create relationship '{rel.id or rel.type}': "
                f"Source entity '{rel.source}' does not exist in graph."
            )

        # Check target node
        if not kg.has_node(rel.target):
            if rel.is_unresolved:
                # Target is an unresolved call reference: create synthetic placeholder node
                self._create_unresolved_placeholder(kg, rel)
            else:
                # Target is an unresolved repository entity reference
                raise NodeNotFoundError(
                    f"Cannot create relationship '{rel.id or rel.type}': "
                    f"Target entity '{rel.target}' does not exist in graph."
                )

        location_data = rel.location.model_dump() if rel.location else None
        rel_props = dict(rel.properties)

        kg.add_relationship(
            source_id=rel.source,
            target_id=rel.target,
            rel_type=rel.type,
            key=rel.id,
            id=rel.id,
            location=location_data,
            properties=rel_props,
            is_derived=rel.is_derived,
            is_primitive=rel.is_primitive,
            is_unresolved=rel.is_unresolved,
        )

    def _create_unresolved_placeholder(self, kg: KnowledgeGraph, rel: Relationship) -> None:
        """Create a clearly-marked KG-internal placeholder node for an unresolved call target."""
        raw_call = rel.properties.get("raw_call") or rel.target.removeprefix("unresolved:")
        node_id = rel.target

        kg.add_node(
            node_id=node_id,
            node_type=NodeType.UNRESOLVED_REFERENCE,
            name=raw_call,
            is_synthetic=True,
            resolved=False,
            raw_call=raw_call,
            properties=dict(rel.properties),
            raise_if_exists=False,  # Unresolved targets can be called multiple times
        )


def build_from_analysis(data: AnalyzerOutput | dict[str, Any] | str | Path) -> KnowledgeGraph:
    """Convenience function to build a KnowledgeGraph from analyzer output."""
    return KnowledgeGraphBuilder().build(data)
