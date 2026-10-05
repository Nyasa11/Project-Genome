"""Unit tests for knowledge_graph.builder (Milestone 3)."""

from pathlib import Path
import pytest

from src.knowledge_graph.builder import KnowledgeGraphBuilder, build_from_analysis
from src.knowledge_graph.graph import KnowledgeGraphError, NodeNotFoundError
from src.knowledge_graph.models import Entity, Relationship
from src.knowledge_graph.schema import NodeType, RelationshipType


def test_build_from_sample_analysis_fixture():
    """Verify builder accurately constructs KnowledgeGraph from Nyasa's sample fixture."""
    fixture_path = Path("data/sample/sample_analysis.json")
    kg = build_from_analysis(fixture_path)

    # Repository entities: exactly 19
    repo_nodes = [
        nid for nid, data in kg.get_all_nodes().items()
        if not data.get("is_synthetic", False)
    ]
    assert len(repo_nodes) == 19

    # Synthetic unresolved placeholder nodes: exactly 7
    synthetic_nodes = [
        nid for nid, data in kg.get_all_nodes().items()
        if data.get("is_synthetic", False)
    ]
    assert len(synthetic_nodes) == 7

    # Total nodes in graph = 26
    assert kg.number_of_nodes() == 26

    # Total edges = 35
    assert kg.number_of_edges() == 35

    # Verify no Module nodes were created
    module_nodes = [
        nid for nid, data in kg.get_all_nodes().items()
        if data.get("type") == "Module"
    ]
    assert len(module_nodes) == 0

    # Verify module_name is kept as a property on File entities
    user_file = kg.get_node("file:models/user.py")
    assert user_file is not None
    assert user_file["properties"]["module_name"] == "models.user"

    # Verify canonical CONTAINS hierarchy exists
    assert kg.has_relationship("dir:models", "file:models/user.py", RelationshipType.CONTAINS)
    assert kg.has_relationship("file:models/user.py", "class:models/user.py:User", RelationshipType.CONTAINS)
    assert kg.has_relationship("class:models/user.py:User", "method:models/user.py:User.__init__", RelationshipType.CONTAINS)

    # Verify no DEFINED_IN relationships exist
    assert not any(r["type"] == "DEFINED_IN" for r in kg.get_all_relationships())

    # Verify derived DEPENDS_ON relationship properties
    depends_rel = kg.get_relationship(
        "file:main.py",
        "file:services/user_service.py",
        RelationshipType.DEPENDS_ON,
    )
    assert depends_rel is not None
    assert depends_rel["properties"]["derived"] is True
    assert depends_rel["properties"]["reasons"] == ["CALLS", "IMPORTS"]
    assert depends_rel["properties"]["weight"] == 2

    # Verify unresolved CALLS target node
    unresolved_node = kg.get_node("unresolved:print")
    assert unresolved_node is not None
    assert unresolved_node["type"] == NodeType.UNRESOLVED_REFERENCE.value
    assert unresolved_node["is_synthetic"] is True
    assert unresolved_node["resolved"] is False
    assert unresolved_node["raw_call"] == "print"


def test_builder_rejects_module_entity():
    """Verify builder raises KnowledgeGraphError if an analyzer output contains a Module entity."""
    builder = KnowledgeGraphBuilder()
    malformed_data = {
        "version": "1.0.0",
        "entities": [
            {
                "id": "module:user",
                "type": "Module",
                "name": "user",
            }
        ],
        "relationships": [],
    }
    with pytest.raises(KnowledgeGraphError, match="Module entities are deprecated in v2.0"):
        builder.build(malformed_data)


def test_builder_rejects_missing_source():
    """Verify builder raises NodeNotFoundError when a relationship references a missing source."""
    builder = KnowledgeGraphBuilder()
    data = {
        "version": "1.0.0",
        "entities": [
            {"id": "file:a.py", "type": "File", "name": "a.py"}
        ],
        "relationships": [
            {"source": "missing:source", "type": "IMPORTS", "target": "file:a.py"}
        ],
    }
    with pytest.raises(NodeNotFoundError, match="Source entity 'missing:source' does not exist"):
        builder.build(data)


def test_builder_rejects_missing_non_unresolved_target():
    """Verify builder raises NodeNotFoundError when a non-unresolved target entity is missing."""
    builder = KnowledgeGraphBuilder()
    data = {
        "version": "1.0.0",
        "entities": [
            {"id": "file:a.py", "type": "File", "name": "a.py"}
        ],
        "relationships": [
            {"source": "file:a.py", "type": "IMPORTS", "target": "file:missing.py"}
        ],
    }
    with pytest.raises(NodeNotFoundError, match="Target entity 'file:missing.py' does not exist"):
        builder.build(data)


def test_builder_skips_external_module_imports():
    """Verify builder ingests repo-local imports, skips external module: imports, creates no module node, and produces a valid KG."""
    from src.knowledge_graph.validation import validate_or_raise

    builder = KnowledgeGraphBuilder()
    data = {
        "version": "1.0.0",
        "entities": [
            {"id": "file:a.py", "type": "File", "name": "a.py"},
            {"id": "file:b.py", "type": "File", "name": "b.py"},
        ],
        "relationships": [
            {
                "id": "rel:imports:local",
                "source": "file:a.py",
                "type": "IMPORTS",
                "target": "file:b.py",
            },
            {
                "id": "rel:imports:external",
                "source": "file:a.py",
                "type": "IMPORTS",
                "target": "module:os",
            },
        ],
    }
    kg = builder.build(data)

    # 1. Repository-local file:a.py -> IMPORTS -> file:b.py is ingested
    assert kg.has_relationship("file:a.py", "file:b.py", RelationshipType.IMPORTS)

    # 2. External file:a.py -> IMPORTS -> module:os was skipped without NodeNotFoundError
    rels = kg.get_all_relationships()
    assert not any(r.get("target") == "module:os" for r in rels)

    # 3. module:os does NOT become a graph node
    assert not kg.has_node("module:os")
    assert "module:os" not in kg.get_all_nodes()

    # 4. The resulting KG remains valid
    validate_or_raise(kg)
