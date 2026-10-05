"""Entity Extractor module for converting parsed AST structures into canonical IR Entity objects and structural CONTAINS edges."""

from typing import Dict, List, Optional, Tuple, Union

from projectgenome.models.entities import Entity, EntityType, SourceLocation
from projectgenome.models.relationships import Relationship, RelationshipType
from projectgenome.parsers.python_parser import ParsedClass, ParsedFileAST, ParsedFunction


def _canonical_id(prefix: str, rel_file_path: str, qualified_name: str) -> str:
    """Return the plain canonical entity ID (no source-location suffix)."""
    return f"{prefix}:{rel_file_path}:{qualified_name}"


def _disambiguated_id(prefix: str, rel_file_path: str, qualified_name: str, start_line: int) -> str:
    """Return a source-location-disambiguated entity ID.

    Format: ``<prefix>:<rel_file_path>:<qualified_name>@L<start_line>``

    Only used when two legitimate definitions share the same qualified name
    within the same file (e.g. conditional ``if sys.version_info`` branches).
    The start_line is stable across runs for a given source file, making these
    IDs fully deterministic.
    """
    return f"{prefix}:{rel_file_path}:{qualified_name}@L{start_line}"


def _patch_relationships(relationships: List[Relationship], old_id: str, new_id: str) -> None:
    """Update source/target/id fields of all relationships referencing ``old_id``."""
    for rel in relationships:
        changed = False
        if rel.source == old_id:
            rel.source = new_id
            changed = True
        if rel.target == old_id:
            rel.target = new_id
            changed = True
        if changed:
            rel.id = rel.id.replace(old_id, new_id)


class EntityExtractor:
    """Transforms ParsedFileAST into canonical Entity objects and top-down CONTAINS edges.

    Duplicate-definition handling
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    In Python it is legal to define the same class/function/method name more
    than once in a single file (e.g. inside ``if sys.version_info >= (3, 11)``
    branches).  The AST parser faithfully extracts all of them.  Without
    special handling both definitions would receive the same deterministic ID,
    causing the KnowledgeGraph builder to raise a strict-duplicate error.

    The extractor resolves this by tracking which ``(prefix, qualified_name)``
    keys have already been emitted for the current file.  When a duplicate is
    encountered:

    1. The **first** occurrence is retroactively renamed from its plain
       canonical form to a source-location-suffixed form
       ``...<qualified_name>@L<start_line>``.
    2. The **new** occurrence also receives a source-location-suffixed form.
    3. Every containment relationship that already referenced the old plain ID
       is updated in-place to the new suffixed ID.

    Unique symbols are completely unaffected — their IDs remain the plain
    canonical format, preserving full backward compatibility.
    """

    def __init__(self, rel_file_path: str, parsed_ast: ParsedFileAST):
        self.rel_file_path = rel_file_path
        self.parsed_ast = parsed_ast
        self.file_id = f"file:{self.rel_file_path}"
        # Maps (prefix, qualified_name) -> current assigned ID string
        # for duplicate detection.  Reset at extract_all() time.
        self._seen: Dict[Tuple[str, str], str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_all(self) -> Tuple[List[Entity], List[Relationship]]:
        """Extracts all entities and containment edges from the parsed AST."""
        self._seen = {}
        self._all_entities: List[Entity] = []  # accumulates ALL entities (classes + methods + funcs)
        self._all_relationships: List[Relationship] = []

        # 1. Process top-level and nested classes
        for parsed_class in self.parsed_ast.classes:
            self._extract_class(parsed_class)

        # 2. Process top-level standalone functions
        for parsed_func in self.parsed_ast.functions:
            self._extract_function(parsed_func)

        return self._all_entities, self._all_relationships

    # Kept for backward compatibility with any callers of the old extract()
    def extract(self) -> Tuple[List[Entity], List[Relationship]]:
        return self.extract_all()

    # ------------------------------------------------------------------
    # Duplicate-aware ID assignment
    # ------------------------------------------------------------------

    def _assign_id(self, prefix: str, qualified_name: str, start_line: int) -> str:
        """Return the correct entity ID, applying disambiguation when needed.

        If this is the first time ``(prefix, qualified_name)`` is seen for the
        current file, the plain canonical ID is returned and recorded.

        If the key was seen before, the first occurrence is retroactively
        renamed to its source-location-suffixed form (patching the already-
        emitted Entity object and all relationships), and a new suffixed ID is
        returned for the current occurrence.

        Args:
            prefix: ``"class"``, ``"func"``, or ``"method"``.
            qualified_name: The fully-qualified name within the file.
            start_line: The 1-based start line of the current definition.

        Returns:
            The final, stable entity ID for this symbol.
        """
        key = (prefix, qualified_name)
        plain_id = _canonical_id(prefix, self.rel_file_path, qualified_name)

        if key not in self._seen:
            self._seen[key] = plain_id
            return plain_id

        # Duplicate detected.
        first_id = self._seen[key]
        # If the first occurrence still has the plain ID, disambiguate it now.
        if first_id == plain_id:
            first_entity = self._find_entity_by_id(plain_id)
            if first_entity is not None:
                first_disambig_id = _disambiguated_id(
                    prefix, self.rel_file_path, qualified_name,
                    first_entity.location.start_line,
                )
                old_id = first_entity.id
                first_entity.id = first_disambig_id
                _patch_relationships(self._all_relationships, old_id, first_disambig_id)
                self._seen[key] = first_disambig_id

        # Return source-location-suffixed ID for THIS (duplicate) occurrence.
        return _disambiguated_id(prefix, self.rel_file_path, qualified_name, start_line)

    def _find_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """Search the in-progress entity list for an entity by ID."""
        for e in self._all_entities:
            if e.id == entity_id:
                return e
        return None

    # ------------------------------------------------------------------
    # Per-symbol extraction (all results appended to self._all_*)
    # ------------------------------------------------------------------

    def _extract_class(self, parsed_class: ParsedClass) -> None:
        class_id = self._assign_id(
            "class", parsed_class.qualified_name, parsed_class.start_line
        )

        location = SourceLocation(
            file=self.rel_file_path,
            start_line=parsed_class.start_line,
            end_line=parsed_class.end_line,
            start_column=parsed_class.start_column,
            end_column=parsed_class.end_column,
        )

        class_entity = Entity(
            id=class_id,
            type=EntityType.CLASS,
            name=parsed_class.name,
            path=self.rel_file_path,
            location=location,
            properties={
                "qualified_name": parsed_class.qualified_name,
                "docstring": parsed_class.docstring,
                "decorators": parsed_class.decorators,
                "bases": [b.raw_name for b in parsed_class.bases],
            },
        )
        self._all_entities.append(class_entity)

        rel_id = f"rel:contains:{self.file_id}->{class_id}"
        self._all_relationships.append(
            Relationship(
                id=rel_id,
                source=self.file_id,
                type=RelationshipType.CONTAINS,
                target=class_id,
                properties={"confidence": "certain"},
            )
        )

        # Class CONTAINS Methods
        for method in parsed_class.methods:
            self._extract_method(class_id, method)

    def _extract_method(self, parent_class_id: str, method: ParsedFunction) -> None:
        method_id = self._assign_id(
            "method", method.qualified_name, method.start_line
        )

        location = SourceLocation(
            file=self.rel_file_path,
            start_line=method.start_line,
            end_line=method.end_line,
            start_column=method.start_column,
            end_column=method.end_column,
        )

        method_entity = Entity(
            id=method_id,
            type=EntityType.METHOD,
            name=method.name,
            path=self.rel_file_path,
            location=location,
            properties={
                "qualified_name": method.qualified_name,
                "docstring": method.docstring,
                "decorators": method.decorators,
                "is_async": method.is_async,
                "parameters": method.parameters,
                "return_type": method.return_type,
            },
        )
        self._all_entities.append(method_entity)

        rel_id = f"rel:contains:{parent_class_id}->{method_id}"
        self._all_relationships.append(
            Relationship(
                id=rel_id,
                source=parent_class_id,
                type=RelationshipType.CONTAINS,
                target=method_id,
                properties={"confidence": "certain"},
            )
        )

    def _extract_function(self, func: ParsedFunction) -> None:
        func_id = self._assign_id(
            "func", func.qualified_name, func.start_line
        )

        location = SourceLocation(
            file=self.rel_file_path,
            start_line=func.start_line,
            end_line=func.end_line,
            start_column=func.start_column,
            end_column=func.end_column,
        )

        func_entity = Entity(
            id=func_id,
            type=EntityType.FUNCTION,
            name=func.name,
            path=self.rel_file_path,
            location=location,
            properties={
                "qualified_name": func.qualified_name,
                "docstring": func.docstring,
                "decorators": func.decorators,
                "is_async": func.is_async,
                "parameters": func.parameters,
                "return_type": func.return_type,
            },
        )
        self._all_entities.append(func_entity)

        # Determine parent for containment (File or outer scope entity).
        # The immediately enclosing named scope can be a class, a method, or a
        # function — try all three prefixes in _seen so we always resolve to the
        # correct, already-emitted entity ID regardless of what the parent is.
        if func.scope_path:
            parent_qual = ".".join(func.scope_path)
            parent_id = None
            for _prefix in ("class", "method", "func"):
                parent_id = self._seen.get((_prefix, parent_qual))
                if parent_id is not None:
                    break
            if parent_id is None:
                # No matching parent in _seen — fall back to a plain func: ID.
                # This only fires if the parser emits an unknown scope structure.
                parent_id = _canonical_id("func", self.rel_file_path, parent_qual)
        else:
            parent_id = self.file_id

        rel_id = f"rel:contains:{parent_id}->{func_id}"
        self._all_relationships.append(
            Relationship(
                id=rel_id,
                source=parent_id,
                type=RelationshipType.CONTAINS,
                target=func_id,
                properties={"confidence": "certain"},
            )
        )
