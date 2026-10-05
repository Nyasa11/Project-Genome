"""Relationship Extractor module for extracting primitive IMPORTS, INHERITS, CALLS edges and deriving optional DEPENDS_ON edges."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from projectgenome.models.entities import Entity, EntityType, SourceLocation
from projectgenome.models.relationships import Relationship, RelationshipType
from projectgenome.parsers.python_parser import ParsedFileAST


class SymbolTable:
    """Repository-wide symbol lookup table mapping module names and entity IDs."""

    def __init__(self, file_asts: Dict[str, ParsedFileAST], entities: List[Entity]):
        self.file_asts = file_asts
        self.entities = entities
        self.entities_by_id: Dict[str, Entity] = {e.id: e for e in entities}
        
        # Mapping: module_name -> relative file path (e.g. "src.services.user" -> "src/services/user.py")
        self.module_to_file: Dict[str, str] = {}
        self.file_to_module: Dict[str, str] = {}
        for e in entities:
            if e.type == EntityType.FILE and "module_name" in e.properties:
                self.module_to_file[e.properties["module_name"]] = e.path
                self.file_to_module[e.path] = e.properties["module_name"]

        # Mapping: (file_path, symbol_name) -> Entity ID
        self.file_symbols: Dict[Tuple[str, str], str] = {}
        for e in entities:
            if e.type in (EntityType.CLASS, EntityType.FUNCTION, EntityType.METHOD):
                qual_name = e.properties.get("qualified_name", e.name)
                self.file_symbols[(e.path, qual_name)] = e.id
                # Also index bare name for simple lookup
                self.file_symbols[(e.path, e.name)] = e.id

    def resolve_import_module(
        self, source_file_path: str, module_name: str, imported_symbol: Optional[str] = None
    ) -> Optional[str]:
        """Resolves an imported module string to a repository-local File path.

        Handles:
        A. Exact module-name match
        B. src/ layout fallback ('src.' + module_name)
        C. Top-level package fallback in src/ layout
        D. Relative imports (leading dots) relative to importing file's package
        """
        if not module_name:
            return None

        # D. Relative imports
        if module_name.startswith("."):
            source_mod = self.file_to_module.get(source_file_path, "")
            is_init = Path(source_file_path).name == "__init__.py"
            if is_init:
                current_pkg = source_mod
            else:
                current_pkg = source_mod.rsplit(".", 1)[0] if "." in source_mod else ""

            dots = len(module_name) - len(module_name.lstrip("."))
            remainder = module_name[dots:]
            pkg_parts = current_pkg.split(".") if current_pkg else []
            ascend = dots - 1

            if ascend <= len(pkg_parts):
                base_parts = pkg_parts[: len(pkg_parts) - ascend] if ascend > 0 else pkg_parts

                if remainder:
                    candidate = ".".join(base_parts + remainder.split(".")) if base_parts else remainder
                    if candidate in self.module_to_file:
                        return self.module_to_file[candidate]
                else:
                    # from . import x
                    if imported_symbol:
                        sym_candidate = ".".join(base_parts + [imported_symbol]) if base_parts else imported_symbol
                        if sym_candidate in self.module_to_file:
                            return self.module_to_file[sym_candidate]
                    pkg_candidate = ".".join(base_parts)
                    if pkg_candidate in self.module_to_file:
                        return self.module_to_file[pkg_candidate]
            return None

        # A. Exact module-name match
        if module_name in self.module_to_file:
            return self.module_to_file[module_name]

        # B & C. src/ layout fallback
        src_candidate = "src." + module_name
        if src_candidate in self.module_to_file:
            return self.module_to_file[src_candidate]

        return None




class RelationshipExtractor:
    """Extracts primitive IMPORTS, INHERITS, CALLS edges across the repository."""

    def __init__(self, file_asts: Dict[str, ParsedFileAST], entities: List[Entity]):
        self.file_asts = file_asts
        self.entities = entities
        self.symbol_table = SymbolTable(file_asts, entities)

    def extract_primitive_relationships(self) -> List[Relationship]:
        relationships: List[Relationship] = []

        for rel_file_path, ast_data in self.file_asts.items():
            file_id = f"file:{rel_file_path}"

            # 1. IMPORTS
            imports_rels = self._extract_imports(rel_file_path, file_id, ast_data)
            relationships.extend(imports_rels)

            # 2. INHERITS
            inherits_rels = self._extract_inherits(rel_file_path, ast_data)
            relationships.extend(inherits_rels)

            # 3. CALLS
            calls_rels = self._extract_calls(rel_file_path, ast_data)
            relationships.extend(calls_rels)

        return relationships

    def _extract_imports(
        self, rel_file_path: str, file_id: str, ast_data: ParsedFileAST
    ) -> List[Relationship]:
        relationships: List[Relationship] = []

        for imp in ast_data.imports:
            module_name = imp.module
            target_file = self.symbol_table.resolve_import_module(
                rel_file_path, module_name, imp.symbol
            )

            if target_file:
                target_id = f"file:{target_file}"
            else:
                target_id = f"module:{module_name}"

            loc = SourceLocation(
                file=rel_file_path,
                start_line=imp.line,
                end_line=imp.line,
                start_column=imp.col,
                end_column=imp.col,
            )

            rel_id = f"rel:imports:{file_id}->{target_id}:{imp.line}"
            relationships.append(
                Relationship(
                    id=rel_id,
                    source=file_id,
                    type=RelationshipType.IMPORTS,
                    target=target_id,
                    location=loc,
                    properties={
                        "imported_symbol": imp.symbol,
                        "alias": imp.alias,
                        "is_wildcard": imp.is_wildcard,
                    },
                )
            )

        return relationships

    def _extract_inherits(
        self, rel_file_path: str, ast_data: ParsedFileAST
    ) -> List[Relationship]:
        relationships: List[Relationship] = []

        for cls_data in ast_data.classes:
            # Use the symbol table to get the correct (possibly disambiguated) ID.
            # Falls back to plain canonical form for unique symbols, matching the
            # entity extractor's output exactly.
            class_id = (
                self.symbol_table.file_symbols.get((rel_file_path, cls_data.qualified_name))
                or f"class:{rel_file_path}:{cls_data.qualified_name}"
            )

            for base in cls_data.bases:
                raw_base = base.raw_name
                target_id = self._resolve_symbol(rel_file_path, ast_data, raw_base)

                if not target_id:
                    target_id = f"external:{raw_base}"

                loc = SourceLocation(
                    file=rel_file_path,
                    start_line=base.line,
                    end_line=base.line,
                    start_column=base.col,
                    end_column=base.col,
                )

                rel_id = f"rel:inherits:{class_id}->{target_id}:{base.line}"
                relationships.append(
                    Relationship(
                        id=rel_id,
                        source=class_id,
                        type=RelationshipType.INHERITS,
                        target=target_id,
                        location=loc,
                        properties={
                            "raw_base_name": raw_base,
                            "resolved": not target_id.startswith("external:"),
                        },
                    )
                )

        return relationships

    def _extract_calls(
        self, rel_file_path: str, ast_data: ParsedFileAST
    ) -> List[Relationship]:
        relationships: List[Relationship] = []
        file_id = f"file:{rel_file_path}"

        # Collect function and method calls
        all_functions = list(ast_data.functions)
        for cls in ast_data.classes:
            all_functions.extend(cls.methods)

        for func in all_functions:
            if func.is_method:
                _prefix = "method"
            else:
                _prefix = "func"
            # Prefer the symbol-table ID so we resolve disambiguated IDs correctly.
            caller_id = (
                self.symbol_table.file_symbols.get((rel_file_path, func.qualified_name))
                or f"{_prefix}:{rel_file_path}:{func.qualified_name}"
            )

            for call in func.calls:
                rel = self._create_call_relationship(rel_file_path, caller_id, call, ast_data)
                relationships.append(rel)

        # Top level calls
        for call in ast_data.top_level_calls:
            rel = self._create_call_relationship(rel_file_path, file_id, call, ast_data)
            relationships.append(rel)

        return relationships

    def _create_call_relationship(
        self, rel_file_path: str, caller_id: str, call: Any, ast_data: ParsedFileAST
    ) -> Relationship:
        raw_call = call.raw_call
        target_id = self._resolve_symbol(rel_file_path, ast_data, raw_call)

        resolved = target_id is not None
        if not target_id:
            target_id = f"unresolved:{raw_call}"

        loc = SourceLocation(
            file=rel_file_path,
            start_line=call.line,
            end_line=call.line,
            start_column=call.col,
            end_column=call.col,
        )

        rel_id = f"rel:calls:{caller_id}->{target_id}:{call.line}:{call.col}"
        return Relationship(
            id=rel_id,
            source=caller_id,
            type=RelationshipType.CALLS,
            target=target_id,
            location=loc,
            properties={
                "raw_call": raw_call,
                "resolved": resolved,
                "confidence": "symbol_table_match" if resolved else "syntactic_only",
            },
        )

    def _resolve_symbol(
        self, current_file: str, ast_data: ParsedFileAST, symbol_expr: str
    ) -> Optional[str]:
        """Attempts deterministic static resolution of symbol expression within file or imported symbols."""
        # 1. Local symbol match
        local_id = self.symbol_table.file_symbols.get((current_file, symbol_expr))
        if local_id:
            return local_id

        # 2. Direct import match
        for imp in ast_data.imports:
            match_name = imp.alias or imp.symbol or imp.module.split(".")[-1]
            if symbol_expr == match_name or symbol_expr.startswith(match_name + "."):
                imported_file = self.symbol_table.module_to_file.get(imp.module)
                if imported_file:
                    target_symbol = imp.symbol or symbol_expr.split(".")[-1]
                    target_id = self.symbol_table.file_symbols.get((imported_file, target_symbol))
                    if target_id:
                        return target_id

        return None


def derive_depends_on_relationships(relationships: List[Relationship]) -> List[Relationship]:
    """Optional post-processing step to derive file-level DEPENDS_ON relationships from primitive edges."""
    file_dependencies: Dict[Tuple[str, str], Set[str]] = {}

    for rel in relationships:
        if rel.properties.get("derived", False):
            continue

        # Extract source and target file paths
        source_file = _extract_file_from_id(rel.source)
        target_file = _extract_file_from_id(rel.target)

        if source_file and target_file and source_file != target_file:
            pair = (source_file, target_file)
            if pair not in file_dependencies:
                file_dependencies[pair] = set()
            file_dependencies[pair].add(rel.type.value if isinstance(rel.type, RelationshipType) else str(rel.type))

    derived_rels: List[Relationship] = []
    for (src_file, tgt_file), reasons in sorted(file_dependencies.items()):
        src_id = f"file:{src_file}"
        tgt_id = f"file:{tgt_file}"
        rel_id = f"rel:depends_on:{src_id}->{tgt_id}"
        derived_rels.append(
            Relationship(
                id=rel_id,
                source=src_id,
                type=RelationshipType.DEPENDS_ON,
                target=tgt_id,
                properties={
                    "derived": True,
                    "reasons": sorted(list(reasons)),
                    "weight": len(reasons),
                },
            )
        )

    return derived_rels


def _extract_file_from_id(entity_id: str) -> Optional[str]:
    """Helper to extract relative file path from entity ID string."""
    if entity_id.startswith("file:"):
        return entity_id[5:]
    elif entity_id.startswith("class:") or entity_id.startswith("func:") or entity_id.startswith("method:"):
        parts = entity_id.split(":", 2)
        if len(parts) >= 2:
            return parts[1]
    return None
