"""Main CLI entrypoint for ProjectGenome Repository Analysis Module."""

import argparse
from datetime import datetime, timezone
import os
import sys
from typing import Dict

from projectgenome.exporters.json_exporter import JSONExporter
from projectgenome.extractors.entity_extractor import EntityExtractor
from projectgenome.extractors.relationship_extractor import (
    RelationshipExtractor,
    derive_depends_on_relationships,
)
from projectgenome.models.schema import RepositoryAnalysisOutput
from projectgenome.parsers.python_parser import ParsedFileAST, parse_python_file
from projectgenome.scanner.repository_scanner import RepositoryScanner


def analyze_repository(
    repo_dir: str, include_derived: bool = True
) -> RepositoryAnalysisOutput:
    """Analyzes a repository directory and returns normalized IR output."""
    abs_repo_dir = os.path.abspath(os.path.expanduser(repo_dir))
    scanner = RepositoryScanner(abs_repo_dir)

    # 1. Scan filesystem layout
    layout_entities, layout_contains = scanner.scan()

    # 2. Parse AST for all Python files
    file_asts: Dict[str, ParsedFileAST] = {}
    code_entities = []
    code_contains = []

    for entity in layout_entities:
        if entity.type.value == "File" and entity.name.endswith(".py"):
            rel_path = entity.path
            abs_path = os.path.join(abs_repo_dir, rel_path)
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    content = f.read()
                ast_data = parse_python_file(rel_path, content)
            except Exception:
                ast_data = ParsedFileAST(file_path=rel_path)

            file_asts[rel_path] = ast_data

            # 3. Extract code entities and CONTAINS edges
            extractor = EntityExtractor(rel_path, ast_data)
            entities, contains = extractor.extract_all()
            code_entities.extend(entities)
            code_contains.extend(contains)

    # Combine all entities
    all_entities = layout_entities + code_entities

    # 4. Extract primitive relationships (IMPORTS, INHERITS, CALLS)
    rel_extractor = RelationshipExtractor(file_asts, all_entities)
    primitive_rels = rel_extractor.extract_primitive_relationships()

    all_relationships = layout_contains + code_contains + primitive_rels

    # 5. Optional derived DEPENDS_ON layer
    if include_derived:
        derived_rels = derive_depends_on_relationships(all_relationships)
        all_relationships.extend(derived_rels)

    timestamp = datetime.now(timezone.utc).isoformat()

    return RepositoryAnalysisOutput(
        repository_identity=scanner.repo_id,
        root_path=abs_repo_dir,
        extracted_at=timestamp,
        entities=all_entities,
        relationships=all_relationships,
    )


def main():
    parser = argparse.ArgumentParser(
        description="ProjectGenome — Repository Intelligence / Analysis Module"
    )
    parser.add_argument(
        "--repo-dir",
        default=".",
        help="Path to repository root directory to analyze (default: current directory)",
    )
    parser.add_argument(
        "--output",
        default="analysis_output.json",
        help="Path to output JSON file (default: analysis_output.json)",
    )
    parser.add_argument(
        "--no-derived",
        action="store_false",
        dest="include_derived",
        help="Exclude derived DEPENDS_ON relationships",
    )
    parser.add_argument(
        "--canonical-only",
        action="store_true",
        help="Exclude run metadata (extracted_at timestamp) from output",
    )

    args = parser.parse_args()

    print(f"[*] Analyzing repository at: {os.path.abspath(args.repo_dir)}")
    output = analyze_repository(args.repo_dir, include_derived=args.include_derived)

    exporter = JSONExporter(output)
    out_file = exporter.export(os.path.expanduser(args.output), canonical_only=args.canonical_only)

    canonical_hash = output.compute_canonical_hash()
    print(f"[+] Repository analysis complete.")
    print(f"    Total Entities: {len(output.entities)}")
    print(f"    Total Relationships: {len(output.relationships)}")
    print(f"    Canonical Hash (SHA-256): {canonical_hash}")
    print(f"[+] Results exported to: {out_file}")


if __name__ == "__main__":
    main()
