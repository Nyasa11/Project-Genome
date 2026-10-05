"""Repository Scanner module for scanning directory trees and collecting physical layout entities."""

import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

from projectgenome.models.entities import Entity, EntityType, SourceLocation
from projectgenome.models.relationships import Relationship, RelationshipType


DEFAULT_IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
    ".egg-info",
    ".idea",
    ".vscode",
    ".mypy_cache",
    ".ruff_cache",
}


def normalize_rel_path(path: str) -> str:
    """Normalize file path to use forward slashes for deterministic cross-platform IDs."""
    clean = path.replace("\\", "/").strip("/")
    return clean if clean else "."


def derive_module_name(rel_file_path: str) -> str:
    """Derive Python logical module import name from relative file path.
    
    Example:
      src/services/user.py -> src.services.user
      src/services/__init__.py -> src.services
    """
    path_obj = Path(normalize_rel_path(rel_file_path))
    parts = list(path_obj.parts)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    
    if parts[-1] == "__init__":
        parts.pop()

    return ".".join(parts)


class RepositoryScanner:
    """Scans a filesystem repository and extracts Repository, Directory, File entities and CONTAINS edges."""

    def __init__(self, root_dir: str, ignore_dirs: Set[str] = None):
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        self.ignore_dirs = ignore_dirs if ignore_dirs is not None else DEFAULT_IGNORE_DIRS
        self.repo_name = os.path.basename(self.root_dir)
        self.repo_id = f"repo:{self.repo_name}"

    def scan(self) -> Tuple[List[Entity], List[Relationship]]:
        """Walks the directory structure and generates entities and canonical CONTAINS relationships."""
        entities: List[Entity] = []
        relationships: List[Relationship] = []

        # 1. Repository entity
        repo_entity = Entity(
            id=self.repo_id,
            type=EntityType.REPOSITORY,
            name=self.repo_name,
            path=".",
            properties={
                "language": "python",
                "abs_path": self.root_dir,
            },
        )
        entities.append(repo_entity)

        # Track directory IDs to connect parents to children
        dir_entities_map: Dict[str, Entity] = {}

        for root, dirs, files in os.walk(self.root_dir):
            # Prune ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]

            rel_root = os.path.relpath(root, self.root_dir)
            norm_rel_root = normalize_rel_path(rel_root)

            # Directory entity
            if norm_rel_root != ".":
                dir_id = f"dir:{norm_rel_root}"
                dir_name = os.path.basename(root)
                dir_entity = Entity(
                    id=dir_id,
                    type=EntityType.DIRECTORY,
                    name=dir_name,
                    path=norm_rel_root,
                )
                entities.append(dir_entity)
                dir_entities_map[norm_rel_root] = dir_entity

                # Parent containment edge
                parent_rel = normalize_rel_path(os.path.relpath(os.path.dirname(root), self.root_dir))
                if parent_rel == ".":
                    parent_id = self.repo_id
                else:
                    parent_id = f"dir:{parent_rel}"

                rel_id = f"rel:contains:{parent_id}->{dir_id}"
                relationships.append(
                    Relationship(
                        id=rel_id,
                        source=parent_id,
                        type=RelationshipType.CONTAINS,
                        target=dir_id,
                        properties={"confidence": "certain"},
                    )
                )

            # Files
            for file_name in files:
                if not file_name.endswith(".py"):
                    continue

                abs_file_path = os.path.join(root, file_name)
                rel_file_path = normalize_rel_path(os.path.relpath(abs_file_path, self.root_dir))
                file_id = f"file:{rel_file_path}"
                module_name = derive_module_name(rel_file_path)

                # Calculate line count and file size
                file_size = os.path.getsize(abs_file_path)
                try:
                    with open(abs_file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        line_count = len(lines)
                except Exception:
                    line_count = 0

                location = SourceLocation(
                    file=rel_file_path,
                    start_line=1,
                    end_line=line_count if line_count > 0 else 1,
                    start_column=0,
                    end_column=0,
                )

                file_entity = Entity(
                    id=file_id,
                    type=EntityType.FILE,
                    name=file_name,
                    path=rel_file_path,
                    location=location,
                    properties={
                        "module_name": module_name,
                        "extension": ".py",
                        "size_bytes": file_size,
                        "line_count": line_count,
                    },
                )
                entities.append(file_entity)

                # Containment edge from directory or repository
                if norm_rel_root == ".":
                    parent_id = self.repo_id
                else:
                    parent_id = f"dir:{norm_rel_root}"

                rel_id = f"rel:contains:{parent_id}->{file_id}"
                relationships.append(
                    Relationship(
                        id=rel_id,
                        source=parent_id,
                        type=RelationshipType.CONTAINS,
                        target=file_id,
                        properties={"confidence": "certain"},
                    )
                )

        return entities, relationships
