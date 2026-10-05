"""Focused unit tests for repository-local and external import resolution."""

import os
import shutil
import tempfile
import unittest

from projectgenome.main import analyze_repository
from projectgenome.models.relationships import RelationshipType


class TestImportResolution(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _write_file(self, rel_path: str, content: str):
        full_path = os.path.join(self.test_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

    def test_1_exact_module_name_import(self):
        """1. Exact module-name repository import resolves to file:<path>."""
        self._write_file("auth.py", "def verify():\n    pass\n")
        self._write_file("service.py", "import auth\n")
        output = analyze_repository(self.test_dir)

        import_rels = [r for r in output.relationships if r.type == RelationshipType.IMPORTS]
        self.assertEqual(len(import_rels), 1)
        rel = import_rels[0]
        self.assertEqual(rel.source, "file:service.py")
        self.assertEqual(rel.target, "file:auth.py")
        self.assertIsNone(rel.properties["imported_symbol"])
        self.assertFalse(rel.properties["is_wildcard"])

    def test_2_src_layout_import(self):
        """2. src-layout import (_pytest.config -> file:src/_pytest/config/__init__.py)."""
        self._write_file("src/_pytest/config/__init__.py", "# config pkg\n")
        self._write_file("src/_pytest/terminal.py", "import _pytest.config\n")
        output = analyze_repository(self.test_dir)

        import_rels = [r for r in output.relationships if r.type == RelationshipType.IMPORTS]
        self.assertEqual(len(import_rels), 1)
        rel = import_rels[0]
        self.assertEqual(rel.source, "file:src/_pytest/terminal.py")
        self.assertEqual(rel.target, "file:src/_pytest/config/__init__.py")

    def test_3_top_level_package_import(self):
        """3. Top-level package import (pytest -> file:src/pytest/__init__.py)."""
        self._write_file("src/pytest/__init__.py", "# pytest pkg\n")
        self._write_file("bench/bench.py", "import pytest\n")
        output = analyze_repository(self.test_dir)

        import_rels = [r for r in output.relationships if r.type == RelationshipType.IMPORTS]
        self.assertEqual(len(import_rels), 1)
        rel = import_rels[0]
        self.assertEqual(rel.source, "file:bench/bench.py")
        self.assertEqual(rel.target, "file:src/pytest/__init__.py")

    def test_4_relative_single_dot_import(self):
        """4. Relative single-dot import (.argparsing from src/_pytest/config/__init__.py)."""
        self._write_file("src/_pytest/config/__init__.py", "from .argparsing import Parser\n")
        self._write_file("src/_pytest/config/argparsing.py", "class Parser:\n    pass\n")
        output = analyze_repository(self.test_dir)

        import_rels = [r for r in output.relationships if r.type == RelationshipType.IMPORTS]
        self.assertEqual(len(import_rels), 1)
        rel = import_rels[0]
        self.assertEqual(rel.source, "file:src/_pytest/config/__init__.py")
        self.assertEqual(rel.target, "file:src/_pytest/config/argparsing.py")
        self.assertEqual(rel.properties["imported_symbol"], "Parser")

    def test_5_relative_multi_dot_import(self):
        """5. Relative multi-dot import (..compat from src/_pytest/_io/terminalwriter.py)."""
        self._write_file("src/_pytest/compat.py", "# compat\n")
        self._write_file("src/_pytest/_io/terminalwriter.py", "from ..compat import assert_never\n")
        output = analyze_repository(self.test_dir)

        import_rels = [r for r in output.relationships if r.type == RelationshipType.IMPORTS]
        self.assertEqual(len(import_rels), 1)
        rel = import_rels[0]
        self.assertEqual(rel.source, "file:src/_pytest/_io/terminalwriter.py")
        self.assertEqual(rel.target, "file:src/_pytest/compat.py")
        self.assertEqual(rel.properties["imported_symbol"], "assert_never")

    def test_6_genuine_stdlib_external_import(self):
        """6. Genuine stdlib/external import (os remains module:os)."""
        self._write_file("main.py", "import os\nfrom sys import argv\n")
        output = analyze_repository(self.test_dir)

        import_rels = [r for r in output.relationships if r.type == RelationshipType.IMPORTS]
        targets = {r.target for r in import_rels}
        self.assertIn("module:os", targets)
        self.assertIn("module:sys", targets)
        for r in import_rels:
            self.assertTrue(r.target.startswith("module:"))

    def test_7_third_party_import(self):
        """7. Third-party import (pluggy remains module:pluggy)."""
        self._write_file("plugin.py", "import pluggy\n")
        output = analyze_repository(self.test_dir)

        import_rels = [r for r in output.relationships if r.type == RelationshipType.IMPORTS]
        self.assertEqual(len(import_rels), 1)
        self.assertEqual(import_rels[0].target, "module:pluggy")


if __name__ == "__main__":
    unittest.main()
