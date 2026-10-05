"""Focused regression tests for the entity extractor duplicate-definition ID disambiguation.

Covers:
  1. Two duplicate class definitions with the same qualified name in one file
     (the canonical pytest CaptureResult pattern).
  2. Duplicate top-level function definitions in one file.
  3. Duplicate method definitions inside the same class.
  4. Unique symbols retain their plain canonical IDs (backward-compatibility).
  5. Containment relationships reference the final disambiguated IDs, not stale
     plain IDs.
  6. Deterministic reproducibility: running the extractor twice on the same
     input produces the same IDs.
"""

import os
import shutil
import sys
import tempfile
import unittest
from typing import Dict, List

# Ensure the project root is on the path regardless of invocation directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from projectgenome.extractors.entity_extractor import EntityExtractor
from projectgenome.models.entities import Entity, EntityType
from projectgenome.models.relationships import Relationship, RelationshipType
from projectgenome.parsers.python_parser import parse_python_file


REL_FILE = "src/_pytest/capture.py"


def _parse_and_extract(source: str, rel_file: str = REL_FILE):
    """Parse source and extract entities/relationships for a synthetic file."""
    parsed_ast = parse_python_file(rel_file, source)
    extractor = EntityExtractor(rel_file, parsed_ast)
    return extractor.extract_all()


class TestUniqueSymbolsBackwardCompat(unittest.TestCase):
    """Unique symbols must keep their plain canonical IDs."""

    def test_single_class_plain_id(self):
        source = "class Foo:\n    pass\n"
        entities, rels = _parse_and_extract(source)
        ids = {e.id for e in entities}
        self.assertIn(f"class:{REL_FILE}:Foo", ids)
        # No disambiguation suffix
        self.assertFalse(any("@L" in eid for eid in ids if "Foo" in eid))

    def test_single_function_plain_id(self):
        source = "def bar():\n    pass\n"
        entities, rels = _parse_and_extract(source)
        ids = {e.id for e in entities}
        self.assertIn(f"func:{REL_FILE}:bar", ids)
        self.assertFalse(any("@L" in eid for eid in ids if "bar" in eid))

    def test_single_method_plain_id(self):
        source = "class S:\n    def m(self):\n        pass\n"
        entities, rels = _parse_and_extract(source)
        ids = {e.id for e in entities}
        self.assertIn(f"class:{REL_FILE}:S", ids)
        self.assertIn(f"method:{REL_FILE}:S.m", ids)
        self.assertFalse(any("@L" in eid for eid in ids))


class TestDuplicateClassDisambiguation(unittest.TestCase):
    """The canonical pytest CaptureResult pattern: two classes with same name."""

    def setUp(self):
        # Lines 1-4: first CaptureResult, lines 6-9: second CaptureResult
        self.source = (
            "import sys\n"                        # 1
            "if sys.version_info >= (3, 11):\n"   # 2
            "    class CaptureResult:\n"           # 3
            "        x = 1\n"                     # 4
            "else:\n"                             # 5
            "    class CaptureResult:\n"           # 6
            "        x = 2\n"                     # 7
        )
        self.entities, self.rels = _parse_and_extract(self.source)
        self.ids = {e.id for e in self.entities}

    def test_two_entities_produced(self):
        capture_entities = [e for e in self.entities if e.name == "CaptureResult"]
        self.assertEqual(len(capture_entities), 2, 
                         f"Expected 2 CaptureResult entities, got {len(capture_entities)}")

    def test_no_plain_canonical_id_when_duplicate(self):
        plain_id = f"class:{REL_FILE}:CaptureResult"
        self.assertNotIn(plain_id, self.ids,
                         "Plain canonical ID must not appear when there are duplicate definitions")

    def test_both_ids_are_disambiguated(self):
        capture_ids = {e.id for e in self.entities if e.name == "CaptureResult"}
        for eid in capture_ids:
            self.assertIn("@L", eid, f"Expected @L suffix in {eid!r}")

    def test_ids_are_distinct(self):
        capture_ids = [e.id for e in self.entities if e.name == "CaptureResult"]
        self.assertEqual(len(capture_ids), len(set(capture_ids)),
                         "Duplicate definitions must have distinct IDs")

    def test_ids_anchored_to_start_line(self):
        capture_ids = {e.id for e in self.entities if e.name == "CaptureResult"}
        # The two definitions start at lines 3 and 6 in our source.
        self.assertIn(f"class:{REL_FILE}:CaptureResult@L3", capture_ids)
        self.assertIn(f"class:{REL_FILE}:CaptureResult@L6", capture_ids)

    def test_containment_rels_use_disambiguated_ids(self):
        contains_rels = [r for r in self.rels if r.type == RelationshipType.CONTAINS]
        targets = {r.target for r in contains_rels}
        # Must reference disambiguated IDs, never the stale plain ID
        plain_id = f"class:{REL_FILE}:CaptureResult"
        self.assertNotIn(plain_id, targets,
                         "Containment relationships must not point to the plain (duplicate) ID")
        # Both disambiguated IDs must be reachable
        self.assertIn(f"class:{REL_FILE}:CaptureResult@L3", targets)
        self.assertIn(f"class:{REL_FILE}:CaptureResult@L6", targets)


class TestDuplicateFunctionDisambiguation(unittest.TestCase):
    """Two top-level functions with the same name in one file."""

    def setUp(self):
        self.source = (
            "import sys\n"                        # 1
            "if sys.version_info >= (3, 11):\n"   # 2
            "    def make_result():\n"             # 3
            "        return 1\n"                  # 4
            "else:\n"                             # 5
            "    def make_result():\n"             # 6
            "        return 2\n"                  # 7
        )
        self.entities, self.rels = _parse_and_extract(self.source)
        self.ids = {e.id for e in self.entities}

    def test_two_function_entities_produced(self):
        func_entities = [e for e in self.entities if e.name == "make_result"]
        self.assertEqual(len(func_entities), 2)

    def test_no_plain_canonical_id(self):
        plain_id = f"func:{REL_FILE}:make_result"
        self.assertNotIn(plain_id, self.ids)

    def test_both_disambiguated_with_line(self):
        self.assertIn(f"func:{REL_FILE}:make_result@L3", self.ids)
        self.assertIn(f"func:{REL_FILE}:make_result@L6", self.ids)

    def test_containment_rels_use_disambiguated_ids(self):
        contains_targets = {r.target for r in self.rels if r.type == RelationshipType.CONTAINS}
        plain_id = f"func:{REL_FILE}:make_result"
        self.assertNotIn(plain_id, contains_targets)
        self.assertIn(f"func:{REL_FILE}:make_result@L3", contains_targets)
        self.assertIn(f"func:{REL_FILE}:make_result@L6", contains_targets)


class TestDuplicateMethodDisambiguation(unittest.TestCase):
    """Two methods with the same name inside the same class."""

    def setUp(self):
        self.source = (
            "import sys\n"                         # 1
            "class Wrapper:\n"                     # 2
            "    if sys.version_info >= (3, 11):\n"# 3
            "        def encode(self):\n"          # 4
            "            return b'new'\n"          # 5
            "    else:\n"                          # 6
            "        def encode(self):\n"          # 7
            "            return b'old'\n"          # 8
        )
        self.entities, self.rels = _parse_and_extract(self.source)
        self.ids = {e.id for e in self.entities}

    def test_two_method_entities_produced(self):
        method_entities = [e for e in self.entities if e.name == "encode"]
        self.assertEqual(len(method_entities), 2)

    def test_no_plain_canonical_method_id(self):
        plain_id = f"method:{REL_FILE}:Wrapper.encode"
        self.assertNotIn(plain_id, self.ids)

    def test_both_disambiguated_with_line(self):
        self.assertIn(f"method:{REL_FILE}:Wrapper.encode@L4", self.ids)
        self.assertIn(f"method:{REL_FILE}:Wrapper.encode@L7", self.ids)

    def test_containment_rels_use_disambiguated_ids(self):
        contains_targets = {r.target for r in self.rels if r.type == RelationshipType.CONTAINS}
        plain_id = f"method:{REL_FILE}:Wrapper.encode"
        self.assertNotIn(plain_id, contains_targets)
        self.assertIn(f"method:{REL_FILE}:Wrapper.encode@L4", contains_targets)
        self.assertIn(f"method:{REL_FILE}:Wrapper.encode@L7", contains_targets)


class TestMixedUniqueAndDuplicateSymbols(unittest.TestCase):
    """In a file with both unique and duplicate symbols, unique ones keep their plain IDs."""

    def setUp(self):
        self.source = (
            "import sys\n"
            "class AlwaysUnique:\n"               # unique class
            "    pass\n"
            "if sys.version_info >= (3, 11):\n"
            "    class CaptureResult:\n"           # first duplicate
            "        x = 1\n"
            "else:\n"
            "    class CaptureResult:\n"           # second duplicate
            "        x = 2\n"
            "def helper():\n"                     # unique function
            "    pass\n"
        )
        self.entities, self.rels = _parse_and_extract(self.source)
        self.ids = {e.id for e in self.entities}

    def test_unique_class_plain_id(self):
        self.assertIn(f"class:{REL_FILE}:AlwaysUnique", self.ids)
        self.assertFalse(any(
            "@L" in eid for eid in self.ids if "AlwaysUnique" in eid
        ))

    def test_unique_function_plain_id(self):
        self.assertIn(f"func:{REL_FILE}:helper", self.ids)
        self.assertFalse(any(
            "@L" in eid for eid in self.ids if "helper" in eid
        ))

    def test_duplicate_class_disambiguated(self):
        plain_id = f"class:{REL_FILE}:CaptureResult"
        self.assertNotIn(plain_id, self.ids)
        self.assertTrue(any("CaptureResult@L" in eid for eid in self.ids))


class TestDeterministicReproducibility(unittest.TestCase):
    """Running the extractor twice on identical input must yield identical IDs."""

    def test_duplicate_class_ids_are_stable(self):
        source = (
            "import sys\n"
            "if sys.version_info >= (3, 11):\n"
            "    class CaptureResult:\n"
            "        x = 1\n"
            "else:\n"
            "    class CaptureResult:\n"
            "        x = 2\n"
        )
        entities1, _ = _parse_and_extract(source)
        entities2, _ = _parse_and_extract(source)

        ids1 = sorted(e.id for e in entities1)
        ids2 = sorted(e.id for e in entities2)
        self.assertEqual(ids1, ids2, "Two runs must produce identical entity IDs")

    def test_unique_symbol_ids_are_stable(self):
        source = "class Foo:\n    def bar(self):\n        pass\n"
        entities1, _ = _parse_and_extract(source)
        entities2, _ = _parse_and_extract(source)

        ids1 = sorted(e.id for e in entities1)
        ids2 = sorted(e.id for e in entities2)
        self.assertEqual(ids1, ids2)



class TestNestedClassInsideMethod(unittest.TestCase):
    """Nested local class inside a method — mirrors the pytest failure_demo.py JSON case.

    The parser sets is_method=False for __repr__ (it is inside a function scope),
    so __repr__ ends up in parsed_ast.functions with
    scope_path=["TestCustomAssertMsg", "test_custom_repr", "JSON"].
    The parent entity "TestCustomAssertMsg.test_custom_repr.JSON" is a CLASS,
    not a func — the scope_path lookup must find it under the "class" prefix.
    """

    def setUp(self):
        self.source = (
            "class TestCustomAssertMsg:\n"          # 1
            "    def test_custom_repr(self):\n"     # 2
            "        class JSON:\n"                 # 3
            "            def __repr__(self):\n"     # 4
            "                return 'json'\n"       # 5
        )
        self.rel_file = "doc/en/example/assertion/failure_demo.py"
        from projectgenome.parsers.python_parser import parse_python_file
        parsed_ast = parse_python_file(self.rel_file, self.source)
        from projectgenome.extractors.entity_extractor import EntityExtractor
        extractor = EntityExtractor(self.rel_file, parsed_ast)
        self.entities, self.rels = extractor.extract_all()
        self.entity_ids = {e.id for e in self.entities}
        self.contains_sources = {r.source for r in self.rels if r.type == RelationshipType.CONTAINS}
        self.contains_targets = {r.target for r in self.rels if r.type == RelationshipType.CONTAINS}

    def test_json_class_entity_has_class_prefix(self):
        """Local class JSON must be emitted with 'class:' prefix."""
        expected = f"class:{self.rel_file}:TestCustomAssertMsg.test_custom_repr.JSON"
        self.assertIn(expected, self.entity_ids,
                      f"Expected class entity not found. Got: {sorted(self.entity_ids)}")

    def test_repr_function_entity_has_func_prefix(self):
        """__repr__ inside local class must be emitted with 'func:' prefix (is_method=False)."""
        expected = f"func:{self.rel_file}:TestCustomAssertMsg.test_custom_repr.JSON.__repr__"
        self.assertIn(expected, self.entity_ids,
                      f"Expected func entity not found. Got: {sorted(self.entity_ids)}")

    def test_no_spurious_func_json_entity(self):
        """Must NOT emit a 'func:...:JSON' entity — JSON is a class, not a function."""
        spurious = f"func:{self.rel_file}:TestCustomAssertMsg.test_custom_repr.JSON"
        self.assertNotIn(spurious, self.entity_ids,
                         "Spurious func: ID for the local class JSON must not appear")

    def test_repr_contains_rel_source_is_class_not_func(self):
        """The CONTAINS relationship for __repr__ must point FROM the class entity, not a nonexistent func entity."""
        class_parent_id = f"class:{self.rel_file}:TestCustomAssertMsg.test_custom_repr.JSON"
        spurious_parent_id = f"func:{self.rel_file}:TestCustomAssertMsg.test_custom_repr.JSON"
        # class entity must appear as source
        self.assertIn(class_parent_id, self.contains_sources,
                      "CONTAINS source for __repr__ must be the class entity ID")
        # spurious func entity must NOT appear as source
        self.assertNotIn(spurious_parent_id, self.contains_sources,
                         "Spurious func: parent ID must not appear as CONTAINS source")

    def test_repr_is_reachable_from_class_entity(self):
        """__repr__ target must appear in the CONTAINS targets reachable from class JSON."""
        repr_id = f"func:{self.rel_file}:TestCustomAssertMsg.test_custom_repr.JSON.__repr__"
        self.assertIn(repr_id, self.contains_targets,
                      "__repr__ must be a CONTAINS target")

    def test_all_contains_sources_exist_as_entities(self):
        """Every non-file CONTAINS relationship source must correspond to an existing entity ID.

        This is the invariant the KG builder enforces with NodeNotFoundError.
        """
        valid_sources = self.entity_ids | {f"file:{self.rel_file}"}
        for source in self.contains_sources:
            # unresolved:/external:/module: sources are not entities in the entity set
            if source.startswith(("unresolved:", "external:", "module:")):
                continue
            self.assertIn(source, valid_sources,
                          f"CONTAINS source {source!r} has no matching entity — "
                          f"KG builder would raise NodeNotFoundError")


class TestNestedFunctionInsideMethod(unittest.TestCase):
    """Nested function inside a method — parent scope_path lookup must use 'method' prefix.

    Without the fix, the lookup used ("func", "MyClass.my_method") which misses;
    _seen stores the method under ("method", "MyClass.my_method").
    The fallback would produce a non-existent 'func:...:MyClass.my_method' parent ID.
    """

    def setUp(self):
        self.source = (
            "class MyClass:\n"           # 1
            "    def my_method(self):\n" # 2
            "        def helper():\n"    # 3
            "            pass\n"         # 4
        )
        self.rel_file = "src/mymodule.py"
        from projectgenome.parsers.python_parser import parse_python_file
        parsed_ast = parse_python_file(self.rel_file, self.source)
        from projectgenome.extractors.entity_extractor import EntityExtractor
        extractor = EntityExtractor(self.rel_file, parsed_ast)
        self.entities, self.rels = extractor.extract_all()
        self.entity_ids = {e.id for e in self.entities}
        self.contains_rels = [r for r in self.rels if r.type == RelationshipType.CONTAINS]
        self.contains_sources = {r.source for r in self.contains_rels}

    def test_method_entity_has_method_prefix(self):
        expected = f"method:{self.rel_file}:MyClass.my_method"
        self.assertIn(expected, self.entity_ids)

    def test_helper_entity_has_func_prefix(self):
        expected = f"func:{self.rel_file}:MyClass.my_method.helper"
        self.assertIn(expected, self.entity_ids)

    def test_helper_contains_source_is_method_not_func(self):
        """CONTAINS for helper must come FROM the method entity, not a spurious func: entity."""
        method_id = f"method:{self.rel_file}:MyClass.my_method"
        spurious_id = f"func:{self.rel_file}:MyClass.my_method"
        helper_id = f"func:{self.rel_file}:MyClass.my_method.helper"
        # Find the rel that targets helper
        helper_rels = [r for r in self.contains_rels if r.target == helper_id]
        self.assertEqual(len(helper_rels), 1, "Expected exactly one CONTAINS rel targeting helper")
        self.assertEqual(helper_rels[0].source, method_id,
                         f"Source must be method entity. Got: {helper_rels[0].source!r}")
        self.assertNotEqual(helper_rels[0].source, spurious_id,
                            "Source must NOT be the spurious func: entity")

    def test_all_contains_sources_exist_as_entities(self):
        """Every non-file CONTAINS source must be a real entity (KG NodeNotFoundError guard)."""
        valid_sources = self.entity_ids | {f"file:{self.rel_file}"}
        for source in self.contains_sources:
            if source.startswith(("unresolved:", "external:", "module:")):
                continue
            self.assertIn(source, valid_sources,
                          f"CONTAINS source {source!r} has no matching entity")


if __name__ == "__main__":
    unittest.main(verbosity=2)
