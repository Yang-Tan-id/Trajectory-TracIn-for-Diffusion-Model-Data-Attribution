from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "script"
    / "export_pint_lds_txt.py"
)
SPEC = importlib.util.spec_from_file_location("export_pint_lds_txt", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ExportPintLdsTxtTest(unittest.TestCase):
    def test_query_ranges(self) -> None:
        self.assertEqual(MODULE.parse_query_ids("0-2,4"), [0, 1, 2, 4])

    def test_prompt_tags_match_dataset_rules(self) -> None:
        celeba = MODULE.CONFIGS["celeba"]
        shape = MODULE.CONFIGS["shape3d"]
        self.assertEqual(MODULE.prompt_tag("A,B C", celeba), "A_B_C")
        self.assertEqual(MODULE.prompt_tag("A,B C", shape), "A_B_C")

    def test_four_targets_and_five_methods(self) -> None:
        self.assertEqual(len(MODULE.TARGETS), 4)
        methods = MODULE.method_configs("shape3d")
        self.assertEqual(len(methods), 5)
        self.assertEqual(methods[1].label, "PINT / BOTH-L2")
        self.assertEqual(methods[3].label, "TRACIN / BOTH-L2")
        self.assertEqual(methods[4].label, "RETRAC / BOTH-L2")

    def test_dataset_specific_pint_orientation(self) -> None:
        self.assertEqual(MODULE.method_configs("shape3d")[0].raw_multiplier, -1.0)
        self.assertEqual(MODULE.method_configs("celeba")[0].raw_multiplier, 1.0)


if __name__ == "__main__":
    unittest.main()
