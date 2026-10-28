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

    def test_four_targets_and_variants(self) -> None:
        self.assertEqual(len(MODULE.TARGETS), 4)
        self.assertEqual(dict(MODULE.VARIANTS)["BOTH-L2"], "query_train_l2")


if __name__ == "__main__":
    unittest.main()
