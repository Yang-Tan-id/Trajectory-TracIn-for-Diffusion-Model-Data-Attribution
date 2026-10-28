from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
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
        self.assertEqual(MODULE.method_configs("shape3d")[0].raw_sign, "p1")
        self.assertEqual(MODULE.method_configs("shape3d")[2].raw_sign, "m1")
        self.assertEqual(MODULE.method_configs("celeba")[0].raw_sign, "m1")
        self.assertEqual(MODULE.method_configs("celeba")[0].adamw_sign, "p1")

    def test_legacy_truncated_query_tag_is_resolved(self) -> None:
        config = MODULE.CONFIGS["celeba"]
        prompt = "Attractive,Black_Hair,High_Cheekbones,Male,Mouth_Slightly_Open"
        full_tag = MODULE.prompt_tag(prompt, config)
        legacy_tag = full_tag[:40]
        with tempfile.TemporaryDirectory() as temporary:
            eval_root = Path(temporary)
            expected = eval_root / f"query_{legacy_tag}" / "initial_seed_0" / "lds"
            expected.mkdir(parents=True)
            actual = MODULE.resolve_query_lds_root(
                eval_root=eval_root,
                prompt=prompt,
                seed=0,
                config=config,
            )
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
