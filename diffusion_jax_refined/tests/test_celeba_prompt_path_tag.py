from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
CELEBA_ROOT = REPO_ROOT / "diffusion_jax_refined" / "celeba"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_standalone_prompt_tag(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "PROMPT_PATH_TAG_MAX_LENGTH"
            for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "prompt_tag":
            selected.append(node)
    if len(selected) != 2:
        raise AssertionError(f"expected tag constant and function in {path}")
    namespace = {"re": re}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["PROMPT_PATH_TAG_MAX_LENGTH"], namespace["prompt_tag"]


class CelebaPromptPathTagTest(unittest.TestCase):
    def test_config_uses_150_character_tags(self) -> None:
        config = load_module(
            "celeba_dataset_config_attr40_for_tag_test",
            CELEBA_ROOT / "dataset_config_attr40.py",
        )
        self.assertEqual(config.PROMPT_PATH_TAG_MAX_LENGTH, 150)
        self.assertEqual(len(config._prompt_path_tag("a" * 200)), 150)
        self.assertEqual(
            tuple(map(len, config._prompt_path_tag_candidates("a" * 200))),
            (150, 80),
        )

    def test_prompts_that_only_differ_after_80_characters_do_not_collide(self) -> None:
        config = load_module(
            "celeba_dataset_config_attr40_for_collision_test",
            CELEBA_ROOT / "dataset_config_attr40.py",
        )
        shared_prefix = "Attractive," + "Shared_Attribute," * 8
        first = config._prompt_path_tag(shared_prefix + "Oval_Face")
        second = config._prompt_path_tag(shared_prefix + "Wearing_Necklace")
        self.assertEqual(first[:80], second[:80])
        self.assertNotEqual(first, second)

    def test_standalone_celeba_helpers_match_config(self) -> None:
        config = load_module(
            "celeba_dataset_config_attr40_for_helper_test",
            CELEBA_ROOT / "dataset_config_attr40.py",
        )
        prompt = "Attractive," + "Shared_Attribute," * 8 + "Wearing_Necklace"
        helper_paths = (
            CELEBA_ROOT / "script" / "run_attr40_query100_sample_lds.py",
            CELEBA_ROOT / "script" / "plot_endpoint_top10_comparison.py",
            CELEBA_ROOT
            / "script"
            / "analyze_pint_raw_vs_adamw_topk_endpoint_clip.py",
        )
        expected = config._prompt_path_tag(prompt)
        for path in helper_paths:
            with self.subTest(path=path):
                max_length, helper = load_standalone_prompt_tag(path)
                self.assertEqual(max_length, 150)
                self.assertEqual(helper(prompt), expected)


if __name__ == "__main__":
    unittest.main()
