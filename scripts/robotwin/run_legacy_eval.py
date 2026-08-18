"""Run the legacy RoboTwin evaluator with configurable RGB-D settings."""

from __future__ import annotations

import ast
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _parse_positive_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


class _TestNumTransformer(ast.NodeTransformer):
    def __init__(self, test_num: int) -> None:
        self.test_num = test_num
        self.in_main = False
        self.replacements = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        was_in_main = self.in_main
        self.in_main = node.name == "main"
        node = self.generic_visit(node)
        self.in_main = was_in_main
        return node

    def visit_Assign(self, node: ast.Assign) -> Any:
        node = self.generic_visit(node)
        if self.in_main and any(
            isinstance(target, ast.Name) and target.id == "test_num"
            for target in node.targets
        ):
            node.value = ast.copy_location(ast.Constant(self.test_num), node.value)
            self.replacements += 1
        return node


def _load_legacy_evaluator(source_path: Path, test_num: int) -> dict[str, Any]:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    transformer = _TestNumTransformer(test_num)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    if transformer.replacements != 1:
        raise RuntimeError(
            "Expected exactly one test_num assignment in legacy RoboTwin main(), "
            f"found {transformer.replacements}"
        )

    namespace: dict[str, Any] = {
        "__file__": str(source_path),
        "__name__": "_robotwin_legacy_eval",
        "__package__": None,
    }
    exec(compile(tree, str(source_path), "exec"), namespace)
    return namespace


def _load_test_render(source_path: Path) -> ModuleType:
    source_dir = str(source_path.parent)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    return importlib.import_module("test_render")


def main() -> None:
    source_value = os.environ.get("ROBOTWIN_LEGACY_EVAL_SCRIPT")
    if not source_value:
        raise ValueError("ROBOTWIN_LEGACY_EVAL_SCRIPT must point to eval_policy.py")

    source_path = Path(source_value).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Legacy RoboTwin evaluator not found: {source_path}")

    test_num = _parse_positive_int("ROBOTWIN_TEST_NUM", 100)
    namespace = _load_legacy_evaluator(source_path, test_num)
    original_eval_policy = namespace["eval_policy"]

    def eval_policy_with_depth(*args: Any, **kwargs: Any):
        task_args = args[2] if len(args) > 2 else kwargs["args"]
        model = args[3] if len(args) > 3 else kwargs["model"]
        uses_depth = bool(getattr(model, "model_uses_depth", False))
        task_args.setdefault("data_type", {})["depth"] = uses_depth
        print(
            f"[TurboVLA eval] episodes={test_num}, "
            f"depth_observations={uses_depth}"
        )
        kwargs["test_num"] = test_num
        return original_eval_policy(*args, **kwargs)

    namespace["main"].__globals__["eval_policy"] = eval_policy_with_depth

    test_render = _load_test_render(source_path)
    try:
        test_render.Sapien_TEST()
    except SystemExit as exc:
        raise RuntimeError("RoboTwin rendering preflight failed") from exc
    usr_args = namespace["parse_args_and_config"]()
    namespace["main"](usr_args)


if __name__ == "__main__":
    main()
