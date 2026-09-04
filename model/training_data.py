"""Dependency-free transformations for supervised fine-tuning examples."""

from __future__ import annotations

import ast
import copy
import textwrap


def make_completion_variant(example: dict[str, str]) -> dict[str, str] | None:
    """Convert a single-function answer into a HumanEval-style body task."""
    try:
        tree = ast.parse(example["output"])
    except SyntaxError:
        return None

    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(functions) != 1:
        return None
    function = functions[0]
    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not body:
        return None

    stub = copy.deepcopy(function)
    stub.body = [ast.Expr(value=ast.Constant(value=example["instruction"].strip()))]
    context_nodes = [
        copy.deepcopy(node)
        for node in tree.body
        if node is not function
        and isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign))
    ]
    prompt_tree = ast.Module(body=[*context_nodes, stub], type_ignores=[])
    body_tree = ast.Module(body=copy.deepcopy(body), type_ignores=[])
    ast.fix_missing_locations(prompt_tree)
    ast.fix_missing_locations(body_tree)
    prompt_code = ast.unparse(prompt_tree)
    completion = textwrap.indent(ast.unparse(body_tree), "    ")
    return {
        "instruction": (
            "Complete the Python function below. Return only its indented body.\n\n"
            f"{prompt_code}"
        ),
        "input": "",
        "output": completion,
    }
