"""Tests for dependency-free training example transformations."""

import ast

from model.training_data import make_completion_variant


def test_make_completion_variant_preserves_indented_body() -> None:
    """A full verified function should produce a valid completion task."""
    example = {
        "instruction": "Return the square of x.",
        "input": "",
        "output": "def square(x: int) -> int:\n    \"\"\"Old docstring.\"\"\"\n    return x * x",
    }

    variant = make_completion_variant(example)

    assert variant is not None
    assert "def square(x: int) -> int:" in variant["instruction"]
    assert "Return the square of x." in variant["instruction"]
    assert variant["output"] == "    return x * x"
    # The prompt and indented completion form a complete Python function.
    prompt_code = variant["instruction"].split("\n\n", 1)[1]
    ast.parse(f"{prompt_code}\n{variant['output']}\n")


def test_make_completion_variant_rejects_multiple_functions() -> None:
    """Completion conversion avoids dropping helper-function context."""
    example = {
        "instruction": "Use a helper to return one.",
        "input": "",
        "output": "def helper():\n    return 1\n\ndef answer():\n    return helper()",
    }

    assert make_completion_variant(example) is None
